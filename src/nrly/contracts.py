from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    field_validator,
    model_validator,
)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelSpec(Contract):
    backend: Literal["tiny", "huggingface"] = "tiny"
    model_id: str = "nrly/tiny-sum-v1"
    revision: str = "1"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    local_files_only: bool = True

    @model_validator(mode="after")
    def immutable_revision(self) -> ModelSpec:
        if self.backend == "huggingface" and not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("huggingface revision must be an exact 40-character commit digest")
        if self.backend == "tiny" and (self.model_id, self.revision) != ("nrly/tiny-sum-v1", "1"):
            raise ValueError("unknown tiny model identity")
        return self


class LocalModelSpec(Contract):
    """An already constructed trusted Python model; never accepted by ExperimentSpec."""

    backend: Literal["local_torch"] = "local_torch"
    model_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(min_length=1, max_length=200)


class GenerationSpec(Contract):
    max_new_tokens: int = Field(default=32, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0, le=2)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    enable_thinking: bool = False
    top_p: float | None = Field(default=None, gt=0, le=1, exclude_if=lambda v: v is None)
    top_k: int | None = Field(default=None, ge=0, le=1000, exclude_if=lambda v: v is None)
    min_new_tokens: int | None = Field(default=None, ge=0, le=32768, exclude_if=lambda v: v is None)
    repetition_penalty: float | None = Field(
        default=None, gt=0, le=10, exclude_if=lambda v: v is None
    )

    @model_validator(mode="after")
    def valid_generation(self) -> GenerationSpec:
        if self.min_new_tokens is not None and self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens cannot exceed max_new_tokens")
        return self


class ChatMessage(Contract):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100000)


class TaskAdapterSpec(Contract):
    schema_version: Literal["nrly-tasks/1"] = "nrly-tasks/1"
    kind: Literal["multiple_choice", "json", "instruction_constraints"]
    choices: list[str] = Field(default_factory=list, max_length=26)
    required: list[str] = Field(default_factory=list, max_length=16)
    forbidden: list[str] = Field(default_factory=list, max_length=16)
    max_chars: int | None = Field(default=None, ge=1, le=1000000)

    @model_validator(mode="after")
    def valid_adapter(self) -> TaskAdapterSpec:
        for values, limit in ((self.choices, 32), (self.required, 1000), (self.forbidden, 1000)):
            if any(not value.strip() or len(value) > limit for value in values):
                raise ValueError("adapter labels and literals must be nonempty and bounded")
            if len(set(values)) != len(values):
                raise ValueError("adapter labels and literals must be unique")
        if self.kind == "multiple_choice":
            if not self.choices or self.required or self.forbidden or self.max_chars is not None:
                raise ValueError("multiple_choice requires only choices")
            if any(value != value.strip() for value in self.choices):
                raise ValueError("choice labels cannot have surrounding whitespace")
        elif self.kind == "instruction_constraints":
            if self.choices or not (self.required or self.forbidden or self.max_chars is not None):
                raise ValueError("instruction_constraints requires literal or length constraints")
            if set(self.required) & set(self.forbidden):
                raise ValueError("a literal cannot be both required and forbidden")
        elif self.choices or self.required or self.forbidden or self.max_chars is not None:
            raise ValueError("json adapter does not accept constraint fields")
        return self


class DiagnosticSpec(Contract):
    top_k: int = Field(default=5, ge=0, le=50)
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    max_logits_elements: int = Field(
        default=67108864, ge=1, le=268435456, exclude_if=lambda v: v == 67108864
    )


class TensorSlice(Contract):
    start: StrictInt = Field(default=0, ge=0, le=2**63 - 1)
    stop: StrictInt = Field(ge=1, le=2**63 - 1)
    step: StrictInt = Field(default=1, ge=1, le=2**63 - 1)

    @model_validator(mode="after")
    def nonempty(self) -> TensorSlice:
        if self.start >= self.stop:
            raise ValueError("tensor slices must be nonempty, forward ranges")
        return self


WeightKind = Literal[
    "weight_scale",
    "weight_replace",
    "weight_patch",
    "weight_add",
    "weight_sparse_add",
    "weight_low_rank",
    "weight_slice_scale",
]


class _OperationFields(Contract):
    target: str | None = None
    factor: float = 1.0
    values: list[float] = Field(default_factory=list, max_length=65536)
    shape: list[StrictInt] = Field(default_factory=list, max_length=8)
    indices: list[StrictInt] = Field(default_factory=list, max_length=65536)
    position: Literal["all", "last"] = "all"
    rank: StrictInt | None = Field(default=None, ge=1, le=256, exclude_if=lambda v: v is None)
    left: list[float] = Field(default_factory=list, max_length=65536, exclude_if=lambda v: not v)
    right: list[float] = Field(default_factory=list, max_length=65536, exclude_if=lambda v: not v)
    slices: list[TensorSlice] = Field(
        default_factory=list, max_length=8, exclude_if=lambda v: not v
    )

    def _validate_operation(self, kind: str) -> None:
        if kind != "identity" and not self.target:
            raise ValueError("an intervention requires an exact target")
        if kind == "identity" and (self.target or self.values or self.shape or self.indices):
            raise ValueError("identity cannot specify an intervention")
        if not math.isfinite(self.factor) or any(
            not math.isfinite(x) for values in (self.values, self.left, self.right) for x in values
        ):
            raise ValueError("intervention values must be finite")
        if any(x <= 0 or x > 2**31 - 1 for x in self.shape) or math.prod(self.shape) > 2**63 - 1:
            raise ValueError("tensor shape must have positive bounded dimensions and size")
        if len(self.values) + len(self.left) + len(self.right) > 65536:
            raise ValueError("operation payload exceeds 65536 values")
        if kind != "weight_low_rank" and (self.rank is not None or self.left or self.right):
            raise ValueError("only weight_low_rank accepts factor matrices")
        if kind != "weight_slice_scale" and self.slices:
            raise ValueError("only weight_slice_scale accepts tensor slices")
        if kind in {"weight_patch", "weight_sparse_add"}:
            if not self.shape or not self.indices or len(self.indices) != len(self.values):
                raise ValueError(
                    "sparse patches require a full shape and equal nonempty indices/values"
                )
            if len(set(self.indices)) != len(self.indices):
                raise ValueError("patch indices must be unique")
            if any(index < 0 or index >= math.prod(self.shape) for index in self.indices):
                raise ValueError("patch index is outside the declared tensor shape")
        elif self.indices:
            raise ValueError("only sparse weight operations accept indices")
        elif kind == "weight_slice_scale":
            if not self.shape or len(self.slices) != len(self.shape) or self.values:
                raise ValueError("slice scaling requires a full shape and one slice per axis")
            if any(part.stop > size for part, size in zip(self.slices, self.shape, strict=True)):
                raise ValueError("weight slice is outside the declared tensor shape")
        elif kind == "weight_low_rank":
            if (
                len(self.shape) != 2
                or self.rank is None
                or len(self.left) != self.shape[0] * self.rank
                or len(self.right) != self.rank * self.shape[1]
                or self.values
            ):
                raise ValueError(
                    "low-rank edits require shape [rows, columns], rank, left and right"
                )
        elif kind in {"activation_add", "weight_replace", "weight_add"}:
            if not self.shape or math.prod(self.shape) != len(self.values):
                raise ValueError("values must match an explicit tensor shape")
        elif self.values or self.shape:
            raise ValueError("this operation does not take a tensor payload")
        if kind.startswith("weight_") and self.position != "all":
            raise ValueError("weight operations do not have token positions")


class WeightOperationSpec(_OperationFields):
    kind: WeightKind

    @model_validator(mode="after")
    def valid_operation(self) -> WeightOperationSpec:
        self._validate_operation(self.kind)
        return self


class ProgramSpec(_OperationFields):
    kind: Literal[
        "identity",
        "activation_scale",
        "activation_add",
        "weight_scale",
        "weight_replace",
        "weight_patch",
        "weight_add",
        "weight_sparse_add",
        "weight_low_rank",
        "weight_slice_scale",
        "pipeline",
    ] = "identity"
    expected_base: str | None = None
    operations: list[WeightOperationSpec] = Field(
        default_factory=list, max_length=32, exclude_if=lambda v: not v
    )

    @model_validator(mode="after")
    def valid_operation(self) -> ProgramSpec:
        if self.kind == "pipeline":
            if not self.operations or not self.expected_base:
                raise ValueError("pipelines require operations and an exact expected_base")
            if (
                self.target is not None
                or self.factor != 1
                or self.values
                or self.shape
                or self.indices
                or self.position != "all"
                or self.rank is not None
                or self.left
                or self.right
                or self.slices
            ):
                raise ValueError("pipeline payload belongs in its ordered operations")
            if sum(len(op.values) + len(op.left) + len(op.right) for op in self.operations) > 65536:
                raise ValueError("pipeline payload exceeds 65536 values")
        else:
            if self.operations:
                raise ValueError("only pipelines accept operations")
            self._validate_operation(self.kind)
        return self


class CaseSpec(Contract):
    id: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    input: str | None = Field(
        default=None, min_length=1, max_length=100000, exclude_if=lambda v: v is None
    )
    expected: str | None = None
    scorer: Literal["exact", "numeric", "final_number"] = "exact"
    tolerance: float = Field(default=1e-6, ge=0, le=1)
    split: Literal["discovery", "validation", "test"] = "discovery"
    role: Literal["target", "regression"] = "target"
    messages: list[ChatMessage] = Field(
        default_factory=list, max_length=128, exclude_if=lambda v: not v
    )
    continuation: str | None = Field(
        default=None, min_length=1, max_length=100000, exclude_if=lambda v: v is None
    )
    group: str | None = Field(
        default=None, min_length=1, max_length=120, exclude_if=lambda v: v is None
    )
    adapter: TaskAdapterSpec | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def valid_case(self) -> CaseSpec:
        if (self.input is not None) == bool(self.messages):
            raise ValueError("provide exactly one of input or messages")
        if sum(len(message.content) for message in self.messages) > 100000:
            raise ValueError("message content exceeds 100000 characters")
        if self.adapter is not None:
            if self.scorer != "exact" or self.tolerance != 1e-6:
                raise ValueError("task adapters cannot be combined with a legacy scorer")
            if self.adapter.kind == "multiple_choice" and self.expected not in self.adapter.choices:
                raise ValueError("multiple_choice expected must be a registered choice")
        return self


class CaptureSpec(Contract):
    target: str = Field(min_length=1, max_length=200)
    max_elements: int = Field(default=16, ge=1, le=4096)
    token_positions: list[StrictInt] = Field(
        default_factory=list, max_length=128, exclude_if=lambda v: not v
    )
    invocation: Literal["first", "latest"] = Field(
        default="latest", exclude_if=lambda v: v == "latest"
    )
    source: Literal["input", "output"] = Field(default="output", exclude_if=lambda v: v == "output")

    @model_validator(mode="after")
    def valid_capture(self) -> CaptureSpec:
        if any(position < 0 or position > 2**31 - 1 for position in self.token_positions):
            raise ValueError("token positions must be bounded nonnegative integers")
        if len(set(self.token_positions)) != len(self.token_positions):
            raise ValueError("token positions must be unique")
        return self


class WeightReadSpec(CaptureSpec):
    offset: StrictInt = Field(default=0, ge=0, le=2**63 - 1, exclude_if=lambda v: v == 0)
    indices: list[StrictInt] = Field(
        default_factory=list, max_length=4096, exclude_if=lambda v: not v
    )
    coordinates: list[list[StrictInt]] = Field(
        default_factory=list, max_length=4096, exclude_if=lambda v: not v
    )
    slices: list[TensorSlice] = Field(
        default_factory=list, max_length=8, exclude_if=lambda v: not v
    )

    @model_validator(mode="after")
    def valid_selection(self) -> WeightReadSpec:
        if self.token_positions or self.invocation != "latest" or self.source != "output":
            raise ValueError("activation capture selectors do not apply to weight reads")
        if sum(bool(v) for v in (self.offset, self.indices, self.coordinates, self.slices)) > 1:
            raise ValueError("choose exactly one of offset, indices, coordinates, or slices")
        if len(self.indices) > self.max_elements or len(self.coordinates) > self.max_elements:
            raise ValueError("selected values exceed max_elements")
        if any(x < 0 or x > 2**63 - 1 for x in self.indices):
            raise ValueError("tensor indices must be nonnegative bounded integers")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("tensor indices must be unique")
        if any(
            not coordinate or len(coordinate) > 8 or any(x < 0 or x > 2**63 - 1 for x in coordinate)
            for coordinate in self.coordinates
        ):
            raise ValueError("coordinates require 1..8 nonnegative bounded integers")
        if len({tuple(c) for c in self.coordinates}) != len(self.coordinates):
            raise ValueError("tensor coordinates must be unique")
        return self


class BudgetSpec(Contract):
    wall_seconds: float = Field(default=120, gt=0, le=86400)
    max_cases: int = Field(default=100, ge=1, le=10000)
    max_output_tokens: int = Field(default=4096, ge=1, le=1000000)
    max_capture_bytes: int = Field(default=262144, ge=0, le=100000000)


class ExperimentSpec(Contract):
    schema_version: Literal["0.1", "0.2", "0.3"] = "0.2"
    name: str = Field(min_length=1, max_length=160)
    purpose: Literal["experiment", "inspection"] = "experiment"
    model: ModelSpec = Field(default_factory=ModelSpec)
    program: ProgramSpec = Field(default_factory=ProgramSpec)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    cases: Annotated[list[CaseSpec], Field(max_length=10000)] = Field(default_factory=list)
    captures: list[CaptureSpec] = Field(default_factory=list, max_length=8)
    weight_reads: list[WeightReadSpec] = Field(default_factory=list, max_length=8)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    diagnostics: DiagnosticSpec | None = Field(default=None, exclude_if=lambda v: v is None)

    @field_validator("weight_reads", mode="before")
    @classmethod
    def legacy_reads(cls, value: object) -> object:
        if isinstance(value, list):
            return [item.model_dump() if isinstance(item, CaptureSpec) else item for item in value]
        return value

    @model_validator(mode="after")
    def unique_cases(self) -> ExperimentSpec:
        extended = (
            self.diagnostics is not None
            or self.generation.max_new_tokens > 512
            or self.generation.top_p is not None
            or self.generation.top_k is not None
            or self.generation.min_new_tokens is not None
            or self.generation.repetition_penalty is not None
            or any(
                c.messages or c.continuation is not None or c.group or c.adapter for c in self.cases
            )
            or any(
                c.token_positions or c.invocation != "latest" or c.source != "output"
                for c in self.captures
            )
        )
        if extended and self.schema_version != "0.3":
            raise ValueError("extended runtime and task features require schema_version 0.3")
        if any(
            (case.continuation is not None) != (self.diagnostics is not None) for case in self.cases
        ):
            raise ValueError(
                "diagnostics and a continuation on every case must be provided together"
            )
        if self.diagnostics is not None and (self.purpose != "experiment" or not self.cases):
            raise ValueError("diagnostics require experiment cases")
        if self.schema_version == "0.1" and (
            self.program.kind
            in {
                "pipeline",
                "weight_add",
                "weight_sparse_add",
                "weight_low_rank",
                "weight_slice_scale",
            }
            or any(
                read.offset or read.indices or read.coordinates or read.slices
                for read in self.weight_reads
            )
        ):
            raise ValueError("extended weight programs and reads require schema_version 0.2")
        if self.purpose == "experiment" and not self.cases:
            raise ValueError("experiments require at least one case")
        if self.purpose == "inspection" and (
            self.cases or self.program.kind != "identity" or self.captures
        ):
            raise ValueError(
                "inspections cannot execute cases, interventions, or activation captures"
            )
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("case ids must be unique")
        if len(self.cases) > self.budget.max_cases:
            raise ValueError("case count exceeds the experiment budget")
        if len({capture.target for capture in self.captures}) != len(self.captures):
            raise ValueError("capture targets must be unique")
        if len({read.target for read in self.weight_reads}) != len(self.weight_reads):
            raise ValueError("weight read targets must be unique")
        return self


class ModelIdentity(Contract):
    backend: str
    model_id: str
    revision: str
    fingerprint: str
    details: dict[str, JsonValue] = Field(default_factory=dict)


class TensorInfo(Contract):
    name: str
    kind: Literal["parameter", "module"]
    shape: list[int] = Field(default_factory=list)
    dtype: str | None = None


class TensorCapture(Contract):
    target: str
    shape: list[int]
    dtype: str
    values: list[float]
    truncated: bool
    flat_indices: list[int] = Field(
        default_factory=list, max_length=4096, exclude_if=lambda v: not v
    )
    coordinates: list[list[int]] = Field(
        default_factory=list, max_length=4096, exclude_if=lambda v: not v
    )
    selection_shape: list[int] = Field(
        default_factory=list, max_length=8, exclude_if=lambda v: not v
    )
    statistics: TensorStatistics | None = Field(default=None, exclude_if=lambda v: v is None)
    tensor_sha256: str | None = Field(default=None, exclude_if=lambda v: v is None)
    token_positions: list[int] = Field(
        default_factory=list, max_length=128, exclude_if=lambda v: not v
    )
    invocation_index: int | None = Field(default=None, ge=0, exclude_if=lambda v: v is None)
    source: Literal["input", "output"] | None = Field(default=None, exclude_if=lambda v: v is None)


class TensorStatistics(Contract):
    scope: Literal["selection"] = "selection"
    count: int = Field(ge=0)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = None
    rms: float | None = None
    zero_count: int = Field(ge=0)


class ModelInspection(Contract):
    tensors: list[TensorInfo] = Field(default_factory=list)
    weights: list[TensorCapture] = Field(default_factory=list)


class WeightChange(Contract):
    target: str
    shape: list[int]
    dtype: str
    before_sha256: str
    after_sha256: str
    changed_elements: int = Field(ge=0)
    total_elements: int = Field(ge=0)
    before_preview: list[float] = Field(default_factory=list, max_length=32)
    after_preview: list[float] = Field(default_factory=list, max_length=32)
    edited_indices: list[int] = Field(default_factory=list, max_length=32)
    edited_before: list[float] = Field(default_factory=list, max_length=32)
    edited_after: list[float] = Field(default_factory=list, max_length=32)
    operation_index: int | None = Field(default=None, ge=0, le=31, exclude_if=lambda v: v is None)
    operation_kind: WeightKind | None = Field(default=None, exclude_if=lambda v: v is None)
    edited_coordinates: list[list[int]] = Field(
        default_factory=list, max_length=32, exclude_if=lambda v: not v
    )
    net_changed_elements: int | None = Field(default=None, ge=0, exclude_if=lambda v: v is None)
    net_before_sha256: str | None = Field(default=None, exclude_if=lambda v: v is None)
    net_after_sha256: str | None = Field(default=None, exclude_if=lambda v: v is None)


class TokenAlternative(Contract):
    token_id: int = Field(ge=0)
    text: str
    logprob: float = Field(le=0)


class TokenObservation(TokenAlternative):
    position: int = Field(ge=1)
    rank: int = Field(ge=1)
    top_k: list[TokenAlternative] = Field(default_factory=list, max_length=50)


class ContinuationScore(Contract):
    mode: Literal["teacher_forced"] = "teacher_forced"
    tokenization: Literal["separate_prompt_and_continuation"] = "separate_prompt_and_continuation"
    prompt_tokens: int = Field(ge=1)
    continuation_tokens: int = Field(ge=1, le=32768)
    sum_logprob: float = Field(le=0)
    mean_logprob: float = Field(le=0)
    tokens: list[TokenObservation] = Field(min_length=1, max_length=32768)

    @model_validator(mode="after")
    def consistent_observations(self) -> ContinuationScore:
        if len(self.tokens) != self.continuation_tokens:
            raise ValueError("continuation token count must match observations")
        if [token.position for token in self.tokens] != list(
            range(self.prompt_tokens, self.prompt_tokens + self.continuation_tokens)
        ):
            raise ValueError(
                "continuation observations require contiguous absolute token positions"
            )
        total = math.fsum(token.logprob for token in self.tokens)
        if not math.isclose(
            self.sum_logprob, total, rel_tol=1e-7, abs_tol=1e-7
        ) or not math.isclose(
            self.mean_logprob, total / self.continuation_tokens, rel_tol=1e-7, abs_tol=1e-7
        ):
            raise ValueError("aggregate log probabilities must match observations")
        return self


class BackendResult(Contract):
    output: str
    elapsed_ms: float = Field(ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None
    captures: list[TensorCapture] = Field(default_factory=list)
    finish_reason: str = "completed"
    base_unchanged: bool = True
    weight_changes: list[WeightChange] = Field(default_factory=list)
    diagnostics: ContinuationScore | None = Field(default=None, exclude_if=lambda v: v is None)
    truncated: bool | None = Field(default=None, exclude_if=lambda v: v is None)


class ErrorInfo(Contract):
    code: str
    message: str


class SampleResult(Contract):
    case_id: str
    status: Literal["completed", "failed"]
    provenance: Literal["torch_fixture", "huggingface", "unavailable"]
    output: str | None = None
    elapsed_ms: float = Field(default=0, ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    captures: list[TensorCapture] = Field(default_factory=list)
    error: ErrorInfo | None = None
    weight_changes: list[WeightChange] = Field(default_factory=list)
    diagnostics: ContinuationScore | None = Field(default=None, exclude_if=lambda v: v is None)
    truncated: bool | None = Field(default=None, exclude_if=lambda v: v is None)


RunStatus = Literal[
    "queued", "running", "cancelling", "succeeded", "failed", "cancelled", "interrupted"
]


class RunRecord(Contract):
    schema_version: Literal["0.1", "0.2", "0.3"] = "0.2"
    id: str
    status: RunStatus
    spec: ExperimentSpec
    spec_digest: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    model_identity: ModelIdentity | None = None
    inspection: ModelInspection | None = None
    samples: list[SampleResult] = Field(default_factory=list)
    error: ErrorInfo | None = None
    elapsed_ms: float = 0
    capture_bytes: int = 0
    output_tokens: int = 0
    parent_run: str | None = None


class NrlyError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)

    def as_info(self) -> ErrorInfo:
        return ErrorInfo(code=self.code, message=str(self))


def canonical_json(value: BaseModel | dict[str, JsonValue] | list[JsonValue]) -> str:
    data = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: BaseModel | dict[str, JsonValue] | list[JsonValue]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
