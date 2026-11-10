"""Public contracts for controlled studies; sealed cases never enter these records."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from nrly.contracts import (
    BudgetSpec,
    CaseSpec,
    Contract,
    ExperimentSpec,
    GenerationSpec,
    ModelIdentity,
    ModelSpec,
    ProgramSpec,
)


class CandidateSpec(Contract):
    name: str = Field(min_length=1, max_length=120)
    program: ProgramSpec

    @model_validator(mode="after")
    def weight_program(self) -> CandidateSpec:
        if self.program.kind not in {"identity", "pipeline"} and not self.program.kind.startswith(
            "weight_"
        ):
            raise ValueError("study candidates and controls must be weight programs or identity")
        return self


class ComparatorSpec(Contract):
    """Operator-registered alternative; an external checkpoint may be fine-tuned elsewhere."""

    name: str = Field(min_length=1, max_length=120)
    kind: Literal["prompt", "model"]
    prefix: str = Field(default="", max_length=10000)
    suffix: str = Field(default="", max_length=10000)
    model: ModelSpec | None = None

    @model_validator(mode="after")
    def valid_comparator(self) -> ComparatorSpec:
        if self.kind == "prompt":
            if self.model is not None or not (self.prefix or self.suffix):
                raise ValueError("prompt comparators require prefix/suffix and no model override")
        elif self.model is None or self.prefix or self.suffix:
            raise ValueError("model comparators require only a pinned model")
        return self

    def transform_cases(self, cases: list[CaseSpec]) -> list[CaseSpec]:
        if self.kind != "prompt":
            return cases
        transformed = []
        for case in cases:
            data = case.model_dump(mode="json")
            if case.messages:
                if case.messages[-1].role != "user":
                    raise ValueError("prompt comparator requires a final user message")
                data["messages"][-1]["content"] = (
                    self.prefix + case.messages[-1].content + self.suffix
                )
            else:
                data["input"] = self.prefix + (case.input or "") + self.suffix
            transformed.append(CaseSpec.model_validate(data))
        return transformed


class StudyBudget(Contract):
    """Admission ceilings, charged without refunds, including unsuccessful attempts."""

    max_attempts: int = Field(default=10, ge=1, le=1000)
    max_runs: int = Field(default=100, ge=1, le=10000)
    max_case_executions: int = Field(default=10000, ge=1, le=1000000)
    max_output_tokens: int = Field(default=100000, ge=1, le=100000000)
    max_capture_bytes: int = Field(default=100000000, ge=0, le=1000000000)
    max_wall_seconds: float = Field(default=7200, gt=0, le=864000)


class StudyUsage(Contract):
    attempts: int = 0
    runs: int = 0
    case_executions: int = 0
    output_tokens: int = 0
    capture_bytes: int = 0
    wall_seconds: float = 0
    final_runs_reserved: int = 0


class StudySpec(Contract):
    schema_version: Literal["0.1", "0.2"] = "0.1"
    name: str = Field(min_length=1, max_length=160)
    model: ModelSpec = Field(default_factory=ModelSpec)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    development_cases: list[CaseSpec] = Field(min_length=1, max_length=10000)
    controls: list[CandidateSpec] = Field(default_factory=list, max_length=8)
    seeds: list[Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]] | None = Field(
        default=None, min_length=1, max_length=32, exclude_if=lambda v: v is None
    )
    comparators: list[ComparatorSpec] = Field(
        default_factory=list, max_length=8, exclude_if=lambda v: not v
    )
    mutable_targets: list[str] | None = Field(default=None, min_length=1, max_length=128)
    run_budget: BudgetSpec = Field(default_factory=BudgetSpec)
    budget: StudyBudget = Field(default_factory=StudyBudget)

    @model_validator(mode="after")
    def pinned_design(self) -> StudySpec:
        if self.schema_version == "0.1" and (
            self.seeds is not None
            or self.comparators
            or self.generation.temperature != 0
            or any(
                case.adapter is not None or case.group is not None
                for case in self.development_cases
            )
        ):
            raise ValueError("sampling, comparators, and task adapters require study schema 0.2")
        if self.seeds is not None and len(set(self.seeds)) != len(self.seeds):
            raise ValueError("study seeds must be unique")
        if self.generation.temperature != 0 and self.seeds is None:
            raise ValueError("sampled studies require explicit preregistered seeds")
        if len({item.name for item in self.comparators}) != len(self.comparators):
            raise ValueError("comparator names must be unique")
        if any(
            not has_scoring_contract(case) or case.split == "test"
            for case in self.development_cases
        ):
            raise ValueError("development cases require expectations and a non-test split")
        if len({case.group for case in self.development_cases if case.group is not None}) > 32:
            raise ValueError("studies support at most 32 registered groups per case set")
        if len({case.id for case in self.development_cases}) != len(self.development_cases):
            raise ValueError("development case ids must be unique")
        if len(self.development_cases) > self.run_budget.max_cases:
            raise ValueError("development cases exceed the per-run budget")
        if not any(case.role == "target" for case in self.development_cases):
            raise ValueError("development cases require at least one target")
        if not any(case.role == "regression" for case in self.development_cases):
            raise ValueError("development cases require at least one regression case")
        names = [control.name for control in self.controls]
        if len(set(names)) != len(names):
            raise ValueError("control names must be unique")
        if self.mutable_targets is not None:
            if len(set(self.mutable_targets)) != len(self.mutable_targets):
                raise ValueError("mutable targets must be unique exact tensor names")
            for control in self.controls:
                validate_targets(control, self.mutable_targets)
        # Reuse actual run validation at admission, including schema and generation limits.
        for comparator in [None, *self.comparators]:
            ExperimentSpec(
                schema_version="0.3" if self.schema_version == "0.2" else "0.2",
                name="study admission validation",
                model=comparator.model if comparator and comparator.model else self.model,
                cases=comparator.transform_cases(self.development_cases)
                if comparator
                else self.development_cases,
                generation=self.generation,
                budget=self.run_budget,
            )
        return self


def has_scoring_contract(case: CaseSpec) -> bool:
    return case.expected is not None or (
        case.adapter is not None and case.adapter.kind == "instruction_constraints"
    )


def study_seeds(spec: StudySpec) -> list[int]:
    return spec.seeds if spec.seeds is not None else [spec.generation.seed]


def study_arm_name(spec: StudySpec, name: str, seed: int) -> str:
    return name if spec.seeds is None else f"{name}@seed={seed}"


def study_arm_names(spec: StudySpec) -> list[str]:
    return [
        "baseline",
        "no_op",
        *("control:" + c.name for c in spec.controls),
        *("comparator:" + c.name for c in spec.comparators),
        "candidate",
    ]


def validate_targets(candidate: CandidateSpec, targets: list[str] | None) -> None:
    if targets is None:
        return
    operations = (
        candidate.program.operations
        if candidate.program.kind == "pipeline"
        else [candidate.program]
    )
    if any(op.kind != "identity" and op.target not in targets for op in operations):
        raise ValueError("program target is outside the preregistered mutable interface")


class HoldoutCommitment(Contract):
    digest: str
    count: int
    targets: int
    regressions: int


class StudyGroupSummary(Contract):
    group: str
    total: int
    passed: int
    unscored: int


class SeedVariation(Contract):
    """Descriptive paired variation, never an independence or population CI claim."""

    seed_count: int
    cases_per_seed: int
    paired_executions: int
    mean_delta: float | None
    sample_stddev: float | None
    minimum_delta: float | None
    maximum_delta: float | None
    interpretation: Literal["descriptive_across_registered_seeds_not_population_ci"] = (
        "descriptive_across_registered_seeds_not_population_ci"
    )


class PairedGroupSummary(Contract):
    group: str
    improved: int
    regressed: int
    unchanged: int
    variation: SeedVariation


class ArmSummary(Contract):
    arm: str
    status: str
    complete: bool
    total: int
    passed: int
    target_total: int
    target_passed: int
    regression_total: int
    regression_passed: int
    seed: int | None = Field(default=None, exclude_if=lambda v: v is None)
    groups: list[StudyGroupSummary] = Field(default_factory=list, exclude_if=lambda v: not v)


class PairedSummary(Contract):
    reference: str
    comparable: bool
    improved_targets: int = 0
    regressed_targets: int = 0
    improved_regressions: int = 0
    regressed_regressions: int = 0
    target_variation: SeedVariation | None = Field(default=None, exclude_if=lambda v: v is None)
    regression_variation: SeedVariation | None = Field(default=None, exclude_if=lambda v: v is None)
    groups: list[PairedGroupSummary] = Field(default_factory=list, exclude_if=lambda v: not v)


class StudyAssessment(Contract):
    scorer_version: Literal["nrly-scorers/2"] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    verdict: Literal["observed_improvement", "no_change", "regressed", "inconclusive"]
    recommendation: Literal["retain_original", "candidate_supported"] = "retain_original"
    scope: Literal["engineering_fixture", "real_model"]
    arms: list[ArmSummary] = Field(default_factory=list)
    comparisons: list[PairedSummary] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class StudyRecord(Contract):
    schema_version: Literal["0.1"] = "0.1"
    id: str
    status: Literal["open", "finalizing", "completed", "interrupted"] = "open"
    spec: StudySpec
    spec_digest: str
    holdout: HoldoutCommitment
    created_at: str
    usage: StudyUsage
    model_identity: ModelIdentity | None = None
    comparator_identities: dict[str, ModelIdentity] = Field(
        default_factory=dict, exclude_if=lambda v: not v
    )
    nominee_attempt_id: str | None = None
    nominee_digest: str | None = None
    finished_at: str | None = None
    final_result: StudyAssessment | None = None
    final_error_code: str | None = None


class AttemptRecord(Contract):
    id: str
    study_id: str
    sequence: int
    candidate: CandidateSpec
    candidate_digest: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]
    created_at: str
    finished_at: str | None = None
    assessment: StudyAssessment | None = None
    error_code: str | None = None


class StudyEvent(Contract):
    sequence: int
    study_id: str
    attempt_id: str | None = None
    kind: str
    created_at: str
    detail_digest: str
    previous_digest: str
    event_digest: str
