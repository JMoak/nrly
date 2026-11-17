"""Binary decision qualification, kept separate from generative-text studies."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from nrly.contracts import Contract, ModelIdentity, ProgramSpec, WeightChange, digest

Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Split = Literal["fit", "evaluation", "challenge"]
Variant = Literal["original", "patch"]
Semantics = Literal["native_binary_probability", "causal_label_likelihood"]


class DecisionRequest(Contract):
    """Only this object reaches inference. No truth, split or previous verdict."""

    state: str = Field(min_length=1, max_length=16000)
    question: str = Field(min_length=1, max_length=4000)


class DecisionCase(Contract):
    id: str = Field(min_length=1, max_length=100)
    family: str = Field(min_length=1, max_length=100)
    split: Split
    state: str = Field(min_length=1, max_length=16000)
    truth: StrictBool


class DecisionProtocol(Contract):
    schema_version: Literal["nrly-decision-protocol/1"] = "nrly-decision-protocol/1"
    question: str = Field(min_length=1, max_length=4000)
    label_origin: Literal["synthetic", "assistant_authored", "independently_reviewed"]
    evaluation_scope: Literal["exploratory"] = "exploratory"
    fixed_threshold: float = Field(default=0.5, ge=0, le=1)
    threshold_grid: tuple[float, ...] = Field(
        default=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
        min_length=1,
        max_length=101,
    )
    max_unsupported_rate: float = Field(default=0.05, ge=0, le=1)
    min_fit_acts: int = Field(default=2, ge=1, le=1000)
    min_fit_families: int = Field(default=2, ge=2, le=1000)
    max_elapsed_seconds: float = Field(default=600, gt=0, le=3600)
    cases: tuple[DecisionCase, ...] = Field(min_length=4, max_length=1000)

    @model_validator(mode="after")
    def disjoint_data(self) -> DecisionProtocol:
        if any(not 0 <= t <= 1 for t in self.threshold_grid):
            raise ValueError("thresholds must be finite probabilities")
        if tuple(sorted(set(self.threshold_grid))) != self.threshold_grid:
            raise ValueError("threshold grid must be increasing and unique")
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("case IDs must be unique")
        # Identical inputs cannot masquerade as independent observations even within a split.
        if len({c.state.strip() for c in self.cases}) != len(self.cases):
            raise ValueError("duplicate rendered states are not independent cases")
        families: dict[str, Split] = {}
        for case in self.cases:
            if families.setdefault(case.family, case.split) != case.split:
                raise ValueError("a source family cannot cross splits")
        if not {"fit", "evaluation"}.issubset({c.split for c in self.cases}):
            raise ValueError("fit and evaluation cases are required")
        fit = [c for c in self.cases if c.split == "fit"]
        if len({c.family for c in fit}) < self.min_fit_families:
            raise ValueError("not enough independent fit families")
        if {c.truth for c in fit} != {True, False}:
            raise ValueError("fitting requires both positive and negative labels")
        return self

    def request(self, case: DecisionCase) -> DecisionRequest:
        return DecisionRequest(state=case.state, question=self.question)


class DecisionExecution(Contract):
    probability: float = Field(strict=True, ge=0, le=1)
    elapsed_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    base_unchanged: Literal[True] = True
    weight_changes: tuple[WeightChange, ...] = Field(default=(), max_length=64)


class DecisionObservation(Contract):
    case_id: str
    variant: Variant
    input_digest: Hash
    program_digest: Hash
    execution: DecisionExecution | None = None
    error_code: Literal["execution_failed", "budget_exceeded", "restoration_failed"] | None = None

    @model_validator(mode="after")
    def one_outcome(self) -> DecisionObservation:
        if (self.execution is None) == (self.error_code is None):
            raise ValueError("exactly one execution or failure is required")
        return self


class FrozenRule(Contract):
    method: Literal["empirical-coverage-grid/1"] = "empirical-coverage-grid/1"
    # None means always review. It is never replaced by a permissive default.
    threshold: float | None = Field(default=None, ge=0, le=1)
    fit_digest: Hash


class DecisionFreeze(Contract):
    protocol_digest: Hash
    identity_digest: Hash
    program_digest: Hash
    semantics: Semantics
    execution_identity: Hash
    original_rule: FrozenRule
    patch_rule: FrozenRule


class DecisionRun(Contract):
    schema_version: Literal["nrly-decision-run/1"] = "nrly-decision-run/1"
    protocol: DecisionProtocol
    identity: ModelIdentity
    program: ProgramSpec
    freeze: DecisionFreeze
    elapsed_ms: float = Field(ge=0)
    budget_exceeded: StrictBool = False
    observations: tuple[DecisionObservation, ...] = Field(min_length=8, max_length=2000)


class ArmMetrics(Contract):
    variant: Variant
    rule: Literal["fixed", "fitted"]
    split: Split
    threshold: float | None
    cases: int
    families: int
    scored: int
    acts: int
    unsupported_acts: int
    coverage: float
    unsupported_rate: float | None
    agreement: float | None
    brier: float | None
    elapsed_ms: float


class DecisionAssessment(Contract):
    """Safe public projection: aggregates only, no case IDs, prompts or labels."""

    schema_version: Literal["nrly-decision-assessment/1"] = "nrly-decision-assessment/1"
    run_digest: Hash
    freeze_digest: Hash
    model_id: str
    semantics: Semantics
    label_origin: str
    status: Literal["complete", "incomplete"]
    recommendation: Literal["retain_original", "further_shadow_evaluation", "incomplete"]
    metrics: tuple[ArmMetrics, ...]
    rescued_supported_acts: int
    introduced_unsupported_acts: int
    lost_supported_acts: int
    max_unsupported_rate: float
    scope: Literal["exploratory; no promotion authorization"] = (
        "exploratory; no promotion authorization"
    )


def original_program(identity: ModelIdentity) -> ProgramSpec:
    return ProgramSpec(expected_base=identity.fingerprint)


def observation_digest(observations: tuple[DecisionObservation, ...]) -> str:
    return digest({"observations": [o.model_dump(mode="json") for o in observations]})
