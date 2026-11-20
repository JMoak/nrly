"""Public records for the lever map: tested sites in a model's weights, not meanings.

A lever is a place in one exact checkpoint, a direction found by a re-runnable recipe,
and what happened when that direction was edited. Names come from the test and the
site, never from an interpretation. Anything inherited from a feature atlas or claimed
by an author is ``pencil``. An entry becomes ``inked`` only through a registered
study verdict, and a refuted entry stays on the map as ``struck``.

Entries pin weights (``PortableBase``), never a machine, so they can travel. Evidence
inside an entry records the runtime that produced it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from nrly.artifacts import TensorArtifact
from nrly.contracts import (
    CaseSpec,
    Contract,
    GenerationSpec,
    PortableBase,
    WeightOperationSpec,
    digest,
)
from nrly.study_contracts import (
    HoldoutCommitment,
    InferenceRule,
    PairedInference,
    StudyAssessment,
)

_DIGEST = r"^[0-9a-f]{64}$"
_SLUG = r"^[a-z0-9][a-z0-9._+-]{1,79}$"
LeverStatus = Literal["pencil", "inked", "struck"]


class LeverSite(Contract):
    """One edited tensor and the module whose activation carries the direction."""

    parameter: str = Field(min_length=1, max_length=200)
    module: str = Field(min_length=1, max_length=200)
    side: Literal["output", "input"] = "output"
    layer: int | None = Field(default=None, ge=0, le=4095)


class RecipeSpec(Contract):
    """A re-runnable derivation; coordinates expire with a checkpoint, recipes do not.

    ``positive`` prompts exhibit the behaviour and ``negative`` prompts do not. Even
    positions of each list fit a direction and odd positions score it, so the site is
    never chosen on the data that produced its direction.
    """

    schema_version: Literal["nrly-recipe/1"] = "nrly-recipe/1"
    kind: Literal["difference_of_means"] = "difference_of_means"
    behavior: str = Field(pattern=_SLUG)
    positive: list[CaseSpec] = Field(min_length=4, max_length=2048)
    negative: list[CaseSpec] = Field(min_length=4, max_length=2048)
    sites: list[LeverSite] = Field(min_length=1, max_length=64)
    selection: Literal["held_out_separation"] = "held_out_separation"
    position: Literal["last_prompt_token"] = "last_prompt_token"
    enable_thinking: bool = False
    edit: Literal["remove_output_direction"] = "remove_output_direction"

    @model_validator(mode="after")
    def valid_recipe(self) -> RecipeSpec:
        identifiers = [case.id for case in (*self.positive, *self.negative)]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("contrast prompt ids must be unique across both sets")
        if any(case.continuation is not None for case in (*self.positive, *self.negative)):
            raise ValueError("contrast prompts cannot carry a teacher-forced continuation")
        if len({site.parameter for site in self.sites}) != len(self.sites):
            raise ValueError("candidate sites must name distinct tensors")
        if any(site.side != "output" for site in self.sites):
            raise ValueError("remove_output_direction requires output-side sites")
        return self


class SiteScore(Contract):
    """How well one candidate site separated the held-out contrast prompts."""

    site: LeverSite
    # (mean positive projection - mean negative projection) / pooled standard deviation,
    # measured on the selection half against the direction fitted on the other half.
    separation: float
    difference_norm: float = Field(ge=0)
    fit_cases: int = Field(ge=2)
    selection_cases: int = Field(ge=2)


class Derivation(Contract):
    """One execution of a recipe on one exact checkpoint, including the sites not chosen."""

    schema_version: Literal["nrly-derivation/1"] = "nrly-derivation/1"
    recipe: RecipeSpec
    recipe_digest: str = Field(pattern=_DIGEST)
    base: PortableBase
    runtime_fingerprint: str = Field(pattern=_DIGEST)
    chosen: LeverSite
    # Unit vector in the chosen module's activation space, refitted on every prompt.
    direction: TensorArtifact
    scores: list[SiteScore] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def consistent(self) -> Derivation:
        if digest(self.recipe) != self.recipe_digest:
            raise ValueError("recipe digest does not match the recorded recipe")
        if [score.site for score in self.scores] != self.recipe.sites:
            raise ValueError("every candidate site must be scored, in recipe order")
        if self.chosen not in self.recipe.sites:
            raise ValueError("the chosen site must be one of the recipe's candidates")
        if len(self.direction.shape) != 1:
            raise ValueError("a direction must be a vector artifact")
        return self


class DosePoint(Contract):
    """One tested strength, compared with the unedited original on development cases."""

    strength: float = Field(ge=-16, le=16)
    operations_digest: str = Field(pattern=_DIGEST)
    inference: PairedInference
    target_rate: float = Field(ge=0, le=1)
    regression_rate: float = Field(ge=0, le=1)
    finish_rate: float | None = Field(default=None, ge=0, le=1)
    accuracy_when_finished: float | None = Field(default=None, ge=0, le=1)
    # Target gain divided by the larger of the registered regression margin and the
    # observed regression cost. None when that denominator is zero.
    selectivity: float | None = None


class StrengthWindow(Contract):
    """Longest run of adjacent tested strengths that help targets and spare regressions."""

    low: float
    high: float
    points: int = Field(ge=1)


class Dial(Contract):
    """Dose-response on development cases. Exploratory: it selects, it never proves."""

    schema_version: Literal["nrly-dial/1"] = "nrly-dial/1"
    evidence: Literal["exploratory_development"] = "exploratory_development"
    rule: InferenceRule
    truncation: Literal["unscored", "fail"] = "unscored"
    generation: GenerationSpec
    seeds: list[int] = Field(min_length=1, max_length=32)
    cases: int = Field(ge=1)
    runtime_fingerprint: str = Field(pattern=_DIGEST)
    baseline_target_rate: float = Field(ge=0, le=1)
    baseline_regression_rate: float = Field(ge=0, le=1)
    baseline_finish_rate: float | None = Field(default=None, ge=0, le=1)
    points: list[DosePoint] = Field(min_length=1, max_length=32)
    window: StrengthWindow | None = None
    nominated: float | None = None

    @model_validator(mode="after")
    def ordered(self) -> Dial:
        strengths = [point.strength for point in self.points]
        if strengths != sorted(set(strengths)):
            raise ValueError("dial strengths must be unique and increasing")
        if self.window is not None and (
            self.window.low not in strengths
            or self.window.high not in strengths
            or self.window.low > self.window.high
        ):
            raise ValueError("the window must span tested strengths")
        if self.nominated is not None and self.nominated not in strengths:
            raise ValueError("the nominated strength must be a tested strength")
        return self


class AtlasSource(Contract):
    """An inherited feature dictionary, pinned to exact bytes. nrly never retrains these."""

    kind: Literal["transcoder", "sae"]
    repo_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    filename: str = Field(min_length=1, max_length=300)
    tensor: str = Field(default="W_dec", min_length=1, max_length=120)
    tensor_sha256: str = Field(pattern=_DIGEST)
    license: str = Field(min_length=1, max_length=80)
    layer: int = Field(ge=0, le=4095)
    writes: str = Field(min_length=1, max_length=120)
    latents: int = Field(ge=1)
    width: int = Field(ge=1)


class LegendEntry(Contract):
    index: int = Field(ge=0)
    cosine: float = Field(ge=-1, le=1)
    url: str | None = Field(default=None, max_length=500)
    label: str | None = Field(default=None, max_length=500)
    label_source: str | None = Field(default=None, max_length=200)


class Legend(Contract):
    """Atlas features nearest a lever's direction. Always pencil: names are hypotheses."""

    schema_version: Literal["nrly-legend/1"] = "nrly-legend/1"
    status: Literal["pencil"] = "pencil"
    source: AtlasSource
    entries: list[LegendEntry] = Field(min_length=1, max_length=32)
    # Share of the direction's squared norm that lies in the span of the listed rows.
    explained: float = Field(ge=0, le=1)
    method: Literal["cosine_top_k+least_squares_span"] = "cosine_top_k+least_squares_span"

    @model_validator(mode="after")
    def ranked(self) -> Legend:
        magnitudes = [abs(entry.cosine) for entry in self.entries]
        if magnitudes != sorted(magnitudes, reverse=True):
            raise ValueError("legend entries must be ordered by absolute cosine")
        if len({entry.index for entry in self.entries}) != len(self.entries):
            raise ValueError("legend entries must name distinct features")
        return self


class RefereeVerdict(Contract):
    """The final, held-out result of a registered study for one exact edit."""

    schema_version: Literal["nrly-verdict/1"] = "nrly-verdict/1"
    # The weights the study actually loaded; a lever accepts only verdicts on its own base.
    base: PortableBase
    study_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    study_spec_digest: str = Field(pattern=_DIGEST)
    holdout: HoldoutCommitment
    operations_digest: str = Field(pattern=_DIGEST)
    nominee_digest: str = Field(pattern=_DIGEST)
    runtime_fingerprint: str = Field(pattern=_DIGEST)
    inference_rule: InferenceRule | None = None
    truncation: Literal["unscored", "fail"] = "unscored"
    seeds: list[int] = Field(min_length=1, max_length=32)
    controls: list[str] = Field(default_factory=list, max_length=8)
    comparators: list[str] = Field(default_factory=list, max_length=8)
    development_cases: int = Field(ge=1)
    assessment: StudyAssessment
    final_error_code: str | None = None
    finished_at: str = Field(min_length=1, max_length=64)


class Contraindication(Contract):
    """A build of the same model where the lever was tried and did not hold."""

    variant: str = Field(min_length=1, max_length=120)
    note: str = Field(min_length=1, max_length=500)
    base: PortableBase | None = None
    evidence_digest: str | None = Field(default=None, pattern=_DIGEST)


def operations_digest(operations: list[WeightOperationSpec]) -> str:
    """Portable identity of an edit: its ordered operations, with no runtime pin."""
    return digest([operation.model_dump(mode="json") for operation in operations])


def lever_name(behavior: str, sites: list[LeverSite]) -> str:
    """Levers are named by test and site, never by a meaning."""
    return behavior + "@" + "+".join(site.parameter for site in sites)


def _judge(verdict: RefereeVerdict | None) -> tuple[LeverStatus, str]:
    if verdict is None:
        return "pencil", "Not refereed: no held-out study has tested this edit."
    if verdict.final_error_code is not None:
        return "pencil", "The referee study did not complete, so it decides nothing."
    if verdict.assessment.scope != "real_model":
        return "pencil", "Refereed on an engineering fixture, which never establishes ink."
    if verdict.inference_rule is None or not verdict.assessment.inference:
        return "pencil", "Refereed without a registered inference rule, so it cannot ink."
    supported = (
        verdict.assessment.verdict == "observed_improvement"
        and verdict.assessment.recommendation == "candidate_supported"
    )
    if not supported:
        return "struck", "Refuted: the held-out study did not support this edit."
    if not verdict.controls:
        return "pencil", "Supported, but no matched control was registered, so it cannot ink."
    return (
        "inked",
        "Supported on held-out cases against the original and every registered "
        "control and comparator.",
    )


def verdict_status(verdict: RefereeVerdict | None) -> LeverStatus:
    """Derive ink from evidence; a status is never chosen by an author.

    ``inked`` needs a completed real-model study under a registered inference rule,
    at least one registered control, and a supported candidate. ``struck`` needs the
    same completed evidence without support. Engineering fixtures, failed executions,
    and studies without an inference rule leave an entry in ``pencil``.
    """
    return _judge(verdict)[0]


def status_reason(verdict: RefereeVerdict | None) -> str:
    """One sentence saying why a verdict yields its status; shares the rule above."""
    return _judge(verdict)[1]


class _LeverContent(Contract):
    schema_version: Literal["nrly-lever/1"] = "nrly-lever/1"
    name: str = Field(min_length=3, max_length=1800)
    behavior: str = Field(pattern=_SLUG)
    base: PortableBase
    sites: list[LeverSite] = Field(min_length=1, max_length=8)
    derivation: Derivation | None = None
    strength: float | None = Field(default=None, ge=-16, le=16)
    operations: list[WeightOperationSpec] = Field(default_factory=list, max_length=32)
    dial: Dial | None = None
    legend: Legend | None = None
    verdict: RefereeVerdict | None = None
    replications: list[RefereeVerdict] = Field(default_factory=list, max_length=64)
    contraindications: list[Contraindication] = Field(default_factory=list, max_length=64)
    attribution: list[str] = Field(default_factory=list, max_length=32)
    parent_id: str | None = Field(default=None, pattern=_DIGEST)
    status: LeverStatus = "pencil"

    @model_validator(mode="after")
    def coherent(self) -> _LeverContent:
        if self.name != lever_name(self.behavior, self.sites):
            raise ValueError("a lever is named by its behavior test and its sites")
        if len({site.parameter for site in self.sites}) != len(self.sites):
            raise ValueError("lever sites must name distinct tensors")
        targets = {site.parameter for site in self.sites}
        if any(operation.target not in targets for operation in self.operations):
            raise ValueError("every operation must edit one of the lever's sites")
        payload = sum(len(op.values) + len(op.left) + len(op.right) for op in self.operations)
        if payload > 65536:
            raise ValueError("lever operations exceed 65536 values")
        if any(not 1 <= len(item) <= 300 for item in self.attribution):
            raise ValueError("attribution lines must contain 1..300 characters")
        if self.derivation is not None:
            if self.derivation.base != self.base:
                raise ValueError("the derivation ran on a different base than the lever pins")
            if self.derivation.chosen not in self.sites:
                raise ValueError("the derived site must be one of the lever's sites")
            if self.derivation.recipe.behavior != self.behavior:
                raise ValueError("the recipe derives a different behavior")
        if self.legend is not None and self.derivation is None:
            raise ValueError("a legend describes a derived direction")
        if self.dial is not None and self.dial.nominated != self.strength:
            raise ValueError("the lever strength must equal the dial's nominated strength")
        if self.strength is not None and not self.operations:
            raise ValueError("a nominated strength requires the operations it compiles to")
        edit = operations_digest(self.operations)
        for record in (self.verdict, *self.replications):
            if record is None:
                continue
            if not self.operations:
                raise ValueError("a verdict needs the edit it refereed")
            if record.operations_digest != edit:
                raise ValueError("a verdict must referee exactly this lever's operations")
            if record.base != self.base:
                raise ValueError("a verdict must be refereed on the weights this lever pins")
        if self.replications and self.verdict is None:
            raise ValueError("replications repeat an existing verdict")
        if self.status != verdict_status(self.verdict):
            raise ValueError("lever status is derived from its verdict and cannot be chosen")
        return self


class LeverEntry(_LeverContent):
    """Immutable, content-addressed map entry; a revision names its parent."""

    id: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def immutable_identity(self) -> LeverEntry:
        content = _LeverContent.model_validate(self.model_dump(exclude={"id"}))
        if digest(content) != self.id:
            raise ValueError("lever content differs from its immutable identity")
        return self


def seal(content: dict[str, object]) -> LeverEntry:
    """Validate lever content, derive its status from its verdict, and assign its id."""
    data = dict(content)
    data.pop("id", None)
    verdict = data.get("verdict")
    parsed = RefereeVerdict.model_validate(verdict) if verdict is not None else None
    data["status"] = verdict_status(parsed)
    validated = _LeverContent.model_validate(data)
    return LeverEntry(id=digest(validated), **validated.model_dump())
