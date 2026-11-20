"""Application operations for the lever map: derive, dial, referee, legend.

These compose existing validated pieces and add no second execution path. Deriving a
direction is trusted local authoring in this process. Every measured point of a dial and
every arm of a referee study runs through the ordinary isolated ``Runner`` and the
ordinary ``StudyEngine``, so the evidence is the same kind as any other nrly run.

A dial is exploratory development evidence: it nominates one strength. Only the
referee's held-out study can change a lever's status, and it tests exactly the edit the
dial nominated.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from nrly.artifacts import MAX_MANIFEST_BYTES, checked_handle, read_local_file, write_immutable
from nrly.contracts import (
    BudgetSpec,
    CaseSpec,
    Contract,
    ExperimentSpec,
    GenerationSpec,
    ModelIdentity,
    ModelSpec,
    NrlyError,
    PortableBase,
    ProgramSpec,
    RunRecord,
    WeightOperationSpec,
    canonical_json,
    digest,
)
from nrly.evaluation import EvaluationReport, evaluate_run
from nrly.lever_contracts import LeverEntry, RecipeSpec, operations_digest
from nrly.levers import LeverMap, build_dial, pencil, verdict_from_study
from nrly.runner import Runner
from nrly.study_contracts import (
    CandidateSpec,
    ComparatorSpec,
    InferenceRule,
    StudyBudget,
    StudySpec,
)

_LINEAR = {"weight_low_rank", "weight_add", "weight_sparse_add"}
_NAME = r"^[a-z0-9][a-z0-9-]{0,63}$"


class LeverPlan(Contract):
    """Local working record: a lever's edit and its matched controls at unit strength.

    Every operation is linear in ``factor``, so any tested strength is the same
    operation with its factor scaled. Controls are the same size of edit pointed
    somewhere that should not matter.
    """

    schema_version: Literal["nrly-lever-plan/1"] = "nrly-lever-plan/1"
    entry_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    base: PortableBase
    unit_operations: list[WeightOperationSpec] = Field(min_length=1, max_length=32)
    controls: dict[str, list[WeightOperationSpec]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def scalable(self) -> LeverPlan:
        groups = [self.unit_operations, *self.controls.values()]
        if any(op.kind not in _LINEAR for group in groups for op in group):
            raise ValueError("planned operations must be linear in their factor")
        if len(self.controls) > 8 or any(not group for group in self.controls.values()):
            raise ValueError("a plan holds at most eight nonempty controls")
        return self


class DeriveRequest(Contract):
    model: ModelSpec
    recipe: RecipeSpec
    control_seed: int = Field(default=0, ge=0, le=2**32 - 1)


class DialRequest(Contract):
    model: ModelSpec
    cases: list[CaseSpec] = Field(min_length=2, max_length=10000)
    strengths: list[float] = Field(min_length=1, max_length=32)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    seeds: list[int] = Field(min_length=1, max_length=32)
    rule: InferenceRule = Field(default_factory=InferenceRule)
    truncation: Literal["unscored", "fail"] = "unscored"
    budget: BudgetSpec = Field(default_factory=BudgetSpec)

    @model_validator(mode="after")
    def valid_sweep(self) -> DialRequest:
        if len(set(self.strengths)) != len(self.strengths) or any(
            value == 0 or not -16 <= value <= 16 for value in self.strengths
        ):
            raise ValueError("strengths must be unique, nonzero and within [-16, 16]")
        if len(set(self.seeds)) != len(self.seeds) or any(
            not 0 <= seed <= 2**32 - 1 for seed in self.seeds
        ):
            raise ValueError("seeds must be unique unsigned 32-bit integers")
        roles = {case.role for case in self.cases}
        if roles != {"target", "regression"}:
            raise ValueError("a dial needs both target and regression cases")
        return self


class RefereeRequest(Contract):
    model: ModelSpec
    development_cases: list[CaseSpec] = Field(min_length=1, max_length=10000)
    comparators: list[ComparatorSpec] = Field(default_factory=list, max_length=8)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    seeds: list[int] | None = Field(default=None, min_length=1, max_length=32)
    inference: InferenceRule = Field(default_factory=InferenceRule)
    truncation: Literal["unscored", "fail"] = "unscored"
    run_budget: BudgetSpec = Field(default_factory=BudgetSpec)
    # Names of planned controls to register; omit to register every planned control.
    controls: list[str] | None = Field(default=None, max_length=8)
    name: str | None = Field(default=None, min_length=1, max_length=160)


class NeuronpediaRef(Contract):
    # Bounds match the atlas adapter, so a label source always fits its record.
    model: str = Field(min_length=1, max_length=80)
    source_id: str = Field(min_length=1, max_length=80)
    fetch_labels: bool = False


class LegendRequest(Contract):
    """Which inherited dictionary to read. ``filename`` may contain ``{layer}``."""

    kind: Literal["transcoder", "sae"]
    repo_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    filename: str = Field(min_length=1, max_length=300)
    license: str = Field(min_length=1, max_length=80)
    writes: str = Field(min_length=1, max_length=120)
    tensor: str = Field(default="W_dec", min_length=1, max_length=120)
    orientation: Literal["latents_by_width", "width_by_latents"] = "latents_by_width"
    top_k: int = Field(default=8, ge=1, le=32)
    neuronpedia: NeuronpediaRef | None = None


def lever_map(root: Path) -> LeverMap:
    return LeverMap(root / "levers")


def scale_operations(
    operations: Sequence[WeightOperationSpec], strength: float
) -> list[WeightOperationSpec]:
    """The same edit at another strength; valid because the operations are linear."""
    if any(operation.kind not in _LINEAR for operation in operations):
        raise NrlyError("unsupported_operation", "only factor-linear operations can be scaled")
    return [
        WeightOperationSpec.model_validate(
            {**operation.model_dump(), "factor": operation.factor * strength}
        )
        for operation in operations
    ]


def _plan(root: Path, levers: LeverMap, entry: LeverEntry) -> LeverPlan:
    origin = levers.history(entry.id)[0].id
    raw = read_local_file(root / "levers" / "plans", f"{origin}.json", MAX_MANIFEST_BYTES)
    plan = LeverPlan.model_validate_json(raw)
    if plan.entry_id != origin or plan.base != entry.base:
        raise NrlyError("plan_mismatch", "the stored plan belongs to another lever")
    return plan


def _local_identity(runner: Runner, model: ModelSpec, base: PortableBase) -> ModelIdentity:
    """Load the model once in a worker to learn this runtime, then verify the weights."""
    record = runner.wait(
        runner.submit(
            ExperimentSpec(name="local runtime identity", purpose="inspection", model=model)
        ).id
    )
    if record.status != "succeeded" or record.model_identity is None:
        code = record.error.code if record.error else record.status
        raise NrlyError("identity_unavailable", f"could not load the model here: {code}")
    if not base.matches(record.model_identity):
        raise NrlyError(
            "base_mismatch", "the model loaded here does not hold the weights this lever pins"
        )
    return record.model_identity


def _program(
    operations: Sequence[WeightOperationSpec], base: PortableBase, identity: ModelIdentity
) -> ProgramSpec:
    from nrly.deployment import bind_operations

    return bind_operations(operations, base, identity)


def derive_lever(
    root: Path, request: DeriveRequest, model_cache: Path | None = None
) -> dict[str, Any]:
    """Run a recipe on the loaded checkpoint and pencil the result onto the map."""
    if request.model.backend != "huggingface":
        raise NrlyError("unsupported_operation", "recipes observe a Hugging Face causal model")
    if model_cache is not None:
        os.environ["HF_HOME"] = str(model_cache.resolve())
    from nrly.backends.huggingface import HuggingFaceBackend
    from nrly.controls import random_orthogonal_direction
    from nrly.recipes import backend_activations, compile_removal, derive

    levers = lever_map(root)
    with HuggingFaceBackend(request.model) as backend:
        derivation = derive(
            request.recipe,
            backend_activations(backend, request.recipe),
            backend.identity,
            levers.tensors,
        )
        direction = levers.tensors.get(derivation.direction)
        weights = backend.read_parameter(derivation.chosen.parameter)
        unit = compile_removal(derivation.chosen, weights, direction, 1.0)
        controls = {
            "random-direction": compile_removal(
                derivation.chosen,
                weights,
                random_orthogonal_direction(direction, request.control_seed),
                1.0,
            )
        }
        others = [score for score in derivation.scores if score.site != derivation.chosen]
        if others:
            # The same direction removed where it separated the prompts least.
            weakest = min(others, key=lambda score: score.separation).site
            elsewhere = backend.read_parameter(weakest.parameter)
            if elsewhere.shape[0] == direction.size:
                controls["wrong-site"] = compile_removal(weakest, elsewhere, direction, 1.0)
    entry = levers.add(pencil(derivation))
    plan = LeverPlan(
        entry_id=entry.id, base=entry.base, unit_operations=unit, controls=controls
    )
    write_immutable(
        root / "levers" / "plans", f"{entry.id}.json", canonical_json(plan).encode("utf-8")
    )
    return {
        "entry_id": entry.id,
        "name": entry.name,
        "status": entry.status,
        "chosen_site": derivation.chosen.model_dump(mode="json"),
        "scores": [score.model_dump(mode="json") for score in derivation.scores],
        "controls": sorted(controls),
        "runtime_fingerprint": derivation.runtime_fingerprint,
    }


def dial_lever(
    root: Path, entry_id: str, request: DialRequest, model_cache: Path | None = None
) -> dict[str, Any]:
    """Measure the lever at each strength on development cases and nominate one."""
    levers = lever_map(root)
    entry = levers.get(checked_handle(entry_id))
    if entry.verdict is not None:
        raise NrlyError("lever_refereed", "a refereed lever keeps the edit that was refereed")
    plan = _plan(root, levers, entry)
    failed: list[dict[str, Any]] = []
    with Runner(root, model_cache) as runner:
        identity = _local_identity(runner, request.model, entry.base)

        def measure(program: ProgramSpec, seed: int) -> RunRecord:
            return runner.wait(
                runner.submit(
                    ExperimentSpec(
                        schema_version="0.3",
                        name=f"dial {entry.behavior}"[:160],
                        model=request.model,
                        program=program,
                        generation=request.generation.model_copy(update={"seed": seed}),
                        cases=request.cases,
                        budget=request.budget,
                    )
                ).id
            )

        def reports(program: ProgramSpec, label: str) -> list[EvaluationReport] | None:
            collected = []
            for seed in request.seeds:
                run = measure(program, seed)
                report = evaluate_run(run, request.truncation)
                if run.status != "succeeded" or not report.complete:
                    failed.append(
                        {
                            "arm": label,
                            "seed": seed,
                            "run_id": run.id,
                            "status": run.status,
                            "error_code": run.error.code if run.error else None,
                            "issues": report.issues[:5],
                            # Unscored cases (for example token-limit truncation with no
                            # registered policy) make a run unusable for a dial.
                            "unscored": report.summary.unscored,
                        }
                    )
                    return None
                collected.append(report)
            return collected

        baseline = reports(ProgramSpec(expected_base=identity.fingerprint), "baseline")
        if baseline is None:
            detail = failed[-1]
            hint = (
                f"{detail['unscored']} cases were unscored; if outputs hit the token limit, "
                'register truncation="fail" or raise max_new_tokens'
                if detail["unscored"]
                else f"run {detail['run_id']} ended {detail['status']}"
            )
            raise NrlyError("dial_incomplete", f"the unedited baseline is unusable: {hint}")
        points: dict[float, tuple[str, list[EvaluationReport]]] = {}
        for strength in sorted(request.strengths):
            operations = scale_operations(plan.unit_operations, strength)
            measured = reports(_program(operations, entry.base, identity), f"strength={strength}")
            if measured is not None:
                points[strength] = (operations_digest(operations), measured)
    if not points:
        raise NrlyError("dial_incomplete", "no tested strength completed; see the run store")
    dial = build_dial(
        rule=request.rule,
        truncation=request.truncation,
        generation=request.generation,
        seeds=request.seeds,
        runtime_fingerprint=identity.fingerprint,
        baseline=baseline,
        points=points,
    )
    nominated = (
        scale_operations(plan.unit_operations, dial.nominated)
        if dial.nominated is not None
        else []
    )
    revised = levers.attach_dial(entry.id, dial, nominated)
    return {
        "entry_id": revised.id,
        "previous_entry_id": entry.id,
        "status": revised.status,
        "evidence": dial.evidence,
        "nominated": dial.nominated,
        "window": dial.window.model_dump(mode="json") if dial.window else None,
        "points": [point.model_dump(mode="json") for point in dial.points],
        "failed": failed,
    }


def nominate_lever(root: Path, entry_id: str, strength: float) -> dict[str, Any]:
    """Operator override: choose a tested strength the dial did not nominate.

    A dial with no working window nominates nothing. Refereeing the best tested strength
    anyway lets a null be struck on held-out cases instead of quietly abandoned. No model
    runs here; the recorded dial keeps its window, so the override stays visible.
    """
    levers = lever_map(root)
    entry = levers.get(checked_handle(entry_id))
    if entry.verdict is not None:
        raise NrlyError("lever_refereed", "a refereed lever keeps the edit that was refereed")
    if entry.dial is None or strength not in [point.strength for point in entry.dial.points]:
        raise NrlyError("lever_unmeasured", "nominate a strength the recorded dial tested")
    plan = _plan(root, levers, entry)
    revised = levers.attach_dial(
        entry.id,
        entry.dial.model_copy(update={"nominated": strength}),
        scale_operations(plan.unit_operations, strength),
    )
    return {
        "entry_id": revised.id,
        "previous_entry_id": entry.id,
        "status": revised.status,
        "nominated": strength,
        "inside_window": entry.dial.window is not None
        and entry.dial.window.low <= strength <= entry.dial.window.high,
    }


def _study_budget(spec_runs: int, cases: int, run_budget: BudgetSpec) -> StudyBudget:
    """One frozen candidate: a development attempt and the reserved final, no search."""
    total = spec_runs * 2
    try:
        return StudyBudget(
            max_attempts=1,
            max_runs=total,
            max_case_executions=spec_runs * cases,
            max_output_tokens=total * run_budget.max_output_tokens,
            max_capture_bytes=total * run_budget.max_capture_bytes,
            max_wall_seconds=total * run_budget.wall_seconds,
        )
    except ValueError as error:
        raise NrlyError(
            "study_budget_exceeded",
            "the referee design exceeds a study's admission ceilings; reduce seeds, "
            "arms, or the per-run budget",
        ) from error


def referee_lever(
    root: Path,
    entry_id: str,
    request: RefereeRequest,
    holdout: Sequence[CaseSpec],
    model_cache: Path | None = None,
    *,
    replication: bool = False,
) -> dict[str, Any]:
    """Put the nominated edit through a registered held-out study and record the verdict."""
    from nrly.studies import StudyEngine

    levers = lever_map(root)
    entry = levers.get(checked_handle(entry_id))
    if not entry.operations or entry.strength is None:
        raise NrlyError("lever_unnominated", "run the dial first; no strength is nominated")
    if replication != (entry.verdict is not None):
        raise NrlyError(
            "lever_refereed" if entry.verdict is not None else "lever_unrefereed",
            "a first verdict needs an unrefereed lever; a replication needs a refereed one",
        )
    plan = _plan(root, levers, entry)
    names = sorted(plan.controls) if request.controls is None else request.controls
    unknown = [name for name in names if name not in plan.controls]
    if unknown or len(set(names)) != len(names):
        raise NrlyError("unknown_control", f"controls must be distinct planned names: {unknown}")
    with Runner(root, model_cache) as runner:
        identity = _local_identity(runner, request.model, entry.base)
        recorded = {record.runtime_fingerprint for record in (entry.verdict, *entry.replications)
                    if record is not None}
        if replication and identity.fingerprint in recorded:
            # Refuse before spending a study: the same runtime is not an independent repeat.
            raise NrlyError(
                "replication_not_independent",
                "this runtime already has a recorded verdict for this lever",
            )
        candidate = CandidateSpec(
            name="lever", program=_program(entry.operations, entry.base, identity)
        )
        controls = [
            CandidateSpec(
                name=name,
                program=_program(
                    scale_operations(plan.controls[name], entry.strength), entry.base, identity
                ),
            )
            for name in names
        ]
        seeds = len(request.seeds) if request.seeds is not None else 1
        runs = (3 + len(controls) + len(request.comparators)) * seeds
        spec = StudySpec(
            schema_version="0.3",
            name=request.name or f"referee {entry.behavior}"[:160],
            model=request.model,
            generation=request.generation,
            development_cases=request.development_cases,
            controls=controls,
            seeds=request.seeds,
            comparators=request.comparators,
            run_budget=request.run_budget,
            budget=_study_budget(
                runs, len(request.development_cases) + len(holdout), request.run_budget
            ),
            truncation=request.truncation,
            inference=request.inference,
        )
        engine = StudyEngine(runner)
        try:
            study = engine.create(spec, list(holdout))
            attempt = engine.wait(study.id, engine.submit_candidate(study.id, candidate).id)
            if attempt.status != "succeeded":
                # The attempt stays in the study ledger; the lever keeps its status.
                return {
                    "entry_id": entry.id,
                    "status": entry.status,
                    "study_id": study.id,
                    "attempt_id": attempt.id,
                    "refereed": False,
                    "error_code": attempt.error_code,
                    "issues": attempt.assessment.issues if attempt.assessment else [],
                }
            engine.finalize(study.id, attempt.id)
            final = engine.wait_final(study.id)
        finally:
            engine.close()
    verdict = verdict_from_study(final, candidate, entry.operations)
    revised = (
        levers.add_replication(entry.id, verdict)
        if replication
        else levers.attach_verdict(entry.id, verdict)
    )
    return {
        "entry_id": revised.id,
        "previous_entry_id": entry.id,
        "status": revised.status,
        "study_id": study.id,
        "refereed": True,
        "replication": replication,
        "verdict": verdict.assessment.verdict,
        "recommendation": verdict.assessment.recommendation,
        "issues": verdict.assessment.issues,
        "final_error_code": verdict.final_error_code,
        "inference": [item.model_dump(mode="json") for item in verdict.assessment.inference],
    }


def legend_lever(
    root: Path,
    entry_id: str,
    request: LegendRequest,
    *,
    cache: Path | None = None,
    local_file: Path | None = None,
) -> dict[str, Any]:
    """Name a lever's direction with an inherited feature dictionary, in pencil."""
    from nrly.atlas import (
        DecoderCache,
        HubFile,
        LocalFile,
        legend,
        load_decoder,
        neuronpedia_labels,
        neuronpedia_url,
    )
    from nrly.lever_contracts import AtlasSource

    levers = lever_map(root)
    entry = levers.get(checked_handle(entry_id))
    if entry.derivation is None or entry.derivation.chosen.layer is None:
        raise NrlyError("legend_unavailable", "a legend needs a derived direction with a layer")
    direction = levers.tensors.get(entry.derivation.direction)
    layer = entry.derivation.chosen.layer
    filename = request.filename.replace("{layer}", str(layer))
    store = DecoderCache((cache or root / "atlas").resolve())
    key = digest(
        {
            "kind": request.kind,
            "repo": request.repo_id,
            "revision": request.revision,
            "file": filename,
            "tensor": request.tensor,
            "orientation": request.orientation,
            "license": request.license,
            "writes": request.writes,
            "width": int(direction.size),
        }
    )
    index = (cache or root / "atlas").resolve() / "index"
    decoder = None
    source: AtlasSource | None = None
    if (index / f"{key}.json").is_file():
        source = AtlasSource.model_validate_json(read_local_file(index, f"{key}.json", 65536))
        decoder = store.get(source.tensor_sha256)
    if decoder is None or source is None:
        source, decoder = load_decoder(
            LocalFile(local_file)
            if local_file is not None
            else HubFile(request.repo_id, request.revision, filename),
            kind=request.kind,
            repo_id=request.repo_id,
            revision=request.revision,
            filename=filename,
            license=request.license,
            layer=layer,
            writes=request.writes,
            expected_width=int(direction.size),
            tensor=request.tensor,
            orientation=request.orientation,
        )
        store.put(source, decoder)
        write_immutable(index, f"{key}.json", canonical_json(source).encode("utf-8"))
    reference = request.neuronpedia
    result = legend(
        direction,
        decoder,
        source,
        top_k=request.top_k,
        url_for=(lambda i: neuronpedia_url(reference.model, reference.source_id, i))
        if reference is not None
        else None,
        label_for=neuronpedia_labels(reference.model, reference.source_id)
        if reference is not None and reference.fetch_labels
        else None,
    )
    attribution = (
        f"Legend from {source.repo_id}@{source.revision[:12]} ({source.license}); "
        "inherited feature names are hypotheses, not findings."
    )
    revised = levers.attach_legend(entry.id, result, attribution)
    return {
        "entry_id": revised.id,
        "previous_entry_id": entry.id,
        "status": revised.status,
        "legend": result.model_dump(mode="json"),
    }
