"""Run and assess fixed-versus-fitted, original-versus-patch comparisons."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal

from nrly.artifacts import MAX_MANIFEST_BYTES, read_local_file, write_immutable
from nrly.contracts import NrlyError, ProgramSpec, canonical_json, digest
from nrly.decision_contracts import (
    ArmMetrics,
    DecisionAssessment,
    DecisionCase,
    DecisionFreeze,
    DecisionObservation,
    DecisionProtocol,
    DecisionRun,
    FrozenRule,
    Split,
    Variant,
    observation_digest,
    original_program,
)
from nrly.lineage import weight_operations

if TYPE_CHECKING:
    from nrly.backends.decision import DecisionBackend


def fit_rule(
    protocol: DecisionProtocol,
    observations: tuple[DecisionObservation, ...],
    variant: Variant,
) -> FrozenRule:
    """Maximize fit coverage under an empirical error budget; no probability calibration.

    Ties prefer the higher threshold. Any missing fit observation disables acts.
    Validation labels never enter this function's calculations or fit digest.
    """
    cases = tuple(c for c in protocol.cases if c.split == "fit")
    by_id = {o.case_id: o for o in observations if o.variant == variant}
    selected = tuple(by_id[c.id] for c in cases)
    fit_digest = digest(
        {
            "cases": [c.model_dump(mode="json") for c in cases],
            "observations": observation_digest(selected),
            "grid": list(protocol.threshold_grid),
            "max_error": protocol.max_unsupported_rate,
            "min_acts": protocol.min_fit_acts,
        }
    )
    if any(o.execution is None for o in selected):
        return FrozenRule(fit_digest=fit_digest)
    best: tuple[int, float] | None = None
    for threshold in protocol.threshold_grid:
        acts = [c for c in cases if accepts(by_id[c.id], threshold)]
        errors = sum(not c.truth for c in acts)
        if (
            len(acts) >= protocol.min_fit_acts
            and errors / len(acts) <= protocol.max_unsupported_rate
        ):
            score = (len(acts), threshold)
            if best is None or score > best:
                best = score
    return FrozenRule(threshold=best[1] if best else None, fit_digest=fit_digest)


def accepts(observation: DecisionObservation, threshold: float | None) -> bool:
    return (
        threshold is not None
        and observation.execution is not None
        and observation.execution.probability >= threshold
    )


def run_decision_study(
    backend: DecisionBackend,
    protocol: DecisionProtocol,
    program: ProgramSpec,
    *,
    record: Callable[[DecisionObservation], None] | None = None,
    on_freeze: Callable[[DecisionFreeze], None] | None = None,
) -> DecisionRun:
    """Trusted local operation; budget checked BETWEEN calls, never a hard timeout.

    The optional journal callback persists each attempt before further work. Exceptions
    in a model invocation become failures; corruption of base weights aborts subsequent
    calls. Ctrl-C still propagates, leaving the already-journaled attempts intact.
    """
    protocol = DecisionProtocol.model_validate(protocol.model_dump())
    program = ProgramSpec.model_validate(program.model_dump())
    weight_operations(program)
    if program.expected_base != backend.identity.fingerprint:
        raise NrlyError("base_mismatch", "decision patch must pin the loaded model identity")
    observations: list[DecisionObservation] = []
    started = perf_counter()
    broken = False

    def collect(cases: tuple[DecisionCase, ...]) -> None:
        nonlocal broken
        for case in cases:
            for variant in ("original", "patch"):
                selected = original_program(backend.identity) if variant == "original" else program
                request = protocol.request(case)
                fields = {
                    "case_id": case.id,
                    "variant": variant,
                    "input_digest": digest(request),
                    "program_digest": digest(selected),
                }
                if broken or perf_counter() - started >= protocol.max_elapsed_seconds:
                    observation = DecisionObservation.model_validate(
                        {
                            **fields,
                            "error_code": "restoration_failed" if broken else "budget_exceeded",
                        }
                    )
                else:
                    try:
                        result = backend.observe(request, selected)
                        observation = DecisionObservation.model_validate(
                            {**fields, "execution": result}
                        )
                    except Exception as error:
                        broken = isinstance(error, NrlyError) and error.code == "restoration_failed"
                        observation = DecisionObservation.model_validate(
                            {
                                **fields,
                                "error_code": "restoration_failed"
                                if broken
                                else "execution_failed",
                            }
                        )
                observations.append(observation)
                if record is not None:
                    record(observation)

    collect(tuple(c for c in protocol.cases if c.split == "fit"))
    freeze = DecisionFreeze(
        protocol_digest=digest(protocol),
        identity_digest=digest(backend.identity),
        program_digest=digest(program),
        semantics=backend.semantics,
        execution_identity=backend.execution_identity,
        original_rule=fit_rule(protocol, tuple(observations), "original"),
        patch_rule=fit_rule(protocol, tuple(observations), "patch"),
    )
    # No held-out inference occurs until both rules and candidate identity have been frozen.
    if on_freeze is not None:
        on_freeze(freeze)
    collect(tuple(c for c in protocol.cases if c.split != "fit"))
    elapsed_ms = (perf_counter() - started) * 1000
    return DecisionRun(
        protocol=protocol,
        identity=backend.identity,
        program=program,
        freeze=freeze,
        elapsed_ms=elapsed_ms,
        budget_exceeded=elapsed_ms > protocol.max_elapsed_seconds * 1000,
        observations=tuple(observations),
    )


def verify_run(run: DecisionRun) -> None:
    run = DecisionRun.model_validate(run.model_dump())
    if run.budget_exceeded != (run.elapsed_ms > run.protocol.max_elapsed_seconds * 1000):
        raise NrlyError("invalid_artifact", "runtime budget accounting mismatch")
    weight_operations(run.program)
    if run.program.expected_base != run.identity.fingerprint:
        raise NrlyError("invalid_artifact", "decision program base mismatch")
    expected = {(c.id, v) for c in run.protocol.cases for v in ("original", "patch")}
    if (
        len(run.observations) != len(expected)
        or {(o.case_id, o.variant) for o in run.observations} != expected
    ):
        raise NrlyError("invalid_artifact", "missing, duplicate or unknown observations")
    cases = {c.id: c for c in run.protocol.cases}
    for observation in run.observations:
        program = (
            original_program(run.identity) if observation.variant == "original" else run.program
        )
        if observation.input_digest != digest(
            run.protocol.request(cases[observation.case_id])
        ) or observation.program_digest != digest(program):
            raise NrlyError("invalid_artifact", "observation input or program binding mismatch")
    expected_freeze = DecisionFreeze(
        protocol_digest=digest(run.protocol),
        identity_digest=digest(run.identity),
        program_digest=digest(run.program),
        semantics=run.freeze.semantics,
        execution_identity=run.freeze.execution_identity,
        original_rule=fit_rule(run.protocol, run.observations, "original"),
        patch_rule=fit_rule(run.protocol, run.observations, "patch"),
    )
    if run.freeze != expected_freeze:
        raise NrlyError("invalid_artifact", "frozen rules or provenance have changed")


def assess_decisions(run: DecisionRun) -> DecisionAssessment:
    verify_run(run)
    index = {(o.case_id, o.variant): o for o in run.observations}
    metrics: list[ArmMetrics] = []
    splits: tuple[Split, ...] = ("fit", "evaluation", "challenge")
    variants: tuple[Variant, ...] = ("original", "patch")
    rules: tuple[Literal["fixed", "fitted"], ...] = ("fixed", "fitted")
    for split in splits:
        cases = [c for c in run.protocol.cases if c.split == split]
        if not cases:
            continue
        for variant in variants:
            frozen = run.freeze.original_rule if variant == "original" else run.freeze.patch_rule
            for rule in rules:
                threshold = run.protocol.fixed_threshold if rule == "fixed" else frozen.threshold
                executions = [(c, index[c.id, variant].execution) for c in cases]
                scored = [(c, e) for c, e in executions if e is not None]
                acts = [c for c in cases if accepts(index[c.id, variant], threshold)]
                errors = sum(not c.truth for c in acts)
                metrics.append(
                    ArmMetrics(
                        variant=variant,
                        rule=rule,
                        split=split,
                        threshold=threshold,
                        cases=len(cases),
                        families=len({c.family for c in cases}),
                        scored=len(scored),
                        acts=len(acts),
                        unsupported_acts=errors,
                        coverage=len(acts) / len(cases),
                        unsupported_rate=errors / len(acts) if acts else None,
                        agreement=sum((e.probability >= 0.5) == c.truth for c, e in scored)
                        / len(scored)
                        if scored
                        else None,
                        brier=sum((e.probability - c.truth) ** 2 for c, e in scored) / len(scored)
                        if scored
                        else None,
                        elapsed_ms=sum(e.elapsed_ms for _, e in scored),
                    )
                )
    rescued = introduced = lost = 0
    for case in run.protocol.cases:
        if case.split == "fit":
            continue
        before = accepts(index[case.id, "original"], run.freeze.original_rule.threshold)
        after = accepts(index[case.id, "patch"], run.freeze.patch_rule.threshold)
        rescued += int(case.truth and after and not before)
        introduced += int(not case.truth and after and not before)
        lost += int(case.truth and before and not after)
    complete = not run.budget_exceeded and all(o.execution is not None for o in run.observations)
    evaluation = {m.variant: m for m in metrics if m.split == "evaluation" and m.rule == "fitted"}
    holds = [m for m in metrics if m.split != "fit" and m.variant == "patch" and m.rule == "fitted"]
    promising = (
        complete
        and rescued > lost
        and introduced == 0
        and evaluation["patch"].coverage > evaluation["original"].coverage
        and all(
            m.acts > 0
            and m.unsupported_rate is not None
            and m.unsupported_rate <= run.protocol.max_unsupported_rate
            for m in holds
        )
    )
    return DecisionAssessment(
        run_digest=digest(run),
        freeze_digest=digest(run.freeze),
        model_id=run.identity.model_id,
        semantics=run.freeze.semantics,
        label_origin=run.protocol.label_origin,
        status="complete" if complete else "incomplete",
        recommendation="incomplete"
        if not complete
        else "further_shadow_evaluation"
        if promising
        else "retain_original",
        metrics=tuple(metrics),
        rescued_supported_acts=rescued,
        introduced_unsupported_acts=introduced,
        lost_supported_acts=lost,
        max_unsupported_rate=run.protocol.max_unsupported_rate,
    )


def save_decision_run(directory: Path, run: DecisionRun) -> None:
    verify_run(run)
    raw = canonical_json(run).encode()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise NrlyError("budget_exceeded", "decision record exceeds 4 MiB")
    write_immutable(directory, "run.json", raw)
    write_immutable(directory, "run.sha256", digest(run).encode())


def read_decision_run(directory: Path) -> DecisionRun:
    run = DecisionRun.model_validate_json(
        read_local_file(directory, "run.json", MAX_MANIFEST_BYTES)
    )
    if digest(run).encode() != read_local_file(directory, "run.sha256", 64):
        raise NrlyError("invalid_artifact", "decision record checksum mismatch")
    verify_run(run)
    return run
