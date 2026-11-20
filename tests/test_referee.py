"""Study schema 0.3 refereeing: truncation policy, paired inference, generation comparators.

Synthetic records exercise scoring and adjudication only. The tiny-backend study checks
execution and integrity; it is engineering evidence, never a model-quality claim.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

import nrly.studies as studies_module
from nrly.contracts import (
    CaseSpec,
    DiagnosticSpec,
    ExperimentSpec,
    GenerationSpec,
    ModelIdentity,
    NrlyError,
    ProgramSpec,
    RunRecord,
    SampleResult,
    digest,
)
from nrly.evaluation import SCORER_VERSION, TRUNCATION_SCORER_VERSION, compare_runs, evaluate_run
from nrly.evaluation.study_reports import assess_study_arms, study_scorer_version
from nrly.runner import Runner
from nrly.studies import StudyEngine, _inconclusive
from nrly.study_contracts import (
    CandidateSpec,
    StudySpec,
    study_arm_generation,
    study_arm_name,
    study_arm_names,
    study_seeds,
)

IDENTITY = ModelIdentity(
    backend="tiny", model_id="nrly/tiny-sum-v1", revision="1", fingerprint="f" * 64
)
RULE = {"resamples": 1000}
CONTROL_PARITY = "candidate did not improve targets beyond every registered control"
Passing = Callable[[str, int], set[str]]
Arms = list[tuple[str, RunRecord]]


def sample(
    case: CaseSpec, output: str | None, *, cut_off: bool = False, **updates: object
) -> SampleResult:
    data: dict[str, object] = {
        "case_id": case.id,
        "status": "completed",
        "provenance": "torch_fixture",
        "output": output,
        "finish_reason": "length" if cut_off else "eos",
        "truncated": cut_off,
    }
    data.update(updates)
    return SampleResult.model_validate(data)


def record(
    cases: list[CaseSpec],
    samples: list[SampleResult],
    *,
    generation: GenerationSpec | None = None,
    identifier: str = "run",
    **spec_updates: object,
) -> RunRecord:
    spec = ExperimentSpec.model_validate(
        {
            "schema_version": "0.3",
            "name": "synthetic referee record",
            "cases": cases,
            "generation": generation or GenerationSpec(),
            **spec_updates,
        }
    )
    return RunRecord(
        id=identifier,
        status="succeeded",
        spec=spec,
        spec_digest=digest(spec),
        created_at="unit",
        model_identity=IDENTITY,
        samples=samples,
    )


def mixed_run() -> RunRecord:
    cases = [
        CaseSpec(id="finished-pass", input="a", expected="A", group="reasoning"),
        CaseSpec(id="cut-off", input="b", expected="B", group="reasoning"),
        CaseSpec(id="finished-fail", input="c", expected="C", group="reasoning"),
        CaseSpec(id="local-pass", input="d", expected="D", role="regression", group="locality"),
        CaseSpec(id="local-cut-off", input="e", expected="E", role="regression", group="locality"),
    ]
    return record(
        cases,
        [
            sample(cases[0], "A"),
            # The text matches, but an output that hit the token limit never passes.
            sample(cases[1], "B", cut_off=True),
            sample(cases[2], "wrong"),
            sample(cases[3], "D"),
            # Records without the explicit flag are recognised by their finish reason.
            sample(cases[4], "partial", finish_reason="length", truncated=None),
        ],
    )


def test_default_policy_leaves_truncated_outputs_unscored_and_unreported() -> None:
    report = evaluate_run(mixed_run())
    assert report == evaluate_run(mixed_run(), "unscored")
    assert (report.scorer_version, report.truncation) == (SCORER_VERSION, "unscored")
    assert not report.complete
    assert [case.outcome for case in report.cases] == [
        "pass",
        "unscored",
        "fail",
        "pass",
        "unscored",
    ]
    assert report.cases[1].reason == report.cases[4].reason == "inference output was truncated"
    assert (report.summary.passed, report.summary.unscored) == (2, 2)
    assert report.summary.accuracy is None
    data = report.model_dump(mode="json")
    assert "truncation" not in data
    for summary in (
        data["summary"],
        data["targets"],
        data["regressions"],
        *data["groups"].values(),
    ):
        assert "finished" not in summary and "finished_passed" not in summary
    assert all("finished" not in case for case in data["cases"])


def test_default_policy_serialization_is_byte_identical_to_scorer_version_two() -> None:
    cases = [
        CaseSpec(id="sum", input="[2, 4]", expected="6", scorer="numeric"),
        CaseSpec(id="zero", input="[0, 0]", expected="0", scorer="numeric", role="regression"),
    ]
    spec = ExperimentSpec(name="legacy record", cases=cases)
    run = RunRecord(
        id="run",
        status="succeeded",
        spec=spec,
        spec_digest=digest(spec),
        created_at="unit",
        model_identity=IDENTITY,
        samples=[
            SampleResult(
                case_id="sum",
                status="completed",
                provenance="torch_fixture",
                output="6",
                finish_reason="length",
            ),
            SampleResult(
                case_id="zero", status="completed", provenance="torch_fixture", output="0"
            ),
        ],
    )
    expected = {
        "schema_version": "0.1",
        "scorer_version": "nrly-scorers/2",
        "run_id": "run",
        "scope": "engineering_fixture",
        "complete": False,
        "issues": [],
        "summary": {
            "total": 2,
            "scored": 1,
            "passed": 1,
            "failed": 0,
            "unscored": 1,
            "accuracy": None,
        },
        "targets": {
            "total": 1,
            "scored": 0,
            "passed": 0,
            "failed": 0,
            "unscored": 1,
            "accuracy": None,
        },
        "regressions": {
            "total": 1,
            "scored": 1,
            "passed": 1,
            "failed": 0,
            "unscored": 0,
            "accuracy": 1.0,
        },
        "cases": [
            {
                "case_id": "sum",
                "split": "discovery",
                "role": "target",
                "outcome": "unscored",
                "output": "6",
                "expected": "6",
                "scorer": "numeric",
                "score": None,
                "reason": "inference output was truncated",
            },
            {
                "case_id": "zero",
                "split": "discovery",
                "role": "regression",
                "outcome": "pass",
                "output": "0",
                "expected": "0",
                "scorer": "numeric",
                "score": 1.0,
                "reason": "numeric match",
            },
        ],
    }
    report = evaluate_run(run)
    assert report.model_dump(mode="json") == expected
    assert report.model_dump_json() == json.dumps(expected, separators=(",", ":"))
    comparison = compare_runs(run, run)
    assert comparison.baseline.model_dump_json() == report.model_dump_json()
    assert not comparison.comparable and comparison.verdict == "inconclusive"


def test_registered_fail_policy_scores_truncation_as_failure_with_finish_counts() -> None:
    report = evaluate_run(mixed_run(), "fail")
    assert (report.scorer_version, report.truncation) == (TRUNCATION_SCORER_VERSION, "fail")
    assert report.scorer_version == "nrly-scorers/3"
    assert report.complete and report.issues == []
    assert [(case.outcome, case.score, case.finished) for case in report.cases] == [
        ("pass", 1.0, True),
        ("fail", 0.0, False),
        ("fail", 0.0, True),
        ("pass", 1.0, True),
        ("fail", 0.0, False),
    ]
    for index in (1, 4):
        reason = report.cases[index].reason
        assert "truncated at the token limit" in reason
        assert "scored as a failure under the registered truncation policy" in reason
    assert report.cases[2].reason == "exact mismatch"
    summary = report.summary
    assert (summary.total, summary.scored, summary.passed, summary.failed, summary.unscored) == (
        5,
        5,
        2,
        3,
        0,
    )
    assert summary.accuracy == pytest.approx(0.4)
    assert (summary.finished, summary.finished_passed) == (3, 2)
    assert (report.targets.total, report.targets.passed) == (3, 1)
    assert (report.targets.finished, report.targets.finished_passed) == (2, 1)
    assert (report.regressions.finished, report.regressions.finished_passed) == (1, 1)
    assert report.regressions.accuracy == pytest.approx(0.5)
    assert report.groups["reasoning"] == report.targets
    assert report.groups["locality"] == report.regressions
    data = report.model_dump(mode="json")
    assert data["truncation"] == "fail"
    assert data["summary"]["finished"] == 3 and data["cases"][1]["finished"] is False


@pytest.mark.parametrize(
    ("scenario", "finished_elsewhere"),
    [
        ("failed_inference", 1),
        ("missing_sample", 1),
        ("missing_output", 1),
        ("wrong_provenance", 1),
        ("unusable_reference", 1),
        ("forged_digest", 0),
        ("teacher_forced", 0),
    ],
)
def test_fail_policy_never_scores_infrastructure_failures_or_diagnostics(
    scenario: str, finished_elsewhere: int
) -> None:
    continuation = "next" if scenario == "teacher_forced" else None
    first = CaseSpec(
        id="first",
        input="a",
        expected="not a number" if scenario == "unusable_reference" else "1",
        scorer="numeric",
        continuation=continuation,
    )
    second = CaseSpec(
        id="second", input="b", expected="B", role="regression", continuation=continuation
    )
    cut_off = sample(first, "1", cut_off=True)
    spec_updates: dict[str, object] = {}
    if scenario == "failed_inference":
        cut_off = sample(
            first,
            "1",
            cut_off=True,
            status="failed",
            error={"code": "timeout", "message": "timeout"},
        )
    elif scenario == "missing_output":
        cut_off = sample(first, None, cut_off=True)
    elif scenario == "wrong_provenance":
        cut_off = sample(first, "1", cut_off=True, provenance="huggingface")
    elif scenario == "teacher_forced":
        cut_off = sample(first, "", truncated=True, finish_reason="teacher_forced")
        spec_updates["diagnostics"] = DiagnosticSpec()
    samples = [cut_off, sample(second, "B")]
    if scenario == "missing_sample":
        samples = samples[1:]
    run = record([first, second], samples, **spec_updates)
    if scenario == "forged_digest":
        run = run.model_copy(update={"spec_digest": "forged"})
    for policy in ("unscored", "fail"):
        report = evaluate_run(run, policy)
        case = report.cases[0]
        assert (case.outcome, case.score, case.finished) == ("unscored", None, None)
        assert "scored as a failure" not in case.reason
        assert not report.complete
        assert (report.targets.passed, report.targets.failed) == (0, 0)
        assert report.summary.accuracy is None
    strict = evaluate_run(run, "fail").summary
    assert (strict.finished, strict.finished_passed) == (finished_elsewhere, finished_elsewhere)


def test_comparison_can_adjudicate_a_baseline_that_hits_the_token_limit() -> None:
    cases = [
        CaseSpec(id="reason-1", input="a", expected="A"),
        CaseSpec(id="reason-2", input="b", expected="B"),
        CaseSpec(id="local", input="c", expected="C", role="regression"),
    ]
    looping = record(
        cases,
        [
            sample(cases[0], "thinking", cut_off=True),
            sample(cases[1], "thinking", cut_off=True),
            sample(cases[2], "C"),
        ],
    )
    answering = record(cases, [sample(case, case.expected) for case in cases])
    default = compare_runs(looping, answering)
    assert not default.comparable and default.verdict == "inconclusive"
    assert default.issues == ["baseline is not a complete scored run"]
    assert default.improved_cases == []
    registered = compare_runs(looping, answering, "fail")
    assert registered.comparable and registered.verdict == "observed_improvement"
    assert registered.improved_cases == ["reason-1", "reason-2"]
    assert registered.unchanged_cases == ["local"]
    assert registered.accuracy_delta == pytest.approx(2 / 3)
    assert registered.baseline.truncation == registered.candidate.truncation == "fail"
    assert (registered.baseline.summary.finished, registered.candidate.summary.finished) == (1, 3)
    # Finishing is reported separately and is never itself a pass.
    reverse = compare_runs(answering, looping, "fail")
    assert reverse.verdict == "observed_regression"


def study(targets: int = 6, regressions: int = 6, **updates: object) -> StudySpec:
    data: dict[str, object] = {
        "schema_version": "0.3",
        "name": "synthetic referee design",
        "development_cases": [
            *(
                {"id": f"target-{index}", "input": f"t{index}", "expected": f"T{index}"}
                for index in range(targets)
            ),
            *(
                {
                    "id": f"local-{index}",
                    "input": f"r{index}",
                    "expected": f"R{index}",
                    "role": "regression",
                }
                for index in range(regressions)
            ),
        ],
    }
    data.update(updates)
    return StudySpec.model_validate(data)


def role_ids(spec: StudySpec, role: str) -> set[str]:
    return {case.id for case in spec.development_cases if case.role == role}


def nothing(_name: str, _seed: int) -> set[str]:
    return set()


def arms(
    spec: StudySpec,
    passing: Passing,
    cut_off: Passing = nothing,
    cases: list[CaseSpec] | None = None,
) -> Arms:
    arm_cases = spec.development_cases if cases is None else cases
    result: Arms = []
    for seed in study_seeds(spec):
        for name in study_arm_names(spec):
            arm = study_arm_name(spec, name, seed)
            samples = [
                sample(
                    case,
                    case.expected if case.id in passing(name, seed) else "wrong",
                    cut_off=case.id in cut_off(name, seed),
                )
                for case in arm_cases
            ]
            run = record(
                arm_cases,
                samples,
                generation=study_arm_generation(spec, name, seed),
                identifier=arm,
            )
            result.append((arm, run))
    return result


def candidate_repairs(spec: StudySpec) -> Passing:
    """Every arm preserves the regression cases; only the candidate solves the targets."""
    local, everything = role_ids(spec, "regression"), {case.id for case in spec.development_cases}
    return lambda name, _seed: everything if name == "candidate" else local


def test_inference_supports_a_candidate_that_beats_the_original_and_every_control() -> None:
    spec = study(inference=RULE, controls=[{"name": "placebo", "program": {"kind": "identity"}}])
    report = assess_study_arms(spec, arms(spec, candidate_repairs(spec)))
    assert report.verdict == "observed_improvement"
    assert report.recommendation == "candidate_supported"
    assert report.issues == []
    # Synthetic tiny-model records can support the engineering path, never model quality.
    assert report.scope == "engineering_fixture"
    references = ["baseline", "no_op", "control:placebo"]
    assert [pair.reference for pair in report.inference] == references
    assert [pair.reference for pair in report.comparisons] == references
    for pair in report.inference:
        assert pair.target_superior and not pair.target_harmed
        assert pair.regression_non_inferior and not pair.regression_inferior
        assert (pair.target.cases, pair.target.seeds, pair.target.improved_cases) == (6, 1, 6)
        assert (pair.target.mean_difference, pair.target.ci_low) == (1.0, 1.0)
        assert pair.target.sign_test_p == pytest.approx(2 / 64)
        assert pair.target.confidence == pytest.approx(0.95)
        assert (pair.regression.tied_cases, pair.regression.mean_difference) == (6, 0.0)
    assert all(pair.improved_targets == 6 for pair in report.comparisons)


def test_inference_is_inconclusive_when_a_registered_comparator_matches_the_candidate() -> None:
    spec = study(
        inference=RULE,
        generation={"enable_thinking": True, "max_new_tokens": 64},
        comparators=[
            {"name": "thinking-off", "kind": "generation", "generation": {"max_new_tokens": 64}}
        ],
    )
    local, everything = role_ids(spec, "regression"), {case.id for case in spec.development_cases}
    report = assess_study_arms(
        spec,
        arms(spec, lambda name, _seed: local if name in {"baseline", "no_op"} else everything),
    )
    assert report.verdict == "inconclusive"
    assert report.recommendation == "retain_original"
    assert report.issues == [CONTROL_PARITY]
    assert [pair.reference for pair in report.inference] == [
        "baseline",
        "no_op",
        "comparator:thinking-off",
    ]
    assert report.inference[0].target_superior
    control = report.inference[-1]
    assert not control.target_superior and not control.target_harmed
    assert (control.target.mean_difference, control.target.tied_cases) == (0.0, 6)
    assert control.target.sign_test_p == 1.0
    # Descriptive counts stay available beside the registered rule.
    assert [pair.improved_targets for pair in report.comparisons] == [6, 6, 0]
    assert all(pair.comparable for pair in report.comparisons)


def test_one_flipped_regression_case_among_many_is_not_a_regression() -> None:
    def one_flip(spec: StudySpec) -> Passing:
        local = role_ids(spec, "regression")
        repaired = ({case.id for case in spec.development_cases}) - {"local-0"}
        return lambda name, _seed: repaired if name == "candidate" else local

    counted = study(targets=12, regressions=40)
    assert assess_study_arms(counted, arms(counted, one_flip(counted))).verdict == "regressed"

    registered = study(
        targets=12, regressions=40, inference={"resamples": 2000, "regression_margin": 0.1}
    )
    report = assess_study_arms(registered, arms(registered, one_flip(registered)))
    assert report.comparisons[0].regressed_regressions == 1
    base = report.inference[0]
    assert (base.regression.cases, base.regression.regressed_cases) == (40, 1)
    assert base.regression.mean_difference == pytest.approx(-1 / 40)
    assert base.regression_non_inferior and not base.regression_inferior
    assert report.verdict == "observed_improvement"
    assert report.recommendation == "candidate_supported"

    # A tighter registered margin withholds support, but one flip is still not a regression.
    strict = study(targets=12, regressions=40, inference={"resamples": 2000})
    report = assess_study_arms(strict, arms(strict, one_flip(strict)))
    assert report.verdict == "inconclusive"
    assert report.recommendation == "retain_original"
    assert report.issues == ["regression non-inferiority against the original was not established"]
    assert not report.inference[0].regression_inferior


@pytest.mark.parametrize("harmed", ["target", "regression"])
def test_inference_reports_significant_harm_as_regressed(harmed: str) -> None:
    spec = study(inference=RULE)
    everything = {case.id for case in spec.development_cases}
    survivors = everything - role_ids(spec, harmed)
    report = assess_study_arms(
        spec, arms(spec, lambda name, _seed: survivors if name == "candidate" else everything)
    )
    assert report.verdict == "regressed"
    assert report.recommendation == "retain_original"
    assert report.issues == []
    base = report.inference[0]
    if harmed == "target":
        assert base.target_harmed and not base.target_superior
        assert (base.target.mean_difference, base.target.ci_high) == (-1.0, -1.0)
        assert base.regression_non_inferior
    else:
        assert base.regression_inferior and not base.regression_non_inferior
        assert base.regression.regressed_cases == 6
    assert report.comparisons[0].improved_targets == 0


def test_inference_without_an_effect_is_no_change() -> None:
    spec = study(inference=RULE)
    local = role_ids(spec, "regression")
    report = assess_study_arms(spec, arms(spec, lambda _name, _seed: local))
    assert (report.verdict, report.recommendation) == ("no_change", "retain_original")
    assert report.issues == [] and len(report.inference) == 2
    assert not report.inference[0].target_superior


def test_seed_repeats_are_nested_in_cases_and_never_inflate_the_sample() -> None:
    spec = study(inference=RULE, seeds=[3, 5, 8])
    report = assess_study_arms(spec, arms(spec, candidate_repairs(spec)))
    assert report.verdict == "observed_improvement"
    effect = report.inference[0].target
    assert (effect.cases, effect.seeds, effect.improved_cases) == (6, 3, 6)
    # Eighteen paired executions carry the evidence of six cases.
    assert report.comparisons[0].improved_targets == 18
    assert report.comparisons[0].target_variation is not None
    assert report.comparisons[0].target_variation.paired_executions == 18
    assert effect.sign_test_p == pytest.approx(2 / 64)

    # Three repaired cases repeated over three seeds are nine executions but three cases,
    # which an exact sign test can never call significant.
    local = role_ids(spec, "regression")
    partial = local | {"target-0", "target-1", "target-2"}
    report = assess_study_arms(
        spec, arms(spec, lambda name, _seed: partial if name == "candidate" else local)
    )
    effect = report.inference[0].target
    assert report.comparisons[0].improved_targets == 9
    assert (effect.improved_cases, effect.tied_cases) == (3, 3)
    assert effect.sign_test_p == pytest.approx(0.25)
    assert (report.verdict, report.recommendation) == ("no_change", "retain_original")

    # An effect present in one seed of three is a fractional case-level difference.
    everything = {case.id for case in spec.development_cases}
    report = assess_study_arms(
        spec,
        arms(spec, lambda name, seed: everything if name == "candidate" and seed == 3 else local),
    )
    effect = report.inference[0].target
    assert (effect.cases, effect.improved_cases) == (6, 6)
    assert effect.mean_difference == pytest.approx(1 / 3)


def test_inference_is_never_drawn_from_incomplete_or_underpowered_evidence() -> None:
    spec = study(inference=RULE)
    complete = arms(spec, candidate_repairs(spec))

    failed = list(complete)
    failed[-1] = ("candidate", complete[-1][1].model_copy(update={"status": "failed"}))
    report = assess_study_arms(spec, failed)
    assert (report.verdict, report.recommendation) == ("inconclusive", "retain_original")
    assert report.inference == [] and report.comparisons == []
    assert report.arms[-1].passed == 0 and not report.arms[-1].complete

    # Without the registered policy a cut-off baseline stays unscored and blocks inference.
    cut_off = arms(
        spec,
        candidate_repairs(spec),
        lambda name, _seed: role_ids(spec, "target") if name == "baseline" else set(),
    )
    report = assess_study_arms(spec, cut_off)
    assert report.verdict == "inconclusive" and report.inference == []
    assert report.issues == ["one or more arms were incomplete or unscored"]

    assert assess_study_arms(spec, complete[:-1]).inference == []

    few = [case for case in spec.development_cases if case.id in {"target-0", "local-0"}]
    report = assess_study_arms(spec, arms(spec, candidate_repairs(spec), cases=few))
    assert (report.verdict, report.recommendation) == ("inconclusive", "retain_original")
    assert report.inference == [] and report.comparisons == []
    assert report.issues == [
        "registered inference requires at least six target and six regression cases"
    ]


def test_study_truncation_policy_adjudicates_arms_that_hit_the_token_limit() -> None:
    def looping(spec: StudySpec) -> Passing:
        targets = role_ids(spec, "target")
        return lambda name, _seed: set() if name == "candidate" else targets

    registered = study(inference=RULE, truncation="fail")
    report = assess_study_arms(
        registered, arms(registered, candidate_repairs(registered), looping(registered))
    )
    assert report.scorer_version == "nrly-scorers/3" == study_scorer_version(registered)
    assert (report.verdict, report.recommendation) == (
        "observed_improvement",
        "candidate_supported",
    )
    assert all(arm.complete for arm in report.arms)
    baseline, candidate = report.arms[0], report.arms[-1]
    assert (baseline.total, baseline.passed, baseline.target_passed) == (12, 6, 0)
    assert (baseline.finished, baseline.finished_passed) == (6, 6)
    assert (candidate.finished, candidate.finished_passed, candidate.passed) == (12, 12, 12)
    assert report.inference[0].target_superior
    assert _inconclusive(registered, "attempt cancelled").scorer_version == "nrly-scorers/3"

    # A candidate that merely stops looping finishes every case and still passes none of them.
    local = role_ids(registered, "regression")
    report = assess_study_arms(
        registered, arms(registered, lambda _name, _seed: local, looping(registered))
    )
    candidate = report.arms[-1]
    assert (candidate.finished, candidate.finished_passed, candidate.target_passed) == (12, 6, 0)
    assert (report.verdict, report.recommendation) == ("no_change", "retain_original")

    # The same evidence without the registered policy cannot be adjudicated, as before.
    default = study()
    report = assess_study_arms(default, arms(default, candidate_repairs(default), looping(default)))
    assert report.scorer_version == "nrly-scorers/2" == study_scorer_version(default)
    assert (report.verdict, report.recommendation) == ("inconclusive", "retain_original")
    assert report.issues == ["one or more arms were incomplete or unscored"]
    assert report.arms[0].passed == 6 and not report.arms[0].complete
    data = report.model_dump(mode="json")
    assert "inference" not in data
    assert all("finished" not in arm and "finished_passed" not in arm for arm in data["arms"])


def test_each_arm_must_have_executed_its_registered_generation() -> None:
    spec = study(
        inference=RULE,
        seeds=[11, 29],
        comparators=[
            {
                "name": "decoding",
                "kind": "generation",
                "generation": {"repetition_penalty": 1.1, "max_new_tokens": 48},
            }
        ],
    )
    registered = GenerationSpec(repetition_penalty=1.1, max_new_tokens=48)
    valid = arms(spec, candidate_repairs(spec))
    by_arm = dict(valid)
    for seed in (11, 29):
        assert by_arm[f"comparator:decoding@seed={seed}"].spec.generation == registered.model_copy(
            update={"seed": seed}
        )
        assert by_arm[f"baseline@seed={seed}"].spec.generation == GenerationSpec(seed=seed)
    report = assess_study_arms(spec, valid)
    assert report.verdict == "observed_improvement" and len(report.inference) == 3
    assert [arm.seed for arm in report.arms] == [11] * 4 + [29] * 4

    def with_generation(position: int, generation: GenerationSpec) -> Arms:
        changed = list(valid)
        name, run = changed[position]
        executed = run.spec.model_copy(update={"generation": generation})
        changed[position] = (
            name,
            run.model_copy(update={"spec": executed, "spec_digest": digest(executed)}),
        )
        return changed

    for changed in (
        # The control ran with the study's settings instead of its registered ones.
        with_generation(2, GenerationSpec(seed=11)),
        # The original ran with the control's settings.
        with_generation(0, registered.model_copy(update={"seed": 11})),
        # The control kept its settings but used another seed block's seed.
        with_generation(6, registered.model_copy(update={"seed": 11})),
    ):
        report = assess_study_arms(spec, changed)
        assert (report.verdict, report.recommendation) == ("inconclusive", "retain_original")
        assert report.issues == ["arm generation differs from its registered seed block"]
        assert report.inference == [] and report.comparisons == []


def tiny_cases(split: str) -> list[CaseSpec]:
    sealed = split == "test"
    return [
        *(
            CaseSpec(
                id=f"{split}-target-{index}",
                input=f"[{index + 20},{index + 1}]" if sealed else f"[{index + 1},{index + 2}]",
                expected=str(2 * index + 21) if sealed else str(2 * index + 3),
                scorer="numeric",
                split=split,
                group="generalization",
            )
            for index in range(6)
        ),
        # Zero sums are unaffected by the output scale; spacing keeps every prompt distinct.
        *(
            CaseSpec(
                id=f"{split}-local-{index}",
                input=("[0.0," if sealed else "[0,") + " " * index + "0]",
                expected="0",
                scorer="numeric",
                split=split,
                role="regression",
                group="locality",
            )
            for index in range(6)
        ),
    ]


def tiny_study(**updates: object) -> StudySpec:
    data: dict[str, object] = {
        "schema_version": "0.3",
        "name": "tiny generation comparator engineering study",
        "seeds": [7],
        "development_cases": tiny_cases("discovery"),
        "comparators": [
            {
                "name": "decoding",
                "kind": "generation",
                "generation": {"max_new_tokens": 48, "repetition_penalty": 1.1},
            }
        ],
        "inference": RULE,
        "truncation": "fail",
        "mutable_targets": ["output.weight"],
        "run_budget": {"wall_seconds": 60, "max_cases": 12, "max_output_tokens": 64},
    }
    data.update(updates)
    return StudySpec.model_validate(data)


def test_generation_comparator_runs_end_to_end_in_a_schema_three_fixture_study(
    tmp_path: Path,
) -> None:
    spec = tiny_study()
    registered = GenerationSpec(max_new_tokens=48, repetition_penalty=1.1, seed=7)
    original = GenerationSpec(seed=7)
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        try:
            study_record = engine.create(spec, tiny_cases("test"))
            assert (study_record.holdout.targets, study_record.holdout.regressions) == (6, 6)
            assert study_record.usage.final_runs_reserved == 4
            candidate = CandidateSpec(
                name="repair",
                program=ProgramSpec(kind="weight_scale", target="output.weight", factor=2),
            )
            attempt = engine.wait(
                study_record.id, engine.submit_candidate(study_record.id, candidate).id, 240
            )
            assert attempt.status == "succeeded", attempt
            assessment = attempt.assessment
            assert assessment is not None
            assert [arm.arm for arm in assessment.arms] == [
                "baseline@seed=7",
                "no_op@seed=7",
                "comparator:decoding@seed=7",
                "candidate@seed=7",
            ]
            assert (assessment.verdict, assessment.scope) == (
                "observed_improvement",
                "engineering_fixture",
            )
            assert assessment.scorer_version == "nrly-scorers/3"
            assert [pair.reference for pair in assessment.inference] == [
                "baseline",
                "no_op",
                "comparator:decoding",
            ]
            assert all(pair.target_superior for pair in assessment.inference)
            assert all(pair.target.cases == 6 for pair in assessment.inference)
            assert [(arm.target_passed, arm.finished) for arm in assessment.arms] == [
                (0, 12),
                (0, 12),
                (0, 12),
                (6, 12),
            ]

            def development_run(arm: str) -> RunRecord:
                with engine._read() as connection:
                    row = connection.execute(
                        "SELECT run_id FROM development_runs WHERE study_id=? AND attempt_id=?"
                        " AND arm=?",
                        (study_record.id, attempt.id, arm),
                    ).fetchone()
                return engine.private_runner.store.get(row[0])

            control = development_run("comparator:decoding@seed=7")
            assert control.spec.generation == registered
            assert control.spec.schema_version == "0.3"
            # The control decodes the unedited original model.
            assert control.spec.program.kind == "identity"
            assert control.spec.model == spec.model
            assert control.spec.cases == spec.development_cases
            assert all(not sample_.weight_changes for sample_ in control.samples)
            assert development_run("baseline@seed=7").spec.generation == original
            assert development_run("no_op@seed=7").spec.generation == original
            edited = development_run("candidate@seed=7")
            assert edited.spec.generation == original
            assert edited.spec.program.kind == "weight_scale"
            page = engine.development_samples(
                study_record.id, attempt.id, "comparator:decoding@seed=7"
            )
            assert page["samples"][0]["output"] == "1.5"

            engine.finalize(study_record.id, attempt.id)
            final = engine.wait_final(study_record.id, 240)
            assert final.status == "completed" and final.final_error_code is None
            result = final.final_result
            assert result is not None
            assert (result.verdict, result.recommendation, result.scope) == (
                "observed_improvement",
                "candidate_supported",
                "engineering_fixture",
            )
            assert len(result.inference) == 3 and result.issues == []
            assert final.usage.runs == 8 and final.usage.attempts == 1
            sealed_runs = [
                run
                for run in engine.private_runner.store.list()
                if run.spec.cases[0].split == "test"
            ]
            assert len(sealed_runs) == 4
            assert sorted(
                (run.spec.program.kind, run.spec.generation == registered) for run in sealed_runs
            ) == [
                ("identity", False),
                ("identity", False),
                ("identity", True),
                ("weight_scale", False),
            ]
            assert all(
                run.spec.generation in (registered, original) and run.status == "succeeded"
                for run in sealed_runs
            )
            public = json.dumps(
                {
                    "study": final.model_dump(mode="json"),
                    "attempt": attempt.model_dump(mode="json"),
                    "context": engine.context_snapshot(study_record.id),
                }
            )
            assert "test-target" not in public and "test-local" not in public
            assert "[20,1]" not in public
            assert runner.store.list() == []
        finally:
            engine.close()


def test_inference_studies_refuse_an_undersized_sealed_holdout(tmp_path: Path) -> None:
    spec = tiny_study()
    holdout = tiny_cases("test")
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        try:
            for short in (holdout[1:], holdout[:-1]):
                with pytest.raises(NrlyError, match="six target and six regression") as refused:
                    engine.create(spec, short)
                assert refused.value.code == "invalid_holdout"
            assert engine.list() == []
            # The count-based rule keeps its original, smaller admission floor.
            counted = engine.create(tiny_study(inference=None), holdout[1:])
            assert (counted.holdout.targets, counted.holdout.regressions) == (5, 6)
        finally:
            engine.close()


def test_sealed_holdout_is_validated_under_every_registered_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tiny_study()
    validated: list[tuple[str, GenerationSpec, int]] = []

    def recording(**fields: object) -> ExperimentSpec:
        experiment = ExperimentSpec.model_validate(fields)
        validated.append((experiment.schema_version, experiment.generation, len(experiment.cases)))
        return experiment

    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        monkeypatch.setattr(studies_module, "ExperimentSpec", recording)
        try:
            engine.create(spec, tiny_cases("test"))
        finally:
            engine.close()
    assert validated == [
        ("0.3", GenerationSpec(), 12),
        ("0.3", GenerationSpec(max_new_tokens=48, repetition_penalty=1.1), 12),
    ]


@pytest.mark.parametrize("schema_version", ["0.1", "0.2"])
@pytest.mark.parametrize(
    "updates",
    [
        {"truncation": "fail"},
        {"inference": {}},
        {
            "comparators": [
                {"name": "decoding", "kind": "generation", "generation": {"max_new_tokens": 48}}
            ]
        },
    ],
)
def test_earlier_study_schemas_reject_the_referee_fields(
    schema_version: str, updates: dict[str, object]
) -> None:
    legacy = study(schema_version=schema_version)
    data = legacy.model_dump(mode="json")
    assert "truncation" not in data and "inference" not in data and "comparators" not in data
    with pytest.raises(ValueError, match="require study schema 0.3"):
        study(schema_version=schema_version, **updates)


def test_earlier_study_schema_assessments_keep_the_count_rule_and_serialization() -> None:
    legacy = study(schema_version="0.2", targets=1, regressions=1)
    report = assess_study_arms(legacy, arms(legacy, candidate_repairs(legacy)))
    assert report.verdict == "observed_improvement" and report.inference == []
    data = report.model_dump(mode="json")
    assert list(data) == [
        "scorer_version",
        "verdict",
        "recommendation",
        "scope",
        "arms",
        "comparisons",
        "issues",
    ]
    assert data["scorer_version"] == "nrly-scorers/2"
    assert all("finished" not in arm for arm in data["arms"])


def test_schema_three_admission_bounds_inference_and_generation_comparators() -> None:
    with pytest.raises(ValueError, match="at least six target and six regression"):
        study(targets=5, inference=RULE)
    with pytest.raises(ValueError, match="must change a decoding setting"):
        study(comparators=[{"name": "same", "kind": "generation", "generation": {}}])
    with pytest.raises(ValueError, match="canonical seed 0"):
        study(comparators=[{"name": "seeded", "kind": "generation", "generation": {"seed": 3}}])
