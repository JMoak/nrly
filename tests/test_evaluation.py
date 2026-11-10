from __future__ import annotations

import subprocess
import sys
from typing import Literal

import pytest

from nrly.contracts import (
    BudgetSpec,
    CaptureSpec,
    CaseSpec,
    ErrorInfo,
    ExperimentSpec,
    GenerationSpec,
    ModelIdentity,
    ModelSpec,
    ProgramSpec,
    RunRecord,
    RunStatus,
    SampleResult,
    digest,
)
from nrly.evaluation import compare_runs, evaluate_run
from nrly.evaluation.reports import _parse_number


@pytest.mark.parametrize("value", ["Answer: 2", "Answer: 2 . \t\n", "Answer: 2! \n", "2. \r\n"])
def test_final_number_keeps_whitespace_and_punctuation_compatibility(value: str) -> None:
    assert _parse_number(value, final=True) == 2


def test_malformed_long_answer_does_not_stall_final_number_evaluation() -> None:
    # A child timeout prevents a regex regression from hanging the test process.
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from nrly.evaluation.reports import _parse_number; "
            "assert _parse_number('1' + ' ' * 100000 + 'x', final=True) is None",
        ],
        timeout=5,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr.decode(errors="replace")


def recorded(
    outputs: list[str | None],
    *,
    cases: list[CaseSpec] | None = None,
    status: RunStatus = "succeeded",
    provenance: Literal["torch_fixture", "huggingface", "unavailable"] = "torch_fixture",
) -> RunRecord:
    if cases is None:
        cases = [
            CaseSpec(id="sum", input="[2, 4]", expected="6", scorer="numeric"),
            CaseSpec(id="zero", input="[0, 0]", expected="0", scorer="numeric", role="regression"),
        ]
    spec = ExperimentSpec(name="Fixture comparison", cases=cases)
    return RunRecord(
        id="run-fixture",
        status=status,
        spec=spec,
        spec_digest=digest(spec),
        created_at="2026-11-10T00:00:00Z",
        model_identity=ModelIdentity(
            backend="tiny",
            model_id="nrly/tiny-sum-v1",
            revision="1",
            fingerprint="a" * 64,
        ),
        samples=[
            SampleResult(
                case_id=case.id,
                status="completed",
                provenance=provenance,
                output=output,
            )
            for case, output in zip(cases, outputs, strict=False)
        ],
    )


def test_completed_fixture_is_explicit_engineering_evidence() -> None:
    report = evaluate_run(recorded(["6", "0"]))
    assert report.scope == "engineering_fixture"
    assert report.complete
    assert report.summary.accuracy == 1.0
    assert report.targets.passed == 1
    assert report.regressions.passed == 1


def test_incorrect_completed_experiment_is_valid_negative_evidence() -> None:
    report = evaluate_run(recorded(["3", "0"]))
    assert report.complete
    assert report.summary.passed == 1
    assert report.summary.failed == 1
    assert report.targets.accuracy == 0.0


@pytest.mark.parametrize(
    "status", ["queued", "running", "cancelling", "failed", "cancelled", "interrupted"]
)
def test_unfinished_or_failed_runs_never_count_as_passes(status: RunStatus) -> None:
    report = evaluate_run(recorded(["6", "0"], status=status))
    assert not report.complete
    assert report.summary.passed == 0
    assert report.summary.accuracy is None
    assert report.summary.unscored == 2


@pytest.mark.parametrize("provenance", ["unavailable", "huggingface"])
def test_wrong_inference_provenance_is_unscored(
    provenance: Literal["torch_fixture", "huggingface", "unavailable"],
) -> None:
    report = evaluate_run(recorded(["6", "0"], provenance=provenance))
    assert not report.complete
    assert report.summary.passed == 0


def test_missing_samples_do_not_shrink_the_accuracy_denominator() -> None:
    report = evaluate_run(recorded(["6"]))
    assert not report.complete
    assert report.summary.total == 2
    assert report.summary.unscored == 1
    assert report.summary.accuracy is None
    assert report.cases[1].reason == "sample is missing"


def test_missing_outputs_are_unscored_even_with_completed_status() -> None:
    report = evaluate_run(recorded([None, "0"]))
    assert not report.complete
    assert report.cases[0].outcome == "unscored"


def test_duplicate_samples_invalidate_the_entire_comparison() -> None:
    run = recorded(["6", "0"])
    duplicate = run.model_copy(update={"samples": [*run.samples, run.samples[0]]})
    report = compare_runs(run, duplicate)
    assert not report.comparable
    assert report.verdict == "inconclusive"
    assert "duplicate sample: sum" in report.candidate.issues
    assert report.candidate.summary.passed == 0


def test_unknown_sample_is_not_silently_dropped() -> None:
    run = recorded(["6", "0"])
    unknown = SampleResult(
        case_id="invented", status="completed", provenance="torch_fixture", output="6"
    )
    report = evaluate_run(run.model_copy(update={"samples": [*run.samples, unknown]}))
    assert not report.complete
    assert "unexpected sample: invented" in report.issues


def test_forged_spec_digest_is_not_trusted() -> None:
    report = evaluate_run(recorded(["6", "0"]).model_copy(update={"spec_digest": "bad"}))
    assert not report.complete
    assert report.summary.passed == 0


def test_missing_identity_cannot_establish_real_inference() -> None:
    report = evaluate_run(recorded(["6", "0"]).model_copy(update={"model_identity": None}))
    assert not report.complete
    assert report.summary.passed == 0


def test_success_status_cannot_override_execution_error() -> None:
    run = recorded(["6", "0"]).model_copy(
        update={"error": ErrorInfo(code="worker_lost", message="worker disappeared")}
    )
    assert not evaluate_run(run).complete


def test_failed_sample_with_an_output_is_unscored() -> None:
    run = recorded(["6", "0"])
    failed = run.samples[0].model_copy(
        update={"status": "failed", "error": ErrorInfo(code="timeout", message="timeout")}
    )
    report = evaluate_run(run.model_copy(update={"samples": [failed, run.samples[1]]}))
    assert report.cases[0].outcome == "unscored"
    assert report.summary.accuracy is None


@pytest.mark.parametrize("expected", [None, "NaN", "infinity", "not a number"])
def test_invalid_reference_is_unscored_not_a_model_failure(expected: str | None) -> None:
    case = CaseSpec(id="bad-reference", input="[2, 4]", expected=expected, scorer="numeric")
    report = evaluate_run(recorded(["6"], cases=[case]))
    assert report.summary.unscored == 1
    assert report.summary.failed == 0


@pytest.mark.parametrize("output", ["NaN", "Infinity", "1e9999", "6 trailing", "almost 6"])
def test_invalid_numeric_model_answer_is_a_task_failure(output: str) -> None:
    report = evaluate_run(recorded([output, "0"]))
    assert report.complete
    assert report.cases[0].outcome == "fail"


def test_numeric_tolerance_is_absolute_and_explicit() -> None:
    case = CaseSpec(id="precision", input="[2, 4]", expected="6", scorer="numeric", tolerance=0.001)
    assert evaluate_run(recorded(["6.0005"], cases=[case])).summary.passed == 1
    assert evaluate_run(recorded(["6.002"], cases=[case])).summary.failed == 1


def test_numeric_scorer_does_not_round_distinct_large_integers_to_the_same_float() -> None:
    case = CaseSpec(
        id="large-number",
        input="Return the precise integer.",
        expected="9007199254740993",
        scorer="numeric",
        tolerance=0,
    )
    assert evaluate_run(recorded(["9007199254740992"], cases=[case])).summary.failed == 1
    assert evaluate_run(recorded(["9007199254740993"], cases=[case])).summary.passed == 1


@pytest.mark.parametrize(
    ("output", "passes"),
    [
        ("ANSWER: 42", True),
        ("The result is 42.", True),
        ("<think>Maybe 17.</think>\nANSWER: 42", True),
        ("Maybe 17.</think>\nANSWER: 42", True),
        ("<think>42</think>", False),
        ("<think>I think 42", False),
        ("42 or 17", False),
        ("42 and some additional words", False),
        ("142", False),
        ("1,042", False),
    ],
)
def test_final_number_requires_a_final_visible_numeric_answer(output: str, passes: bool) -> None:
    case = CaseSpec(id="answer", input="Return the answer.", expected="42", scorer="final_number")
    report = evaluate_run(recorded([output], cases=[case]))
    assert (report.cases[0].outcome == "pass") is passes


def test_exact_scoring_preserves_case_and_internal_whitespace() -> None:
    case = CaseSpec(id="text", input="Return A B.", expected="A B")
    assert evaluate_run(recorded([" A B\n"], cases=[case])).summary.passed == 1
    assert evaluate_run(recorded(["a b"], cases=[case])).summary.failed == 1
    assert evaluate_run(recorded(["A  B"], cases=[case])).summary.failed == 1


def test_comparison_reports_repair_and_preserved_regression_case() -> None:
    comparison = compare_runs(recorded(["3", "0"]), recorded(["6", "0"]))
    assert comparison.comparable
    assert comparison.verdict == "observed_improvement"
    assert comparison.improved_cases == ["sum"]
    assert comparison.regressed_cases == []
    assert comparison.unchanged_cases == ["zero"]
    assert comparison.accuracy_delta == 0.5


def test_improvement_cannot_hide_a_regression() -> None:
    comparison = compare_runs(recorded(["3", "0"]), recorded(["6", "1"]))
    assert comparison.comparable
    assert comparison.verdict == "mixed"
    assert comparison.improved_cases == ["sum"]
    assert comparison.regressed_cases == ["zero"]
    assert comparison.accuracy_delta == 0.0


def test_unchanged_negative_candidate_is_no_change() -> None:
    comparison = compare_runs(recorded(["3", "0"]), recorded(["3", "0"]))
    assert comparison.comparable
    assert comparison.verdict == "no_change"


def test_harmful_candidate_is_a_completed_comparison() -> None:
    comparison = compare_runs(recorded(["6", "0"]), recorded(["3", "0"]))
    assert comparison.comparable
    assert comparison.verdict == "observed_regression"


def test_incomplete_candidate_is_inconclusive() -> None:
    comparison = compare_runs(recorded(["3", "0"]), recorded(["6"]))
    assert not comparison.comparable
    assert comparison.verdict == "inconclusive"
    assert comparison.accuracy_delta is None
    assert comparison.improved_cases == []


def test_changed_case_expectation_prevents_comparison() -> None:
    baseline = recorded(["3", "0"])
    cases = [baseline.spec.cases[0].model_copy(update={"expected": "3"}), baseline.spec.cases[1]]
    comparison = compare_runs(baseline, recorded(["3", "0"], cases=cases))
    assert not comparison.comparable
    assert any("scorer settings" in issue for issue in comparison.issues)


def test_changed_generation_settings_prevent_comparison() -> None:
    baseline = recorded(["3", "0"])
    spec = baseline.spec.model_copy(update={"generation": GenerationSpec(seed=9)})
    candidate = baseline.model_copy(update={"spec": spec, "spec_digest": digest(spec)})
    assert not compare_runs(baseline, candidate).comparable


def test_changed_model_fingerprint_prevents_comparison() -> None:
    baseline = recorded(["3", "0"])
    assert baseline.model_identity is not None
    candidate = baseline.model_copy(
        update={
            "model_identity": baseline.model_identity.model_copy(update={"fingerprint": "b" * 64})
        }
    )
    assert not compare_runs(baseline, candidate).comparable


def test_capture_budget_and_program_can_differ_between_comparable_runs() -> None:
    baseline = recorded(["3", "0"])
    candidate = recorded(["6", "0"])
    spec = candidate.spec.model_copy(
        update={
            "program": ProgramSpec(kind="weight_scale", target="output.weight", factor=2),
            "budget": BudgetSpec(wall_seconds=60),
            "captures": [CaptureSpec(target="hidden")],
        }
    )
    candidate = candidate.model_copy(update={"spec": spec, "spec_digest": digest(spec)})
    assert compare_runs(baseline, candidate).comparable


def test_real_model_requires_real_model_provenance() -> None:
    run = recorded(["6", "0"])
    model = ModelSpec(backend="huggingface", model_id="Qwen/Qwen3-0.6B", revision="a" * 40)
    spec = run.spec.model_copy(update={"model": model})
    identity = ModelIdentity(
        backend="huggingface",
        model_id=model.model_id,
        revision=model.revision,
        fingerprint="b" * 64,
    )
    run = run.model_copy(
        update={"spec": spec, "spec_digest": digest(spec), "model_identity": identity}
    )
    assert not evaluate_run(run).complete
    samples = [sample.model_copy(update={"provenance": "huggingface"}) for sample in run.samples]
    report = evaluate_run(run.model_copy(update={"samples": samples}))
    assert report.complete
    assert report.scope == "real_model"
