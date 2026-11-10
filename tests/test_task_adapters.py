from __future__ import annotations

import pytest

from nrly.contracts import (
    CaseSpec,
    DiagnosticSpec,
    ExperimentSpec,
    ModelIdentity,
    RunRecord,
    SampleResult,
    TaskAdapterSpec,
    digest,
)
from nrly.evaluation import evaluate_run
from nrly.evaluation.adapters import score_adapter


@pytest.mark.parametrize(
    "output,passed",
    [("A", True), (" A\n", True), ("Answer: A", False), ("A or B", False), ("B", False)],
)
def test_multiple_choice_is_strict_label_scoring_not_loglikelihood(
    output: str, passed: bool
) -> None:
    case = CaseSpec(
        id="choice",
        input="Select",
        expected="A",
        adapter=TaskAdapterSpec(kind="multiple_choice", choices=["A", "B"]),
    )
    assert score_adapter(case, output)[0] is passed


@pytest.mark.parametrize(
    "expected,output,passed",
    [
        ('{"a":1,"b":[2,true]}', '{"b":[2,true],"a":1.0}', True),
        ('{"a":true}', '{"a":1}', False),
        ('{"a":1}', '{"a":1,"a":1}', False),
        ('{"a":1}', '```json\n{"a":1}\n```', False),
        ("null", "NaN", False),
        ("null", "1e9999", False),
        ("[1,2]", "[2,1]", False),
        ("{}", '{"x":1}', False),
        ("NaN", "NaN", None),
        ('{"a":1,"a":1}', '{"a":1}', None),
    ],
)
def test_json_adapter_uses_strict_structural_equality(
    expected: str,
    output: str,
    passed: bool | None,
) -> None:
    case = CaseSpec(
        id="json", input="Return JSON", expected=expected, adapter=TaskAdapterSpec(kind="json")
    )
    assert score_adapter(case, output)[0] is passed


def test_json_depth_ceiling_is_explicit_and_bounded() -> None:
    case = CaseSpec(
        id="json", input="Return JSON", expected="null", adapter=TaskAdapterSpec(kind="json")
    )
    assert score_adapter(case, "[" * 65 + "null" + "]" * 65)[0] is False


@pytest.mark.parametrize(
    "output,passed",
    [("Hello", True), ("hello", False), ("Hello bad", False), ("Hello world!", False)],
)
def test_literal_instruction_constraints_are_bounded_case_sensitive_checks(
    output: str,
    passed: bool,
) -> None:
    case = CaseSpec(
        id="constraint",
        input="Say Hello",
        expected=None,
        adapter=TaskAdapterSpec(
            kind="instruction_constraints", required=["Hello"], forbidden=["bad"], max_chars=10
        ),
    )
    assert score_adapter(case, output)[0] is passed


@pytest.mark.parametrize(
    "adapter",
    [
        {"kind": "multiple_choice", "choices": ["A", "A"]},
        {"kind": "multiple_choice", "choices": [" A"]},
        {"kind": "multiple_choice", "choices": ["A"], "required": ["x"]},
        {"kind": "json", "max_chars": 1},
        {"kind": "instruction_constraints"},
        {"kind": "instruction_constraints", "required": ["x"], "forbidden": ["x"]},
    ],
)
def test_ambiguous_adapter_contracts_are_rejected(adapter: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TaskAdapterSpec.model_validate(adapter)


def recorded(case: CaseSpec, output: str, **sample_updates: object) -> RunRecord:
    experiment = ExperimentSpec(
        schema_version="0.3",
        name="scoring fixture",
        cases=[case],
        diagnostics=DiagnosticSpec() if case.continuation else None,
    )
    sample = SampleResult.model_validate(
        {
            "case_id": case.id,
            "status": "completed",
            "provenance": "torch_fixture",
            "output": output,
            **sample_updates,
        }
    )
    return RunRecord(
        id="fixture",
        status="succeeded",
        created_at="unit",
        spec=experiment,
        spec_digest=digest(experiment),
        samples=[sample],
        model_identity=ModelIdentity(
            backend="tiny", model_id="nrly/tiny-sum-v1", revision="1", fingerprint="unit"
        ),
    )


def test_groups_are_reported_and_adapters_are_versioned_per_case() -> None:
    case = CaseSpec(
        id="target",
        input="Return JSON",
        expected='{"a":1}',
        group="generalization",
        adapter=TaskAdapterSpec(kind="json"),
    )
    report = evaluate_run(recorded(case, '{"a":1}'))
    assert report.complete and report.groups["generalization"].passed == 1
    assert report.cases[0].scorer == "nrly-tasks/1/json"


@pytest.mark.parametrize(
    "updates",
    [
        {"truncated": True},
        {"finish_reason": "teacher_forced"},
        {"finish_reason": "length"},
        {"status": "failed"},
    ],
)
def test_truncated_diagnostic_or_failed_outputs_never_score_as_task_success(
    updates: dict[str, object],
) -> None:
    case = CaseSpec(id="case", input="Answer", expected="A")
    report = evaluate_run(recorded(case, "A", **updates))
    assert not report.complete
    assert report.summary.passed == 0 and report.summary.unscored == 1


def test_empty_teacher_forced_output_cannot_pass_empty_expected() -> None:
    case = CaseSpec(id="diagnostic", input="Prompt", continuation="target", expected="")
    report = evaluate_run(recorded(case, "", finish_reason="teacher_forced", truncated=False))
    assert not report.complete and report.summary.passed == 0


def test_legacy_unknown_truncation_remains_replayable() -> None:
    case = CaseSpec(id="case", input="Answer", expected="A")
    report = evaluate_run(recorded(case, "A"))
    assert report.complete and report.summary.passed == 1


def test_diagnostic_spec_prevents_task_scoring_even_if_sample_finish_reason_is_missing() -> None:
    case = CaseSpec(id="diagnostic", input="Prompt", continuation="answer", expected="answer")
    report = evaluate_run(recorded(case, "answer"))
    assert not report.complete and report.summary.passed == 0


@pytest.mark.parametrize(
    "expected,output,passed",
    [
        ("0.10000000000000001", "0.1", False),
        ("9007199254740993.0", "9007199254740992", False),
        ('{"number":0.10}', '{"number":0.1}', True),
        ("1e10001", "1e10001", None),
        ("1e10000", "1e10001", False),
        ("9" * 4097, "9" * 4097, None),
    ],
)
def test_json_numeric_equality_preserves_decimal_precision_and_explicit_bounds(
    expected: str,
    output: str,
    passed: bool | None,
) -> None:
    case = CaseSpec(
        id="json", input="Return JSON", expected=expected, adapter=TaskAdapterSpec(kind="json")
    )
    assert score_adapter(case, output)[0] is passed


def test_revised_completion_policy_has_new_scorer_version_and_preserves_run_digest() -> None:
    case = CaseSpec(id="historical", input="Answer", expected="A")
    run = recorded(case, "A", finish_reason="length")
    original = run.model_dump_json()
    report = evaluate_run(run)
    assert report.scorer_version == "nrly-scorers/2"
    assert not report.complete and report.summary.unscored == 1
    assert run.model_dump_json() == original and run.spec_digest == digest(run.spec)
