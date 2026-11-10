from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, DecimalException, localcontext
from typing import Literal

from pydantic import Field

from nrly.contracts import CaseSpec, Contract, RunRecord, SampleResult, digest
from nrly.evaluation.adapters import score_adapter

SCORER_VERSION: Literal["nrly-scorers/2"] = "nrly-scorers/2"

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_WHOLE_NUMBER = re.compile(rf"\s*({NUMBER})\s*\Z")
_FINAL_NUMBER = re.compile(rf"(?<![\w.,+-])({NUMBER})\s*[.!]?\Z")


class CaseEvaluation(Contract):
    case_id: str
    split: Literal["discovery", "validation", "test"]
    role: Literal["target", "regression"]
    outcome: Literal["pass", "fail", "unscored"]
    output: str | None
    expected: str | None
    scorer: str
    score: float | None
    reason: str
    group: str | None = Field(default=None, exclude_if=lambda v: v is None)


class GroupSummary(Contract):
    total: int
    scored: int
    passed: int
    failed: int
    unscored: int
    accuracy: float | None


class EvaluationReport(Contract):
    schema_version: Literal["0.1"] = "0.1"
    scorer_version: Literal["nrly-scorers/2"] = SCORER_VERSION
    run_id: str
    scope: Literal["engineering_fixture", "real_model"]
    complete: bool
    issues: list[str]
    summary: GroupSummary
    targets: GroupSummary
    regressions: GroupSummary
    cases: list[CaseEvaluation]
    groups: dict[str, GroupSummary] = Field(default_factory=dict, exclude_if=lambda v: not v)


class ComparisonReport(Contract):
    schema_version: Literal["0.1"] = "0.1"
    baseline_run: str
    candidate_run: str
    comparable: bool
    issues: list[str]
    verdict: Literal[
        "observed_improvement", "observed_regression", "mixed", "no_change", "inconclusive"
    ]
    accuracy_delta: float | None
    improved_cases: list[str]
    regressed_cases: list[str]
    unchanged_cases: list[str]
    baseline: EvaluationReport
    candidate: EvaluationReport


def evaluate_run(run: RunRecord) -> EvaluationReport:
    """Score recorded inference; infrastructure failures are never task passes."""
    issues = _run_issues(run)
    counts = Counter(sample.case_id for sample in run.samples)
    expected_ids = {case.id for case in run.spec.cases}
    for case_id in sorted(set(counts) - expected_ids):
        issues.append(f"unexpected sample: {case_id}")
    for case_id, count in sorted(counts.items()):
        if count > 1:
            issues.append(f"duplicate sample: {case_id}")
    samples = {sample.case_id: sample for sample in run.samples}
    cases = [_evaluate_case(case, samples.get(case.id), run, issues) for case in run.spec.cases]
    summary = _summarize(cases)
    return EvaluationReport(
        run_id=run.id,
        scope="engineering_fixture" if run.spec.model.backend == "tiny" else "real_model",
        complete=not issues and summary.unscored == 0,
        issues=issues,
        summary=summary,
        targets=_summarize([case for case in cases if case.role == "target"]),
        regressions=_summarize([case for case in cases if case.role == "regression"]),
        cases=cases,
        groups={
            group: _summarize([case for case in cases if case.group == group])
            for group in sorted({case.group for case in cases if case.group is not None})
        },
    )


def compare_runs(baseline: RunRecord, candidate: RunRecord) -> ComparisonReport:
    """Return observed paired changes only for compatible, fully scored runs."""
    baseline_report = evaluate_run(baseline)
    candidate_report = evaluate_run(candidate)
    issues = _comparison_issues(baseline, candidate)
    if not baseline_report.complete:
        issues.append("baseline is not a complete scored run")
    if not candidate_report.complete:
        issues.append("candidate is not a complete scored run")
    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []
    delta: float | None = None
    verdict: Literal[
        "observed_improvement", "observed_regression", "mixed", "no_change", "inconclusive"
    ] = "inconclusive"
    if not issues:
        for before, after in zip(baseline_report.cases, candidate_report.cases, strict=True):
            if before.outcome == after.outcome:
                unchanged.append(before.case_id)
            elif after.outcome == "pass":
                improved.append(before.case_id)
            else:
                regressed.append(before.case_id)
        before_accuracy = baseline_report.summary.accuracy
        after_accuracy = candidate_report.summary.accuracy
        if before_accuracy is not None and after_accuracy is not None:
            delta = after_accuracy - before_accuracy
        if improved and regressed:
            verdict = "mixed"
        elif improved:
            verdict = "observed_improvement"
        elif regressed:
            verdict = "observed_regression"
        else:
            verdict = "no_change"
    return ComparisonReport(
        baseline_run=baseline.id,
        candidate_run=candidate.id,
        comparable=not issues,
        issues=issues,
        verdict=verdict,
        accuracy_delta=delta,
        improved_cases=improved,
        regressed_cases=regressed,
        unchanged_cases=unchanged,
        baseline=baseline_report,
        candidate=candidate_report,
    )


def _run_issues(run: RunRecord) -> list[str]:
    issues: list[str] = []
    if run.spec.purpose != "experiment":
        issues.append("inspection runs have no task inference to score")
    if run.status != "succeeded":
        issues.append(f"run status is {run.status}")
    if run.error is not None:
        issues.append(f"run has an execution error: {run.error.code}")
    if run.spec_digest != digest(run.spec):
        issues.append("experiment digest does not match its specification")
    identity = run.model_identity
    if identity is None:
        issues.append("model identity is missing")
    elif not identity.fingerprint:
        issues.append("model fingerprint is missing")
    elif (identity.backend, identity.model_id, identity.revision) != (
        run.spec.model.backend,
        run.spec.model.model_id,
        run.spec.model.revision,
    ):
        issues.append("executed model does not match requested model")
    return issues


def _evaluate_case(
    case: CaseSpec,
    sample: SampleResult | None,
    run: RunRecord,
    run_issues: list[str],
) -> CaseEvaluation:
    outcome: Literal["pass", "fail", "unscored"] = "unscored"
    score: float | None = None
    reason = _sample_issue(case, sample, run, run_issues)
    if reason is None and sample is not None and sample.output is not None:
        passed, reason = _score(case, sample.output)
        if passed is not None:
            outcome = "pass" if passed else "fail"
            score = 1.0 if passed else 0.0
    return CaseEvaluation(
        case_id=case.id,
        split=case.split,
        role=case.role,
        group=case.group,
        outcome=outcome,
        output=sample.output if sample is not None else None,
        expected=case.expected,
        scorer=case.adapter.schema_version + "/" + case.adapter.kind
        if case.adapter
        else case.scorer,
        score=score,
        reason=reason or "unscored",
    )


def _sample_issue(
    case: CaseSpec,
    sample: SampleResult | None,
    run: RunRecord,
    run_issues: list[str],
) -> str | None:
    if run_issues:
        return "run integrity or completion checks failed"
    if sample is None:
        return "sample is missing"
    if sample.status != "completed" or sample.error is not None:
        return "inference failed"
    expected_provenance = "torch_fixture" if run.spec.model.backend == "tiny" else "huggingface"
    if sample.provenance != expected_provenance:
        return "inference provenance does not match the model backend"
    if sample.output is None:
        return "inference output is missing"
    if sample.truncated is True or sample.finish_reason == "length":
        return "inference output was truncated"
    if sample.finish_reason == "teacher_forced" or run.spec.diagnostics is not None:
        return "teacher-forced diagnostics are not task inference"
    if case.expected is None and (
        case.adapter is None or case.adapter.kind != "instruction_constraints"
    ):
        return "expected answer is missing"
    return None


def _score(case: CaseSpec, output: str) -> tuple[bool | None, str]:
    if case.adapter is not None:
        return score_adapter(case, output)
    if case.expected is None:
        return None, "expected answer is missing"
    if case.scorer == "exact":
        matches = output.strip() == case.expected.strip()
        return matches, "exact match" if matches else "exact mismatch"
    expected = _parse_number(case.expected, final=False)
    if expected is None:
        return None, "expected answer is not a finite number"
    actual = _parse_number(output, final=case.scorer == "final_number")
    if actual is None:
        return False, "output does not contain a finite answer in the required format"
    try:
        with localcontext() as context:
            context.prec = (
                max(28, len(actual.as_tuple().digits), len(expected.as_tuple().digits)) + 4
            )
            matches = abs(actual - expected) <= Decimal(str(case.tolerance))
    except DecimalException:
        return False, "numeric answer exceeds the scorer's supported arithmetic range"
    return matches, "numeric match" if matches else "numeric mismatch"


def _parse_number(value: str, *, final: bool) -> Decimal | None:
    if final:
        closing = value.lower().rfind("</think>")
        if closing >= 0:
            value = value[closing + len("</think>") :]
        if "<think" in value.lower():
            return None
        # Strip once instead of using adjacent optional whitespace repetitions.
        # Otherwise a long malformed suffix causes quadratic regex backtracking.
        match = _FINAL_NUMBER.search(value.rstrip())
    else:
        match = _WHOLE_NUMBER.fullmatch(value)
    if match is None:
        return None
    try:
        number = Decimal(match.group(1))
    except DecimalException:
        return None
    return number if number.is_finite() else None


def _summarize(cases: list[CaseEvaluation]) -> GroupSummary:
    passed = sum(case.outcome == "pass" for case in cases)
    failed = sum(case.outcome == "fail" for case in cases)
    unscored = len(cases) - passed - failed
    return GroupSummary(
        total=len(cases),
        scored=passed + failed,
        passed=passed,
        failed=failed,
        unscored=unscored,
        accuracy=passed / len(cases) if cases and not unscored else None,
    )


def _comparison_issues(baseline: RunRecord, candidate: RunRecord) -> list[str]:
    issues: list[str] = []
    if baseline.model_identity != candidate.model_identity:
        issues.append("executed model identities differ")
    left_model = baseline.spec.model
    right_model = candidate.spec.model
    left_settings = (
        left_model.backend,
        left_model.model_id,
        left_model.revision,
        left_model.device,
        left_model.dtype,
    )
    right_settings = (
        right_model.backend,
        right_model.model_id,
        right_model.revision,
        right_model.device,
        right_model.dtype,
    )
    if left_settings != right_settings:
        issues.append("model execution settings differ")
    if baseline.spec.generation != candidate.spec.generation:
        issues.append("generation settings differ")
    if baseline.spec.cases != candidate.spec.cases:
        issues.append("ordered cases, expectations, or scorer settings differ")
    return issues
