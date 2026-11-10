"""Conservative preregistered paired studies, with descriptive seed variation."""

from __future__ import annotations

from statistics import mean, stdev

from nrly.contracts import ModelIdentity, RunRecord, digest
from nrly.evaluation.reports import SCORER_VERSION, CaseEvaluation, EvaluationReport, evaluate_run
from nrly.study_contracts import (
    ArmSummary,
    PairedGroupSummary,
    PairedSummary,
    SeedVariation,
    StudyAssessment,
    StudyGroupSummary,
    StudySpec,
    study_arm_name,
    study_arm_names,
    study_seeds,
)


def _variation(deltas: list[list[float]]) -> SeedVariation:
    block_means = [mean(block) for block in deltas if block]
    count = len(deltas[0]) if deltas else 0
    return SeedVariation(
        seed_count=len(deltas),
        cases_per_seed=count,
        paired_executions=sum(len(block) for block in deltas),
        mean_delta=mean(block_means) if block_means else None,
        sample_stddev=stdev(block_means) if len(block_means) > 1 else None,
        minimum_delta=min(block_means) if block_means else None,
        maximum_delta=max(block_means) if block_means else None,
    )


def _deltas(before: list[CaseEvaluation], after: list[CaseEvaluation]) -> list[float]:
    return [
        float(right.outcome == "pass") - float(left.outcome == "pass")
        for left, right in zip(before, after, strict=True)
    ]


def assess_study_arms(spec: StudySpec, arms: list[tuple[str, RunRecord]]) -> StudyAssessment:
    """No outputs or case identifiers escape; seed repeats are not independent cases."""
    summaries: list[ArmSummary] = []
    comparisons: list[PairedSummary] = []
    issues: list[str] = []
    names, seeds = study_arm_names(spec), study_seeds(spec)
    expected = [study_arm_name(spec, name, seed) for seed in seeds for name in names]
    if [name for name, _run in arms] != expected:
        issues.append("the preregistered arm sequence did not complete")
    reports: dict[str, EvaluationReport] = {}
    for arm, run in arms:
        report = evaluate_run(run)
        reports[arm] = report
        summaries.append(
            ArmSummary(
                arm=arm,
                status=run.status,
                complete=report.complete,
                total=report.summary.total,
                passed=report.summary.passed,
                target_total=report.targets.total,
                target_passed=report.targets.passed,
                regression_total=report.regressions.total,
                regression_passed=report.regressions.passed,
                seed=run.spec.generation.seed if spec.schema_version == "0.2" else None,
                groups=[
                    StudyGroupSummary(
                        group=group, total=value.total, passed=value.passed, unscored=value.unscored
                    )
                    for group, value in report.groups.items()
                ],
            )
        )
        if not report.complete:
            issues.append("one or more arms were incomplete or unscored")
    verdict = "inconclusive"
    if not issues:
        runs = dict(arms)
        first = runs[study_arm_name(spec, "baseline", seeds[0])]
        base_cases = first.spec.cases
        identities: dict[str, ModelIdentity | None] = {}
        candidate_program = runs[study_arm_name(spec, "candidate", seeds[0])].spec.program
        comparators = {"comparator:" + c.name: c for c in spec.comparators}
        controls = {"control:" + c.name: c for c in spec.controls}
        for seed in seeds:
            for name in names:
                run = runs[study_arm_name(spec, name, seed)]
                comparator = comparators.get(name)
                model = comparator.model if comparator and comparator.model else spec.model
                try:
                    cases = comparator.transform_cases(base_cases) if comparator else base_cases
                except ValueError:
                    issues.append("registered prompt transformation is invalid for arm cases")
                    continue
                if run.spec.model != model or run.spec.cases != cases:
                    issues.append("arm model or ordered cases differ from the registered design")
                if run.spec.generation != spec.generation.model_copy(update={"seed": seed}):
                    issues.append("arm generation differs from its registered seed block")
                model_key = digest(model)
                if model_key in identities and identities[model_key] != run.model_identity:
                    issues.append("executed model identity changed across registered arms or seeds")
                identities[model_key] = run.model_identity
                # Compare edit content while permitting engine-added base fingerprint pins.
                if name == "candidate":
                    expected_program = candidate_program
                elif name in controls:
                    expected_program = controls[name].program
                else:
                    expected_program = None
                if expected_program is None:
                    if run.spec.program.kind != "identity":
                        issues.append("baseline or comparator applied an unregistered edit")
                elif run.spec.program.model_dump(
                    exclude={"expected_base"}
                ) != expected_program.model_dump(exclude={"expected_base"}):
                    issues.append("registered edit changed between arms or seeds")
            baseline = runs[study_arm_name(spec, "baseline", seed)]
            no_op = runs[study_arm_name(spec, "no_op", seed)]
            if [s.output for s in baseline.samples] != [s.output for s in no_op.samples]:
                issues.append("baseline and repeated no-op disagreed")
        if not issues:
            for reference in names[:-1]:
                target_deltas: list[list[float]] = []
                regression_deltas: list[list[float]] = []
                groups: dict[str, list[list[float]]] = {
                    group: []
                    for group in sorted(
                        {case.group for case in base_cases if case.group is not None}
                    )
                }
                for seed in seeds:
                    before = reports[study_arm_name(spec, reference, seed)].cases
                    after = reports[study_arm_name(spec, "candidate", seed)].cases
                    changes = _deltas(before, after)
                    target_deltas.append(
                        [
                            delta
                            for case, delta in zip(before, changes, strict=True)
                            if case.role == "target"
                        ]
                    )
                    regression_deltas.append(
                        [
                            delta
                            for case, delta in zip(before, changes, strict=True)
                            if case.role == "regression"
                        ]
                    )
                    for group, blocks in groups.items():
                        blocks.append(
                            [
                                delta
                                for case, delta in zip(before, changes, strict=True)
                                if case.group == group
                            ]
                        )
                target_flat = [delta for block in target_deltas for delta in block]
                regression_flat = [delta for block in regression_deltas for delta in block]
                comparisons.append(
                    PairedSummary(
                        reference=reference,
                        comparable=True,
                        improved_targets=sum(delta > 0 for delta in target_flat),
                        regressed_targets=sum(delta < 0 for delta in target_flat),
                        improved_regressions=sum(delta > 0 for delta in regression_flat),
                        regressed_regressions=sum(delta < 0 for delta in regression_flat),
                        target_variation=_variation(target_deltas)
                        if spec.schema_version == "0.2"
                        else None,
                        regression_variation=_variation(regression_deltas)
                        if spec.schema_version == "0.2"
                        else None,
                        groups=[
                            PairedGroupSummary(
                                group=group,
                                improved=sum(delta > 0 for block in blocks for delta in block),
                                regressed=sum(delta < 0 for block in blocks for delta in block),
                                unchanged=sum(delta == 0 for block in blocks for delta in block),
                                variation=_variation(blocks),
                            )
                            for group, blocks in groups.items()
                        ],
                    )
                )
            base = comparisons[0]
            if base.regressed_targets or base.regressed_regressions:
                verdict = "regressed"
            elif not base.improved_targets and not base.improved_regressions:
                verdict = "no_change"
            elif all(
                pair.improved_targets > pair.regressed_targets and pair.regressed_regressions == 0
                for pair in comparisons
            ):
                verdict = "observed_improvement"
            else:
                issues.append("candidate did not improve targets beyond every registered control")
    return StudyAssessment.model_validate(
        {
            "scorer_version": SCORER_VERSION,
            "verdict": verdict,
            "recommendation": "candidate_supported"
            if verdict == "observed_improvement"
            else "retain_original",
            "scope": "engineering_fixture" if spec.model.backend == "tiny" else "real_model",
            "arms": summaries,
            "comparisons": comparisons,
            "issues": list(dict.fromkeys(issues)),
        }
    )
