from __future__ import annotations

import json
from pathlib import Path

import pytest

from nrly.contracts import (
    CaseSpec,
    ExperimentSpec,
    ModelIdentity,
    ModelSpec,
    NrlyError,
    ProgramSpec,
    RunRecord,
    SampleResult,
    digest,
)
from nrly.evaluation import compare_runs
from nrly.runner import Runner
from nrly.studies import StudyEngine, assess_study_arms
from nrly.study_contracts import (
    CandidateSpec,
    ComparatorSpec,
    StudySpec,
    StudyUsage,
    study_arm_name,
    study_arm_names,
    study_seeds,
)


def design(**updates: object) -> StudySpec:
    data: dict[str, object] = {
        "schema_version": "0.2",
        "name": "paired fixture design",
        "seeds": [11, 29],
        "development_cases": [
            {
                "id": "target",
                "input": "[2,4]",
                "expected": "6",
                "scorer": "numeric",
                "group": "generalization",
            },
            {
                "id": "local",
                "input": "[0,0]",
                "expected": "0",
                "scorer": "numeric",
                "role": "regression",
                "group": "locality",
            },
        ],
        "run_budget": {"wall_seconds": 60, "max_cases": 2, "max_output_tokens": 64},
    }
    data.update(updates)
    return StudySpec.model_validate(data)


def sealed() -> list[CaseSpec]:
    return [
        CaseSpec(
            id="sealed-target",
            input="[4,6]",
            expected="10",
            scorer="numeric",
            split="test",
            group="generalization",
        ),
        CaseSpec(
            id="sealed-local",
            input="[0.0,0.0]",
            expected="0",
            scorer="numeric",
            split="test",
            role="regression",
            group="locality",
        ),
    ]


def fake_arms(
    spec: StudySpec, improved_seeds: set[int] | None = None
) -> list[tuple[str, RunRecord]]:
    result = []
    for seed in study_seeds(spec):
        for name in study_arm_names(spec):
            comparator = next((c for c in spec.comparators if name == "comparator:" + c.name), None)
            model = comparator.model if comparator and comparator.model else spec.model
            cases = (
                comparator.transform_cases(spec.development_cases)
                if comparator
                else spec.development_cases
            )
            experiment = ExperimentSpec(
                schema_version="0.3",
                name="synthetic scoring fixture",
                model=model,
                cases=cases,
                generation=spec.generation.model_copy(update={"seed": seed}),
            )
            arm = study_arm_name(spec, name, seed)
            good = name == "candidate" and (improved_seeds is None or seed in improved_seeds)
            result.append(
                (
                    arm,
                    RunRecord(
                        id=arm,
                        status="succeeded",
                        created_at="unit",
                        spec=experiment,
                        spec_digest=digest(experiment),
                        model_identity=ModelIdentity(
                            backend=model.backend,
                            model_id=model.model_id,
                            revision=model.revision,
                            fingerprint=digest(model),
                        ),
                        samples=[
                            SampleResult(
                                case_id=case.id,
                                status="completed",
                                provenance="torch_fixture"
                                if model.backend == "tiny"
                                else "huggingface",
                                output=case.expected
                                if good or case.role == "regression"
                                else "wrong",
                                truncated=False,
                            )
                            for case in cases
                        ],
                    ),
                )
            )
    return result


def replace_spec(run: RunRecord, **updates: object) -> RunRecord:
    spec = ExperimentSpec.model_validate({**run.spec.model_dump(), **updates})
    return run.model_copy(update={"spec": spec, "spec_digest": digest(spec)})


def test_seed_variation_counts_paired_cases_without_claiming_independence() -> None:
    spec = design()
    report = assess_study_arms(spec, fake_arms(spec, {11}))
    assert report.verdict == "observed_improvement"
    comparison = report.comparisons[0]
    assert comparison.improved_targets == 1
    variation = comparison.target_variation
    assert variation is not None
    assert (variation.seed_count, variation.cases_per_seed, variation.paired_executions) == (
        2,
        1,
        2,
    )
    assert variation.mean_delta == 0.5
    assert variation.sample_stddev == pytest.approx(2**-0.5)
    assert (variation.minimum_delta, variation.maximum_delta) == (0, 1)
    assert variation.interpretation == "descriptive_across_registered_seeds_not_population_ci"
    assert comparison.groups[0].group == "generalization"
    assert comparison.groups[1].variation.mean_delta == 0
    one = design(seeds=[11])
    assert (
        assess_study_arms(one, fake_arms(one)).comparisons[0].target_variation.sample_stddev is None
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": "0.1"},
        {"seeds": [11, 11]},
        {"seeds": [True]},
        {"seeds": None, "generation": {"temperature": 0.8}},
    ],
)
def test_invalid_sampling_designs_rejected(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        design(**updates)


def test_sampling_is_explicit_and_legacy_serialization_unchanged() -> None:
    spec = design(generation={"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1024})
    assert spec.generation.temperature == 0.8
    legacy = StudySpec(
        name="legacy",
        development_cases=[
            CaseSpec(id="t", input="x", expected="x"),
            CaseSpec(id="r", input="y", expected="y", role="regression"),
        ],
    )
    assert "seeds" not in legacy.model_dump() and "comparators" not in legacy.model_dump()
    assert study_arm_name(legacy, "candidate", 0) == "candidate"


@pytest.mark.parametrize(
    "dimension,multiplier",
    [
        ("runs", 1),
        ("case_executions", 2),
        ("output_tokens", 64),
        ("capture_bytes", 262144),
        ("wall_seconds", 60),
    ],
)
def test_admission_charges_all_seeds_controls_comparators_and_reserved_final(
    dimension: str,
    multiplier: int,
) -> None:
    spec = design(comparators=[{"name": "prompt", "kind": "prompt", "suffix": " "}])
    final = StudyEngine._charge(spec, StudyUsage(), 2, final=True)
    charged = StudyEngine._charge(spec, final, 2, final=False)
    assert final.final_runs_reserved == 8
    assert getattr(charged, dimension) == 16 * multiplier
    ceiling = "max_" + dimension
    tight = design(comparators=spec.comparators, budget={ceiling: 16 * multiplier - 1})
    with pytest.raises(NrlyError, match="budget exhausted"):
        StudyEngine._charge(
            tight, StudyEngine._charge(tight, StudyUsage(), 2, final=True), 2, final=False
        )


def test_comparators_are_preregistered_not_a_general_identity_bypass() -> None:
    external = ModelSpec(backend="huggingface", model_id="external/fine-tuned", revision="a" * 40)
    spec = design(
        comparators=[
            ComparatorSpec(name="prompt", kind="prompt", prefix=" ", suffix=" "),
            ComparatorSpec(name="external", kind="model", model=external),
        ]
    )
    arms = fake_arms(spec)
    report = assess_study_arms(spec, arms)
    assert report.verdict == "observed_improvement"
    assert len(report.comparisons) == 4
    assert not compare_runs(arms[3][1], arms[4][1]).comparable
    changed = replace_spec(arms[2][1], cases=spec.development_cases)
    arms[2] = (arms[2][0], changed)
    assert assess_study_arms(spec, arms).verdict == "inconclusive"


@pytest.mark.parametrize("failure", ["missing", "seed", "base_identity", "truncated", "failed"])
def test_missing_unpaired_or_incomplete_seed_blocks_cannot_support_improvement(
    failure: str,
) -> None:
    spec = design()
    arms = fake_arms(spec)
    if failure == "missing":
        arms.pop()
    else:
        name, run = arms[-1]
        if failure == "seed":
            run = replace_spec(run, generation=run.spec.generation.model_copy(update={"seed": 77}))
        elif failure == "base_identity":
            run = run.model_copy(
                update={
                    "model_identity": run.model_identity.model_copy(
                        update={"fingerprint": "changed"}
                    )
                }
            )
        elif failure == "truncated":
            run = run.model_copy(
                update={
                    "samples": [
                        sample.model_copy(update={"truncated": True}) for sample in run.samples
                    ]
                }
            )
        else:
            run = run.model_copy(update={"status": "failed"})
        arms[-1] = (name, run)
    report = assess_study_arms(spec, arms)
    assert report.verdict == "inconclusive"
    assert report.recommendation == "retain_original"


def test_external_model_identity_must_stay_fixed_across_seeds() -> None:
    spec = design(
        comparators=[
            {
                "name": "external",
                "kind": "model",
                "model": {
                    "backend": "huggingface",
                    "model_id": "external/model",
                    "revision": "b" * 40,
                },
            }
        ]
    )
    arms = fake_arms(spec)
    name, run = arms[-2]
    arms[-2] = (
        name,
        run.model_copy(
            update={
                "model_identity": run.model_identity.model_copy(
                    update={"fingerprint": "different-cache"}
                )
            }
        ),
    )
    assert assess_study_arms(spec, arms).verdict == "inconclusive"


def test_prompt_comparator_wraps_only_final_user_turn_and_preserves_case_contract() -> None:
    case = CaseSpec(
        id="chat",
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "question"}],
        expected="A",
        group="generalization",
    )
    comparator = ComparatorSpec(name="prompt", kind="prompt", prefix="Before ", suffix=" After")
    transformed = comparator.transform_cases([case])[0]
    assert transformed.messages[0] == case.messages[0]
    assert transformed.messages[1].content == "Before question After"
    assert transformed.expected == case.expected and transformed.group == case.group
    assert case.messages[1].content == "question"
    invalid = case.model_copy(update={"messages": [case.messages[0]]})
    with pytest.raises(ValueError, match="final user"):
        comparator.transform_cases([invalid])


def test_real_fixture_repeated_execution_keeps_holdout_sealed_and_budget_reserved(
    tmp_path: Path,
) -> None:
    spec = design(comparators=[{"name": "space", "kind": "prompt", "suffix": " "}])
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        try:
            study = engine.create(spec, sealed())
            assert study.usage.final_runs_reserved == 8
            candidate = CandidateSpec(
                name="repair",
                program=ProgramSpec(kind="weight_scale", target="output.weight", factor=2),
            )
            attempt = engine.wait(study.id, engine.submit_candidate(study.id, candidate).id, 120)
            assert attempt.status == "succeeded", attempt
            assert attempt.assessment.verdict == "observed_improvement"
            for seed in spec.seeds:
                page = engine.development_samples(study.id, attempt.id, f"candidate@seed={seed}")
                assert page["samples"][0]["output"] == "6"
                prompt = engine.development_samples(
                    study.id, attempt.id, f"comparator:space@seed={seed}"
                )
                assert prompt["samples"][0]["output"] == "3"
            engine.finalize(study.id, attempt.id)
            final = engine.wait_final(study.id, 120)
            assert final.final_result.verdict == "observed_improvement"
            assert final.usage.runs == 16 and len(engine.private_runner.store.list()) == 16
            public = json.dumps(final.model_dump(mode="json"))
            assert "sealed-target" not in public and "[4,6]" not in public
            assert not runner.store.list()
        finally:
            engine.close()


def test_persistent_comparator_identity_pin_cannot_drift_between_attempts(tmp_path: Path) -> None:
    model = ModelSpec(backend="huggingface", model_id="external/model", revision="c" * 40)
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        try:
            study = engine.create(
                design(comparators=[ComparatorSpec(name="external", kind="model", model=model)]),
                sealed(),
            )
            identity = ModelIdentity(
                backend="huggingface",
                model_id=model.model_id,
                revision=model.revision,
                fingerprint="pinned",
            )
            engine._pin_identity(study.id, identity, "external")
            assert engine.get(study.id).comparator_identities["external"] == identity
            with pytest.raises(NrlyError, match="identity changed"):
                engine._pin_identity(
                    study.id, identity.model_copy(update={"fingerprint": "drift"}), "external"
                )
        finally:
            engine.close()


@pytest.mark.parametrize("chat", [False, True])
def test_holdout_overlap_checks_effective_registered_prompts(tmp_path: Path, chat: bool) -> None:
    spec = design(comparators=[{"name": "prompt", "kind": "prompt", "prefix": "prefix "}])
    cases = sealed()
    if chat:
        cases[0] = CaseSpec(
            id="secret-chat",
            messages=[{"role": "user", "content": "[2,4]"}],
            expected="6",
            split="test",
        )
    else:
        cases[0] = CaseSpec(id="secret-plain", input="prefix [2,4]", expected="6", split="test")
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        try:
            with pytest.raises(NrlyError, match="must be disjoint"):
                engine.create(spec, cases)
        finally:
            engine.close()


def test_single_model_identity_is_shared_by_duplicate_model_registrations(tmp_path: Path) -> None:
    spec = design(comparators=[ComparatorSpec(name="same-model", kind="model", model=ModelSpec())])
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        try:
            study = engine.create(spec, sealed())
            identity = ModelIdentity(
                backend="tiny", model_id="nrly/tiny-sum-v1", revision="1", fingerprint="pinned"
            )
            engine._pin_identity(study.id, identity)
            with pytest.raises(NrlyError, match="identity changed"):
                engine._pin_identity(
                    study.id, identity.model_copy(update={"fingerprint": "drift"}), "same-model"
                )
        finally:
            engine.close()


def test_literal_constraint_studies_use_registered_constraints_without_dummy_expected() -> None:
    spec = design(
        development_cases=[
            {
                "id": "target",
                "input": "Say Hello",
                "adapter": {"kind": "instruction_constraints", "required": ["Hello"]},
            },
            {
                "id": "regression",
                "input": "Say Hi",
                "role": "regression",
                "adapter": {"kind": "instruction_constraints", "required": ["Hi"]},
            },
        ]
    )
    assert all(case.expected is None for case in spec.development_cases)


def test_study_assessments_stamp_policy_without_rewriting_legacy_payloads() -> None:
    from nrly.evaluation.reports import SCORER_VERSION
    from nrly.studies import _inconclusive
    from nrly.study_contracts import StudyAssessment

    old_payload = {
        "verdict": "inconclusive",
        "recommendation": "retain_original",
        "scope": "engineering_fixture",
        "arms": [],
        "comparisons": [],
        "issues": [],
    }
    old = StudyAssessment.model_validate(old_payload)
    assert old.model_dump(mode="json") == old_payload
    spec = design()
    assert assess_study_arms(spec, fake_arms(spec)).scorer_version == SCORER_VERSION
    assert assess_study_arms(spec, []).scorer_version == SCORER_VERSION
    assert _inconclusive(spec, "attempt cancelled").scorer_version == SCORER_VERSION
