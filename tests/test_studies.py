from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nrly.api import Application
from nrly.contracts import (
    CaseSpec,
    ExperimentSpec,
    ModelIdentity,
    NrlyError,
    ProgramSpec,
    RunRecord,
    SampleResult,
    canonical_json,
    digest,
)
from nrly.runner import Runner
from nrly.studies import StudyEngine, StudyReader, assess_study_arms
from nrly.study_contracts import CandidateSpec, StudyBudget, StudySpec


def design(**updates: object) -> StudySpec:
    data: dict[str, object] = {
        "name": "tiny controlled engineering study",
        "development_cases": [
            {"id": "development-sum", "input": "[2,4]", "expected": "6", "scorer": "numeric"},
            {
                "id": "development-zero",
                "input": "[0,0]",
                "expected": "0",
                "scorer": "numeric",
                "role": "regression",
            },
        ],
        "mutable_targets": ["output.weight"],
        "run_budget": {"wall_seconds": 60, "max_cases": 2, "max_output_tokens": 64},
    }
    data.update(updates)
    return StudySpec.model_validate(data)


def sealed_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            id="operator-secret-target",
            input="[123,321]",
            expected="444",
            scorer="numeric",
            split="test",
        ),
        CaseSpec(
            id="operator-secret-regression",
            input="[0.0,0.0]",
            expected="0",
            scorer="numeric",
            split="test",
            role="regression",
        ),
    ]


def repair() -> CandidateSpec:
    return CandidateSpec(
        name="repair",
        program=ProgramSpec(
            kind="weight_scale",
            target="output.weight",
            factor=2,
        ),
    )


def test_actual_fixture_study_seals_final_and_keeps_development_evidence(tmp_path: Path) -> None:
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        try:
            spec = design(
                controls=[
                    CandidateSpec(
                        name="unit-scale-control",
                        program=ProgramSpec(
                            kind="weight_scale",
                            target="output.weight",
                            factor=1,
                        ),
                    ).model_dump()
                ]
            )
            study = engine.create(spec, sealed_cases(), request_id="operator-study")
            assert engine.create(spec, sealed_cases(), request_id="operator-study") == study
            original = repair()
            attempt = engine.submit_candidate(study.id, original, request_id="candidate-1")
            done = engine.wait(study.id, attempt.id, timeout=120)
            assert done.status == "succeeded", done
            assert done.assessment is not None
            assert done.assessment.verdict == "observed_improvement"
            assert engine.submit_candidate(study.id, original, request_id="candidate-1") == done
            assert engine.development_cases(study.id)["cases"][0]["expected"] == "6"
            samples = engine.development_samples(study.id, attempt.id, "candidate")
            assert samples["samples"][0]["output"] == "6"

            reader = StudyReader(tmp_path)
            frozen = engine.finalize(study.id, attempt.id)
            assert frozen.nominee_digest == digest(original)
            final = engine.wait_final(study.id, timeout=120)
            assert final.status == "completed", final
            assert final.final_result is not None
            assert final.final_result.verdict == "observed_improvement"
            assert final.final_result.scope == "engineering_fixture"
            assert final.final_result.recommendation == "candidate_supported"
            assert reader.get(study.id) == final
            assert engine.finalize(study.id, attempt.id) == final
            assert final.usage.attempts == 1 and final.usage.runs == 8
            assert runner.store.list() == []
            assert len(engine.private_runner.store.list()) == 8

            app = Application.__new__(Application)
            app.runner, app.store = runner, runner.store
            for private in engine.private_runner.store.list():
                with pytest.raises(NrlyError, match="no experiment run"):
                    app.get_run(private.id)
            assert app.list_runs() == {"runs": []}
            public = json.dumps(
                {
                    "study": final.model_dump(mode="json"),
                    "attempts": [
                        item.model_dump(mode="json") for item in reader.attempts(study.id)
                    ],
                    "events": [item.model_dump(mode="json") for item in reader.events(study.id)],
                }
            )
            for case in sealed_cases():
                assert case.id not in public and case.input not in public
            for private in engine.private_runner.store.list():
                assert private.id not in public
            with pytest.raises(NrlyError, match="no attempt"):
                reader.development_samples(study.id, "f" * 32, "candidate")
            with pytest.raises(NrlyError, match="no recorded run"):
                reader.development_samples(study.id, attempt.id, "final:candidate")
            with pytest.raises(NrlyError, match="no longer accepts"):
                engine.submit_candidate(study.id, original)
            events = reader.events(study.id)
            for previous, current in zip(events, events[1:], strict=False):
                assert current.previous_digest == previous.event_digest
            for event in events:
                assert digest(event.model_dump(exclude={"event_digest"})) == event.event_digest
        finally:
            engine.close()


def fake_run(spec: StudySpec, cases: list[CaseSpec], good: bool, identifier: str) -> RunRecord:
    experiment = ExperimentSpec(
        name="synthetic scoring unit test",
        model=spec.model,
        cases=cases,
        generation=spec.generation,
    )
    return RunRecord(
        id=identifier,
        status="succeeded",
        spec=experiment,
        spec_digest=digest(experiment),
        created_at="unit",
        model_identity=ModelIdentity(
            backend="tiny",
            model_id="nrly/tiny-sum-v1",
            revision="1",
            fingerprint="unit",
        ),
        samples=[
            SampleResult(
                case_id=case.id,
                status="completed",
                provenance="torch_fixture",
                output=case.expected if good or case.role == "regression" else "incorrect",
            )
            for case in cases
        ],
    )


def fake_arms(spec: StudySpec, good: bool = True) -> list[tuple[str, RunRecord]]:
    cases = spec.development_cases
    return [
        ("baseline", fake_run(spec, cases, False, "baseline")),
        ("no_op", fake_run(spec, cases, False, "no_op")),
        *(
            ("control:" + control.name, fake_run(spec, cases, False, control.name))
            for control in spec.controls
        ),
        ("candidate", fake_run(spec, cases, good, "candidate")),
    ]


def test_scoring_requires_controls_and_regression_preservation() -> None:
    spec = design(controls=[CandidateSpec(name="placebo", program=ProgramSpec()).model_dump()])
    arms = fake_arms(spec)
    assert assess_study_arms(spec, arms).verdict == "observed_improvement"
    assert assess_study_arms(spec, fake_arms(spec, False)).verdict == "no_change"
    assert assess_study_arms(spec, arms[:-1]).verdict == "inconclusive"
    arms[2] = ("control:placebo", fake_run(spec, spec.development_cases, True, "control"))
    report = assess_study_arms(spec, arms)
    assert report.verdict == "inconclusive" and report.recommendation == "retain_original"
    arms = fake_arms(spec)
    broken = arms[-1][1].samples[1].model_copy(update={"output": "regression"})
    arms[-1] = (
        "candidate",
        arms[-1][1].model_copy(update={"samples": [arms[-1][1].samples[0], broken]}),
    )
    assert assess_study_arms(spec, arms).verdict == "regressed"
    arms = fake_arms(spec)
    arms[1] = ("no_op", fake_run(spec, spec.development_cases, True, "noop"))
    assert assess_study_arms(spec, arms).verdict == "inconclusive"
    arms = fake_arms(spec)
    arms[-1] = ("candidate", arms[-1][1].model_copy(update={"status": "cancelled"}))
    report = assess_study_arms(spec, arms)
    assert report.verdict == "inconclusive"
    assert report.arms[-1].passed == 0


def test_racing_retries_reserve_once_and_cancelled_budget_never_refunds(tmp_path: Path) -> None:
    spec = design(budget=StudyBudget(max_attempts=1, max_runs=6).model_dump())
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        release = threading.Event()
        blocker = engine.pool.submit(release.wait, 30)
        try:
            study = engine.create(spec, sealed_cases())
            with ThreadPoolExecutor(max_workers=8) as pool:
                records = list(
                    pool.map(
                        lambda _index: engine.submit_candidate(study.id, repair(), "same-request"),
                        range(16),
                    )
                )
            assert len({item.id for item in records}) == 1
            assert engine.get(study.id).usage.attempts == 1
            with pytest.raises(NrlyError, match="budget exhausted"):
                engine.submit_candidate(study.id, repair(), "different-request")
            conflict = CandidateSpec(name="different", program=ProgramSpec())
            with pytest.raises(NrlyError, match="another candidate"):
                engine.submit_candidate(study.id, conflict, "same-request")
            engine.cancel(study.id, records[0].id)
            release.set()
            blocker.result(5)
            cancelled = engine.wait(study.id, records[0].id, 5)
            assert cancelled.status == "cancelled"
            assert engine.get(study.id).usage.runs == 6
            with pytest.raises(NrlyError, match="budget exhausted"):
                engine.submit_candidate(study.id, repair(), "new-budget-bypass")
        finally:
            release.set()
            engine.close()


def test_candidate_payload_is_detached_and_ledger_terminal_rows_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        monkeypatch.setattr(engine, "_arms", lambda study, *_args: fake_arms(study.spec))
        try:
            study = engine.create(design(), sealed_cases())
            candidate = CandidateSpec(
                name="replace",
                program=ProgramSpec(
                    kind="weight_replace",
                    target="output.weight",
                    shape=[1, 2],
                    values=[1, 1],
                ),
            )
            registered = engine.submit_candidate(study.id, candidate)
            candidate.program.values[0] = 999
            done = engine.wait(study.id, registered.id, 5)
            assert done.candidate.program.values == [1, 1]
            with engine._write() as connection:
                with pytest.raises(NrlyError, match="cannot be rewritten"):
                    engine._save_attempt(connection, done.model_copy(update={"status": "failed"}))
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    connection.execute("DELETE FROM events")
            changed = done.candidate.model_copy(update={"name": "tamper"})
            with engine._write() as connection:
                with pytest.raises(NrlyError, match="identity cannot change"):
                    engine._save_attempt(connection, done.model_copy(update={"candidate": changed}))
        finally:
            engine.close()


def test_restart_retains_attempt_and_final_freeze_without_reexecution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        monkeypatch.setattr(engine, "_arms", lambda study, *_args: fake_arms(study.spec))
        study = engine.create(design(), sealed_cases())
        done = engine.wait(study.id, engine.submit_candidate(study.id, repair()).id, 5)
        engine.close()
        # Simulate process death after durable nomination but before dispatch.
        with sqlite3.connect(engine.database) as connection:
            frozen = engine.get(study.id).model_copy(
                update={
                    "status": "finalizing",
                    "nominee_attempt_id": done.id,
                    "nominee_digest": done.candidate_digest,
                }
            )
            connection.execute(
                "UPDATE studies SET payload=? WHERE id=?", (canonical_json(frozen), study.id)
            )
        restarted = StudyEngine(runner)
        try:
            recovered = restarted.get(study.id)
            assert recovered.status == "interrupted"
            assert recovered.final_result is not None
            assert recovered.final_result.verdict == "inconclusive"
            assert restarted.finalize(study.id, done.id) == recovered
            assert restarted.private_runner.store.list() == []
            assert restarted.get_attempt(study.id, done.id) == done
        finally:
            restarted.close()


def test_design_rejects_holdout_overlap_and_outside_mutable_targets(tmp_path: Path) -> None:
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        try:
            overlap = sealed_cases()
            overlap[0] = overlap[0].model_copy(update={"input": "[2,4]"})
            with pytest.raises(NrlyError, match="must be disjoint"):
                engine.create(design(), overlap)
            study = engine.create(design(), sealed_cases())
            wrong = CandidateSpec(
                name="outside",
                program=ProgramSpec(
                    kind="weight_scale",
                    target="hidden.weight",
                    factor=2,
                ),
            )
            with pytest.raises(NrlyError, match="preregistered mutable"):
                engine.submit_candidate(study.id, wrong)
            assert engine.get(study.id).usage.attempts == 0
            with pytest.raises(NrlyError, match="budget exhausted"):
                engine.create(design(budget=StudyBudget(max_runs=5).model_dump()), sealed_cases())
            with pytest.raises(ValueError, match="non-test split"):
                design(development_cases=sealed_cases())
        finally:
            engine.close()


def test_final_cancellation_keeps_nominee_frozen_and_cannot_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        monkeypatch.setattr(engine, "_arms", lambda study, *_args: fake_arms(study.spec))
        release = threading.Event()
        try:
            study = engine.create(design(), sealed_cases())
            first = engine.wait(study.id, engine.submit_candidate(study.id, repair()).id, 5)
            second = engine.wait(study.id, engine.submit_candidate(study.id, repair()).id, 5)
            blocker = engine.pool.submit(release.wait, 30)
            engine.finalize(study.id, first.id)
            with pytest.raises(NrlyError, match="different final nominee"):
                engine.finalize(study.id, second.id)
            engine.cancel_final(study.id)
            release.set()
            blocker.result(5)
            cancelled = engine.wait_final(study.id, 5)
            assert cancelled.status == "completed"
            assert cancelled.final_error_code == "study_cancelled"
            assert cancelled.final_result is not None
            assert cancelled.final_result.recommendation == "retain_original"
            assert engine.finalize(study.id, first.id) == cancelled
            assert engine.cancel_final(study.id) == cancelled
            assert engine.get(study.id).usage.attempts == 2
        finally:
            release.set()
            engine.close()


def test_restart_interrupts_admitted_attempt_without_releasing_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        monkeypatch.setattr(engine, "_arms", lambda study, *_args: fake_arms(study.spec))
        study = engine.create(design(budget={"max_attempts": 1}), sealed_cases())
        done = engine.wait(study.id, engine.submit_candidate(study.id, repair()).id, 5)
        engine.close()
        # Model an abrupt stop after admission. No inference is silently retried.
        with sqlite3.connect(engine.database) as connection:
            orphan = done.model_copy(update={"status": "running", "assessment": None})
            connection.execute(
                "UPDATE attempts SET payload=? WHERE id=?", (canonical_json(orphan), done.id)
            )
        restarted = StudyEngine(runner)
        try:
            recovered = restarted.get_attempt(study.id, done.id)
            assert recovered.status == "interrupted"
            assert recovered.error_code == "study_restarted"
            assert restarted.get(study.id).usage.attempts == 1
            with pytest.raises(NrlyError, match="budget exhausted"):
                restarted.submit_candidate(study.id, repair())
        finally:
            restarted.close()


@pytest.mark.parametrize(
    "dimension,ceiling",
    [
        ("max_runs", 6),
        ("max_case_executions", 12),
        ("max_output_tokens", 384),
        ("max_capture_bytes", 6 * 262144),
        ("max_wall_seconds", 360),
    ],
)
def test_each_study_budget_dimension_includes_reserved_final(
    dimension: str,
    ceiling: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        monkeypatch.setattr(engine, "_arms", lambda study, *_args: fake_arms(study.spec))
        try:
            study = engine.create(design(budget={dimension: ceiling}), sealed_cases())
            engine.wait(study.id, engine.submit_candidate(study.id, repair()).id, 5)
            with pytest.raises(NrlyError, match="budget exhausted"):
                engine.submit_candidate(study.id, repair())
        finally:
            engine.close()


def test_development_output_pagination_reconstructs_unicode_without_private_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)
        monkeypatch.setattr(engine, "_arms", lambda study, *_args: fake_arms(study.spec))
        try:
            study = engine.create(design(), sealed_cases())
            attempt = engine.wait(study.id, engine.submit_candidate(study.id, repair()).id, 5)
            output = ("🙂 café 🧬 漢字 e\u0301\n" * 350) + "END"
            recorded = fake_run(study.spec, study.spec.development_cases, True, "fixture")
            private, _created = engine.private_runner.store.create(recorded.spec)
            sample = recorded.samples[0].model_copy(update={"output": output})
            engine.private_runner.store.save(
                private.model_copy(
                    update={
                        "status": "succeeded",
                        "model_identity": recorded.model_identity,
                        "samples": [sample],
                    }
                )
            )
            with engine._write() as connection:
                connection.execute(
                    "INSERT INTO development_runs VALUES (?,?,?,?)",
                    (study.id, attempt.id, "candidate", private.id),
                )
            reader = StudyReader(tmp_path)
            chunks: list[str] = []
            for start in range(0, len(output), 37):
                page = reader.development_samples(
                    study.id,
                    attempt.id,
                    "candidate",
                    limit=1,
                    output_limit=37,
                    output_offset=start,
                )
                item = page["samples"][0]
                assert item["output_offset"] == start
                assert item["output_characters"] == len(output)
                assert item["more_output"] == (start + 37 < len(output))
                assert len(item["output"]) <= 37
                assert private.id not in json.dumps(page)
                chunks.append(item["output"])
            assert "".join(chunks) == output
            end = reader.development_samples(
                study.id,
                attempt.id,
                "candidate",
                output_offset=len(output),
            )["samples"][0]
            assert end["output"] == "" and end["more_output"] is False
            with pytest.raises(NrlyError, match="invalid development sample pagination"):
                reader.development_samples(study.id, attempt.id, "candidate", output_offset=-1)
            with pytest.raises(NrlyError, match="no attempt"):
                reader.development_samples(study.id, private.id, "candidate", output_offset=1)
        finally:
            engine.close()


def test_comparable_scientific_inconclusiveness_completes_attempt_and_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = design(
        schema_version="0.2", comparators=[{"name": "prompt", "kind": "prompt", "suffix": " "}]
    )
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)

        def completed_arms(study, _candidate, cases, _key, _event):
            return [
                ("baseline", fake_run(study.spec, cases, False, "baseline")),
                ("no_op", fake_run(study.spec, cases, False, "no-op")),
                (
                    "comparator:prompt",
                    fake_run(
                        study.spec, study.spec.comparators[0].transform_cases(cases), True, "prompt"
                    ),
                ),
                ("candidate", fake_run(study.spec, cases, True, "candidate")),
            ]

        monkeypatch.setattr(engine, "_arms", completed_arms)
        try:
            study = engine.create(spec, sealed_cases())
            attempt = engine.wait(study.id, engine.submit_candidate(study.id, repair()).id, 5)
            assert attempt.status == "succeeded" and attempt.error_code is None
            assert attempt.assessment is not None
            assert attempt.assessment.verdict == "inconclusive"
            assert attempt.assessment.recommendation == "retain_original"
            assert attempt.assessment.comparisons[0].improved_targets == 1
            assert attempt.assessment.comparisons[-1].improved_targets == 0
            assert all(arm.complete for arm in attempt.assessment.arms)
            assert all(pair.comparable for pair in attempt.assessment.comparisons)
            engine.finalize(study.id, attempt.id)
            final = engine.wait_final(study.id, 5)
            assert final.status == "completed" and final.final_error_code is None
            assert final.final_result is not None
            assert final.final_result.verdict == "inconclusive"
            assert final.final_result.recommendation == "retain_original"
            assert final.nominee_attempt_id == attempt.id
            assert final.usage.runs == 8 and final.usage.attempts == 1
        finally:
            engine.close()


@pytest.mark.parametrize("failure", ["missing", "incomplete", "identity", "no_op"])
def test_incomplete_or_incomparable_evidence_still_fails_attempt_and_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    reject_discovery = True
    with Runner(tmp_path) as runner:
        engine = StudyEngine(runner)

        def incomplete_arms(study, _candidate, cases, key, _event):
            arms = [
                ("baseline", fake_run(study.spec, cases, False, "baseline")),
                ("no_op", fake_run(study.spec, cases, False, "no-op")),
                ("candidate", fake_run(study.spec, cases, True, "candidate")),
            ]
            if not reject_discovery and not key.startswith("final:"):
                return arms
            if failure == "missing":
                return arms[:-1]
            if failure == "no_op":
                arms[1] = ("no_op", fake_run(study.spec, cases, True, "no-op"))
            else:
                candidate = arms[-1][1]
                if failure == "incomplete":
                    candidate = candidate.model_copy(update={"status": "failed"})
                else:
                    candidate = candidate.model_copy(
                        update={
                            "model_identity": candidate.model_identity.model_copy(
                                update={"fingerprint": "different"}
                            )
                        }
                    )
                arms[-1] = ("candidate", candidate)
            return arms

        monkeypatch.setattr(engine, "_arms", incomplete_arms)
        try:
            study = engine.create(design(), sealed_cases())
            failed = engine.wait(study.id, engine.submit_candidate(study.id, repair()).id, 5)
            assert failed.status == "failed" and failed.error_code == "study_inconclusive"
            with pytest.raises(NrlyError, match="completed discovery"):
                engine.finalize(study.id, failed.id)
            reject_discovery = False
            complete = engine.wait(study.id, engine.submit_candidate(study.id, repair()).id, 5)
            assert complete.status == "succeeded"
            engine.finalize(study.id, complete.id)
            final = engine.wait_final(study.id, 5)
            assert final.status == "completed" and final.final_error_code == "study_inconclusive"
            assert final.final_result is not None and final.final_result.verdict == "inconclusive"
            assert final.final_result.recommendation == "retain_original"
            assert final.usage.attempts == 2 and final.usage.runs == 9
        finally:
            engine.close()
