from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from nrly.contracts import (
    BackendResult,
    BudgetSpec,
    CaptureSpec,
    CaseSpec,
    ExperimentSpec,
    GenerationSpec,
    ModelIdentity,
    ModelSpec,
    NrlyError,
    ProgramSpec,
)
from nrly.evaluation import compare_runs, evaluate_run
from nrly.runner import Runner
from nrly.store import Store, read_bundle
from nrly.worker import execute


def specification(**updates: object) -> ExperimentSpec:
    data: dict[str, object] = {
        "name": "runner engineering fixture",
        "cases": [
            {"id": "sum", "input": "[2,4]", "expected": "6", "scorer": "numeric"},
            {
                "id": "zero",
                "input": "[0,0]",
                "expected": "0",
                "scorer": "numeric",
                "role": "regression",
            },
        ],
        "budget": {"wall_seconds": 60, "max_capture_bytes": 0},
    }
    data.update(updates)
    return ExperimentSpec.model_validate(data)


def test_actual_fixture_child_process_compare_export_and_replay(tmp_path: Path) -> None:
    with Runner(tmp_path / "workspace") as runner:
        original = specification()
        baseline = runner.wait(runner.submit(original, request_id="baseline").id, timeout=60)
        duplicate = runner.submit(original, request_id="baseline")
        assert duplicate == baseline
        assert baseline.status == "succeeded", baseline.error
        assert [sample.output for sample in baseline.samples] == ["3", "0"]
        assert baseline.capture_bytes == 0
        assert baseline.model_identity is not None
        repaired_spec = original.model_copy(
            update={
                "program": ProgramSpec(kind="weight_scale", target="output.weight", factor=2),
                "budget": original.budget.model_copy(update={"max_capture_bytes": 100_000}),
            }
        )
        repaired = runner.wait(runner.submit(repaired_spec).id, timeout=60)
        assert repaired.status == "succeeded", repaired.error
        assert [sample.output for sample in repaired.samples] == ["6", "0"]
        comparison = compare_runs(baseline, repaired)
        assert comparison.comparable
        assert comparison.verdict == "observed_improvement"
        assert comparison.accuracy_delta == 0.5
        assert comparison.candidate.scope == "engineering_fixture"
        assert comparison.candidate.regressions.passed == 1
        bundle = runner.store.export(repaired.id, tmp_path / "bundle")
        exported = read_bundle(bundle)
        replay = runner.wait(runner.submit(exported.spec, parent_run=exported.id).id, timeout=60)
        assert replay.status == "succeeded", replay.error
        assert replay.parent_run == repaired.id
        assert replay.model_identity == repaired.model_identity
        assert [sample.output for sample in replay.samples] == ["6", "0"]


def test_real_inspection_reads_weights_without_task_outputs(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        name="read base weights",
        purpose="inspection",
        weight_reads=[CaptureSpec(target="output.weight", max_elements=2)],
        budget=BudgetSpec(wall_seconds=60),
    )
    with Runner(tmp_path) as runner:
        result = runner.wait(runner.submit(spec).id, timeout=60)
    assert result.status == "succeeded", result.error
    assert result.samples == []
    assert result.output_tokens == 0
    assert result.inspection is not None
    assert result.inspection.weights[0].values == [0.5, 0.5]
    assert result.inspection.weights[0].shape == [1, 2]
    assert any(tensor.name == "output.weight" for tensor in result.inspection.tensors)
    report = evaluate_run(result)
    assert not report.complete
    assert report.summary.accuracy is None
    assert compare_runs(result, result).verdict == "inconclusive"


def test_capture_budget_failure_is_never_scored_as_success(tmp_path: Path) -> None:
    spec = specification(captures=[{"target": "hidden"}])
    with Runner(tmp_path) as runner:
        result = runner.wait(runner.submit(spec).id, timeout=60)
    assert result.status == "failed"
    assert result.error is not None and result.error.code == "budget_exceeded"
    assert not evaluate_run(result).complete
    assert evaluate_run(result).summary.passed == 0


def test_inspection_replay_refuses_a_different_base(tmp_path: Path) -> None:
    spec = ExperimentSpec(
        name="incorrect inspection base",
        purpose="inspection",
        program=ProgramSpec(expected_base="incorrect-base"),
    )
    with Runner(tmp_path) as runner:
        result = runner.wait(runner.submit(spec).id, timeout=60)
    assert result.status == "failed"
    assert result.error is not None and result.error.code == "base_mismatch"
    assert result.inspection is None


def test_wall_budget_terminates_compute_child(tmp_path: Path) -> None:
    spec = specification(budget={"wall_seconds": 0.001})
    with Runner(tmp_path) as runner:
        result = runner.wait(runner.submit(spec).id, timeout=10)
    assert result.status == "failed"
    assert result.error is not None and result.error.code == "budget_exceeded"
    assert result.elapsed_ms < 10000
    assert not evaluate_run(result).complete


def test_running_cancellation_is_terminal_and_idempotent(tmp_path: Path) -> None:
    with Runner(tmp_path) as runner:
        run = runner.submit(specification())
        deadline = time.monotonic() + 5
        while runner.store.get(run.id).status == "queued" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert runner.store.get(run.id).status == "running"
        runner.cancel(run.id)
        cancelled = runner.wait(run.id, timeout=10)
        assert cancelled.status == "cancelled"
        assert runner.cancel(run.id) == cancelled
        assert evaluate_run(cancelled).summary.passed == 0


def test_queued_cancellation_wait_returns_record(tmp_path: Path) -> None:
    release = threading.Event()
    with Runner(tmp_path) as runner:
        blocker = runner.pool.submit(release.wait, 10)
        try:
            run = runner.submit(specification())
            assert runner.cancel(run.id).status == "cancelled"
            assert runner.wait(run.id, timeout=1).status == "cancelled"
        finally:
            release.set()
            blocker.result(timeout=2)


def test_exclusive_lease_releases_after_shutdown(tmp_path: Path) -> None:
    with Runner(tmp_path):
        with pytest.raises(NrlyError) as error:
            Runner(tmp_path)
        assert error.value.code == "workspace_busy"
    with Runner(tmp_path):
        pass


def test_restart_marks_unfinished_runs_interrupted(tmp_path: Path) -> None:
    store = Store(tmp_path)
    orphan, _ = store.create(specification())
    with Runner(tmp_path) as runner:
        recovered = runner.store.get(orphan.id)
        assert recovered.status == "interrupted"
        assert recovered.error is not None and recovered.error.code == "runner_restarted"
        assert evaluate_run(recovered).summary.passed == 0


def test_closed_runner_refuses_submission(tmp_path: Path) -> None:
    runner = Runner(tmp_path)
    runner.close()
    runner.close()
    with pytest.raises(NrlyError) as error:
        runner.submit(specification())
    assert error.value.code == "runner_closed"


def test_parent_pipe_closure_terminates_worker_watchdog() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "from nrly.worker import parent_watchdog; parent_watchdog()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        assert process.stdin is not None
        process.stdin.close()
        assert process.wait(timeout=5) == 70
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if process.stderr is not None:
            process.stderr.close()


def test_token_budget_refuses_to_change_generation_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A unit-test backend exercises accounting; this is not model acceptance evidence."""
    from nrly import backends

    calls: list[int] = []
    spec = ExperimentSpec(
        name="accounting unit test",
        model=ModelSpec(backend="huggingface", model_id="unit-test-only", revision="a" * 40),
        generation=GenerationSpec(max_new_tokens=10),
        budget=BudgetSpec(max_output_tokens=15),
        cases=[CaseSpec(id="first", input="unit"), CaseSpec(id="second", input="unit")],
    )

    class AccountingBackend:
        identity = ModelIdentity(
            backend="huggingface",
            model_id="unit-test-only",
            revision="a" * 40,
            fingerprint="unit-test-only",
        )

        def __enter__(self) -> AccountingBackend:
            return self

        def __exit__(self, *_arguments: object) -> None:
            pass

        def run_case(
            self,
            _case: CaseSpec,
            _program: ProgramSpec,
            _captures: list[CaptureSpec],
            generation: GenerationSpec,
        ) -> BackendResult:
            calls.append(generation.max_new_tokens)
            return BackendResult(output="unit", output_tokens=8, elapsed_ms=0)

    monkeypatch.setattr(backends, "load_backend", lambda _spec: AccountingBackend())
    spec_file, result_file = tmp_path / "spec.json", tmp_path / "progress.json"
    spec_file.write_text(spec.model_dump_json(), encoding="utf-8")
    assert execute(spec_file, result_file) == 1
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert calls == [10]
    assert result["output_tokens"] == 8
    assert result["error"]["code"] == "budget_exceeded"
    assert len(result["samples"]) == 1
