from __future__ import annotations

import threading
import time
from pathlib import Path

from nrly.contracts import CaseSpec, ExperimentSpec
from nrly.runner import Runner


def test_waiting_for_another_worker_is_cancellable_without_loading_a_model(tmp_path: Path) -> None:
    gate = threading.Lock()
    gate.acquire()
    try:
        with Runner(tmp_path, execution_gate=gate) as runner:
            queued = runner.submit(
                ExperimentSpec(
                    name="waiting for shared compute",
                    cases=[CaseSpec(id="one", input="[2,4]")],
                )
            )
            time.sleep(0.15)
            record = runner.store.get(queued.id)
            assert record.status == "queued"
            assert record.started_at is None
            runner.cancel(queued.id)
            cancelled = runner.wait(queued.id, timeout=2)
            assert cancelled.status == "cancelled"
            assert cancelled.model_identity is None
            assert not (runner.store.run_dir(queued.id) / "worker.log").exists()
    finally:
        gate.release()


def test_shared_gate_releases_after_a_failed_process(tmp_path: Path) -> None:
    gate = threading.Lock()
    with Runner(tmp_path, execution_gate=gate) as runner:
        spec = ExperimentSpec(
            name="invalid numerical input", cases=[CaseSpec(id="one", input="bad")]
        )
        result = runner.wait(runner.submit(spec).id, timeout=60)
        assert result.status == "failed"
        assert gate.acquire(blocking=False)
        gate.release()
