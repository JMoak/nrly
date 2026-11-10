from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from _thread import LockType
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO

from nrly.contracts import ErrorInfo, ExperimentSpec, NrlyError, RunRecord
from nrly.store import TERMINAL, Store, now


class WorkspaceLease:
    def __init__(self, root: Path) -> None:
        self.stream: BinaryIO = (root / "runner.lock").open("a+b")
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise NrlyError("workspace_busy", "another nrly runner owns this workspace") from error

    def close(self) -> None:
        if self.stream.closed:
            return
        if sys.platform == "win32":
            import msvcrt

            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


class Runner:
    def __init__(
        self,
        root: Path,
        model_cache: Path | None = None,
        execution_gate: LockType | None = None,
    ) -> None:
        self.store = Store(root)
        self.lease = WorkspaceLease(self.store.root)
        self.model_cache = (model_cache or (self.store.root / "models")).resolve()
        self.execution_gate = execution_gate if execution_gate is not None else threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nrly")
        self.futures: dict[str, Future[None]] = {}
        self.cancellations: dict[str, threading.Event] = {}
        self.guard = threading.RLock()
        self.closed = False
        try:
            for record in self.store.unfinished():
                self.store.save(
                    record.model_copy(
                        update={
                            "status": "interrupted",
                            "finished_at": now(),
                            "error": ErrorInfo(
                                code="runner_restarted",
                                message="prior runner ended before completion",
                            ),
                        }
                    )
                )
        except Exception:
            self.pool.shutdown(wait=True)
            self.lease.close()
            raise

    def submit(
        self, spec: ExperimentSpec, request_id: str | None = None, parent_run: str | None = None
    ) -> RunRecord:
        with self.guard:
            if self.closed:
                raise NrlyError("runner_closed", "runner is shutting down")
            record, created = self.store.create(spec, request_id, parent_run)
            if created:
                self.cancellations[record.id] = threading.Event()
                self.futures[record.id] = self.pool.submit(self._execute, record.id)
            return record

    def wait(self, run_id: str, timeout: float | None = None) -> RunRecord:
        future = self.futures.get(run_id)
        if future:
            try:
                future.result(timeout=timeout)
            except CancelledError:
                pass
        return self.store.get(run_id)

    def cancel(self, run_id: str) -> RunRecord:
        with self.guard:
            record = self.store.get(run_id)
            if record.status in TERMINAL:
                return record
            event = self.cancellations.get(run_id)
            if event is None:
                raise NrlyError("run_not_owned", "this runner does not own the active job")
            event.set()
            if record.status == "queued":
                future = self.futures.get(run_id)
                if future and future.cancel():
                    record = record.model_copy(update={"status": "cancelled", "finished_at": now()})
                else:
                    record = record.model_copy(update={"status": "cancelling"})
            else:
                record = record.model_copy(update={"status": "cancelling"})
            self.store.save(record)
            return record

    def close(self) -> None:
        with self.guard:
            if self.closed:
                return
            self.closed = True
            for event in self.cancellations.values():
                event.set()
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.lease.close()

    def __enter__(self) -> Runner:
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def _execute(self, run_id: str) -> None:
        event = self.cancellations[run_id]
        while not self.execution_gate.acquire(timeout=0.1):
            if event.is_set():
                with self.guard:
                    record = self.store.get(run_id)
                    if record.status not in TERMINAL:
                        self.store.save(
                            record.model_copy(
                                update={
                                    "status": "cancelled",
                                    "finished_at": now(),
                                }
                            )
                        )
                return
        try:
            self._execute_owned(run_id)
        finally:
            self.execution_gate.release()

    def _execute_owned(self, run_id: str) -> None:
        started = time.perf_counter()
        event = self.cancellations[run_id]
        process: subprocess.Popen[bytes] | None = None
        with self.guard:
            record = self.store.get(run_id)
            if event.is_set():
                if record.status not in TERMINAL:
                    self.store.save(
                        record.model_copy(update={"status": "cancelled", "finished_at": now()})
                    )
                return
            record = record.model_copy(update={"status": "running", "started_at": now()})
            self.store.save(record)
        directory = self.store.run_dir(run_id)
        result_file = directory / "progress.json"
        failure: ErrorInfo | None = None
        status = "failed"
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HOME": str(self.model_cache),
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        try:
            with (directory / "worker.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "nrly.worker",
                        "--spec",
                        str(directory / "spec.json"),
                        "--result",
                        str(result_file),
                        "--watch-parent",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=directory,
                    env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                while process.poll() is None:
                    if event.is_set():
                        status = "cancelled"
                        break
                    if time.perf_counter() - started > record.spec.budget.wall_seconds:
                        failure = ErrorInfo(
                            code="budget_exceeded", message="wall time budget exceeded"
                        )
                        break
                    if (directory / "worker.log").stat().st_size > 1024 * 1024:
                        failure = ErrorInfo(
                            code="budget_exceeded", message="worker log limit exceeded"
                        )
                        break
                    time.sleep(0.05)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                if status != "cancelled" and failure is None:
                    status = (
                        "cancelled"
                        if event.is_set()
                        else ("succeeded" if process.returncode == 0 else "failed")
                    )
            if result_file.exists():
                if result_file.stat().st_size > 16 * 1024 * 1024:
                    raise NrlyError("budget_exceeded", "worker result exceeds size limit")
                result = json.loads(result_file.read_text(encoding="utf-8"))
                updated = record.model_dump(mode="json")
                updated.update(
                    {
                        key: result[key]
                        for key in (
                            "samples",
                            "model_identity",
                            "inspection",
                            "capture_bytes",
                            "output_tokens",
                        )
                    }
                )
                record = RunRecord.model_validate(updated)
                if result.get("error") and failure is None:
                    failure = ErrorInfo.model_validate(result["error"])
            if status == "succeeded" and (
                len(record.samples) != len(record.spec.cases)
                or (record.spec.purpose == "inspection" and record.inspection is None)
                or record.model_identity is None
                or failure
            ):
                status = "failed"
                failure = failure or ErrorInfo(
                    code="incomplete_result", message="worker omitted cases"
                )
            if status == "failed" and failure is None:
                tail = (directory / "worker.log").read_text(encoding="utf-8", errors="replace")[
                    -2000:
                ]
                failure = ErrorInfo(
                    code="worker_failed", message=tail or "worker exited without a result"
                )
        except Exception as error:
            status = "failed"
            failure = (
                error.as_info()
                if isinstance(error, NrlyError)
                else ErrorInfo(code="runner_failed", message=str(error)[:2000])
            )
        finally:
            if process:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=3)
                if process.stdin:
                    process.stdin.close()
            with self.guard:
                final_data = record.model_dump(mode="json")
                final_data.update(
                    status=status,
                    finished_at=now(),
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    error=failure.model_dump() if failure else None,
                )
                self.store.save(RunRecord.model_validate(final_data))
