from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from nrly.contracts import ErrorInfo, ExperimentSpec, ModelInspection, NrlyError, SampleResult
from nrly.store import atomic_text


def execute(spec_file: Path, result_file: Path) -> int:
    from nrly.backends import load_backend

    spec = ExperimentSpec.model_validate_json(spec_file.read_bytes())
    samples: list[SampleResult] = []
    identity = None
    inspection: ModelInspection | None = None
    output_tokens = 0
    capture_bytes = 0
    error_info: ErrorInfo | None = None

    def publish() -> None:
        data = {
            "model_identity": identity.model_dump(mode="json") if identity else None,
            "inspection": inspection.model_dump(mode="json") if inspection else None,
            "samples": [sample.model_dump(mode="json") for sample in samples],
            "error": error_info.model_dump() if error_info else None,
            "output_tokens": output_tokens,
            "capture_bytes": capture_bytes,
        }
        serialized = json.dumps(data, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode()) > 16 * 1024 * 1024:
            raise NrlyError("budget_exceeded", "worker result exceeds 16 MiB")
        atomic_text(result_file, serialized)

    try:
        with load_backend(spec.model) as backend:
            identity = backend.identity
            if (
                spec.program.expected_base is not None
                and spec.program.expected_base != identity.fingerprint
            ):
                raise NrlyError(
                    "base_mismatch", "program expected_base does not match the loaded model"
                )
            publish()
            if spec.purpose == "inspection" or spec.weight_reads:
                weights = [backend.inspect_weight_spec(read) for read in spec.weight_reads]
                backend.verify_base_unchanged()
                if weights:
                    capture_bytes = len(
                        json.dumps([weight.model_dump() for weight in weights]).encode()
                    )
                if capture_bytes > spec.budget.max_capture_bytes:
                    raise NrlyError("budget_exceeded", "weight preview byte budget exceeded")
                inspection = ModelInspection(tensors=backend.describe(), weights=weights)
                backend.verify_base_unchanged()
            for case in spec.cases:
                remaining = spec.budget.max_output_tokens - output_tokens
                if (
                    spec.model.backend == "huggingface"
                    and spec.diagnostics is None
                    and remaining < spec.generation.max_new_tokens
                ):
                    raise NrlyError(
                        "budget_exceeded",
                        "remaining output token budget cannot honor the generation specification",
                    )
                generation = spec.generation
                started = time.perf_counter()
                try:
                    result = (
                        backend.run_case(case, spec.program, spec.captures, generation)
                        if spec.diagnostics is None
                        else backend.run_case(
                            case, spec.program, spec.captures, generation, spec.diagnostics
                        )
                    )
                    if not result.base_unchanged:
                        raise NrlyError("restoration_failed", "base model was not restored")
                    if spec.model.backend == "huggingface" and result.output_tokens is None:
                        raise NrlyError(
                            "invalid_backend_result", "output token accounting is missing"
                        )
                    if result.output_tokens is not None and not (
                        0 <= result.output_tokens <= min(generation.max_new_tokens, remaining)
                    ):
                        raise NrlyError(
                            "budget_exceeded", "backend exceeded its output token budget"
                        )
                    output_tokens += result.output_tokens or 0
                    if result.captures:
                        capture_bytes += len(
                            json.dumps(
                                [capture.model_dump() for capture in result.captures]
                            ).encode()
                        )
                    if result.weight_changes:
                        capture_bytes += len(
                            json.dumps(
                                [change.model_dump() for change in result.weight_changes]
                            ).encode()
                        )
                    if result.diagnostics is not None:
                        capture_bytes += len(
                            json.dumps(result.diagnostics.model_dump(), ensure_ascii=False).encode()
                        )
                    if capture_bytes > spec.budget.max_capture_bytes:
                        raise NrlyError("budget_exceeded", "capture byte budget exceeded")
                    samples.append(
                        SampleResult(
                            case_id=case.id,
                            status="completed",
                            provenance="torch_fixture"
                            if spec.model.backend == "tiny"
                            else "huggingface",
                            output=result.output,
                            elapsed_ms=result.elapsed_ms,
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                            captures=result.captures,
                            finish_reason=result.finish_reason,
                            weight_changes=result.weight_changes,
                            diagnostics=result.diagnostics,
                            truncated=result.truncated if spec.schema_version == "0.3" else None,
                        )
                    )
                    publish()
                except Exception as error:
                    error_info = (
                        error.as_info()
                        if isinstance(error, NrlyError)
                        else ErrorInfo(code="execution_failed", message=str(error)[:2000])
                    )
                    samples.append(
                        SampleResult(
                            case_id=case.id,
                            status="failed",
                            provenance="torch_fixture"
                            if spec.model.backend == "tiny"
                            else "huggingface",
                            elapsed_ms=(time.perf_counter() - started) * 1000,
                            error=error_info,
                        )
                    )
                    break
    except Exception as error:
        error_info = (
            error.as_info()
            if isinstance(error, NrlyError)
            else ErrorInfo(code="backend_failed", message=str(error)[:2000])
        )
    publish()
    return 1 if error_info else 0


def parent_watchdog() -> None:
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        # Blocking reads here can deadlock native DLL initialization on Windows.
        # Poll the inherited pipe without blocking or taking Python stream locks.
        peek_pipe = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
        peek_pipe.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        peek_pipe.restype = wintypes.BOOL
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        while bool(peek_pipe(handle, None, 0, None, None, None)):
            time.sleep(0.1)
        closed = True
    else:
        closed = os.read(sys.stdin.fileno(), 1) == b""
    if closed:
        os._exit(70)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--watch-parent", action="store_true")
    arguments = parser.parse_args()
    if arguments.watch_parent:
        threading.Thread(target=parent_watchdog, daemon=True).start()
    return execute(arguments.spec, arguments.result)


if __name__ == "__main__":
    raise SystemExit(main())
