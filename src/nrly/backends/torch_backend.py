from __future__ import annotations

import gc
import hashlib
import json
import platform
from abc import ABC, abstractmethod
from collections.abc import Callable
from time import perf_counter
from types import TracebackType
from typing import cast

import torch
from pydantic import JsonValue
from torch import nn

from nrly.contracts import (
    BackendResult,
    CaptureSpec,
    CaseSpec,
    ContinuationScore,
    DiagnosticSpec,
    GenerationSpec,
    LocalModelSpec,
    ModelIdentity,
    ModelSpec,
    NrlyError,
    ProgramSpec,
    TensorCapture,
    TensorInfo,
    WeightReadSpec,
)
from nrly.kernel import capture_activations, intervene, tensor_digest
from nrly.kernel.inspection import inspect_tensor


class TorchBackend(ABC):
    def __init__(
        self,
        spec: ModelSpec | LocalModelSpec,
        model: nn.Module,
        extra: dict[str, JsonValue] | None = None,
    ) -> None:
        self.spec = spec
        self.model = model.eval()
        self.closed = False
        self._base_digest = tensor_digest(model)
        first = next(model.parameters())
        details: dict[str, JsonValue] = {
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "device": str(first.device),
            "dtype": str(first.dtype),
            "weights_sha256": self._base_digest,
            "cuda_runtime": torch.version.cuda,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "capture_semantics": (
                "latest forward invocation; first max_elements in flattened output; "
                "post-intervention"
            ),
            "execution": "native PyTorch forward hooks; no model reimplementation",
            **(extra or {}),
        }
        if first.device.type == "cuda":
            details["device_name"] = torch.cuda.get_device_name(first.device)
            cudnn_version = cast(Callable[[], int | None], torch.backends.cudnn.version)
            details["cudnn_version"] = cudnn_version()
        manifest = {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "details": details,
        }
        fingerprint = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.identity = ModelIdentity(
            backend=spec.backend,
            model_id=spec.model_id,
            revision=spec.revision,
            fingerprint=fingerprint,
            details=details,
        )

    def __enter__(self) -> TorchBackend:
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _error: BaseException | None,
        _trace: TracebackType | None,
    ) -> None:
        self.close()

    def describe(self) -> list[TensorInfo]:
        self._require_open()
        modules = [
            TensorInfo(name=name, kind="module")
            for name, _module in self.model.named_modules()
            if name
        ]
        parameters = [
            TensorInfo(
                name=name,
                kind="parameter",
                shape=list(value.shape),
                dtype=str(value.dtype),
            )
            for name, value in self.model.named_parameters(remove_duplicate=False)
        ]
        return modules + parameters

    def read_tensor(self, name: str, max_elements: int = 16) -> dict[str, JsonValue]:
        capture = self.inspect_weight(name, max_elements)
        return {"name": name, **capture.model_dump(mode="json", exclude={"target"})}

    def inspect_weight(self, target: str, max_elements: int = 16) -> TensorCapture:
        if not 1 <= max_elements <= 4096:
            raise NrlyError("invalid_request", "max_elements must be between 1 and 4096")
        return self.inspect_weight_spec(WeightReadSpec(target=target, max_elements=max_elements))

    def inspect_weight_spec(self, spec: WeightReadSpec) -> TensorCapture:
        self._require_open()
        parameters = dict(self.model.named_parameters(remove_duplicate=False))
        if spec.target not in parameters:
            raise NrlyError("target_mismatch", f"unknown parameter: {spec.target}")
        return inspect_tensor(parameters[spec.target], spec)

    def run_case(
        self,
        case: CaseSpec,
        program: ProgramSpec,
        captures: list[CaptureSpec],
        generation: GenerationSpec,
        diagnostics: DiagnosticSpec | None = None,
    ) -> BackendResult:
        self._require_open()
        self.verify_base_unchanged()
        started = perf_counter()
        diagnostic_result = None
        input_tokens: int | None
        output_tokens: int | None
        try:
            with (
                intervene(self.model, program, self.identity.fingerprint) as changes,
                capture_activations(self.model, captures) as recorded,
            ):
                with torch.inference_mode():
                    if diagnostics is not None:
                        diagnostic_result = self._diagnose(case, generation, diagnostics)
                        output, input_tokens, output_tokens, finish_reason = (
                            "",
                            diagnostic_result.prompt_tokens,
                            0,
                            "teacher_forced",
                        )
                    else:
                        if case.continuation is not None:
                            raise NrlyError("invalid_request", "continuation requires diagnostics")
                        output, input_tokens, output_tokens, finish_reason = self._execute(
                            case, generation
                        )
        finally:
            self.verify_base_unchanged()
        return BackendResult(
            output=output,
            elapsed_ms=(perf_counter() - started) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            captures=list(recorded.values()),
            finish_reason=finish_reason,
            base_unchanged=True,
            weight_changes=changes,
            diagnostics=diagnostic_result,
            truncated=(finish_reason == "length")
            if finish_reason in {"length", "eos", "teacher_forced"}
            else None,
        )

    def _diagnose(
        self, case: CaseSpec, generation: GenerationSpec, diagnostics: DiagnosticSpec
    ) -> ContinuationScore:
        raise NrlyError("unsupported_operation", "this backend does not support causal diagnostics")

    def verify_base_unchanged(self) -> None:
        self._require_open()
        if tensor_digest(self.model) != self._base_digest:
            raise NrlyError(
                "restoration_failed",
                "model state differs from the pinned base",
            )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_open(self) -> None:
        if self.closed:
            raise NrlyError("backend_closed", "the model backend is closed")

    @abstractmethod
    def _execute(
        self, case: CaseSpec, generation: GenerationSpec
    ) -> tuple[str, int | None, int | None, str]:
        raise NotImplementedError


def model_device(spec: ModelSpec) -> torch.device:
    requested = "cuda" if spec.device == "auto" and torch.cuda.is_available() else spec.device
    if requested == "auto":
        requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise NrlyError("device_unavailable", "CUDA was requested but is unavailable")
    return torch.device(requested)


def model_dtype(spec: ModelSpec) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[spec.dtype]
