from __future__ import annotations

from collections.abc import Callable

import torch
from pydantic import Field
from torch import nn

from nrly.backends.torch_backend import TorchBackend
from nrly.contracts import CaseSpec, Contract, GenerationSpec, LocalModelSpec, NrlyError


class LocalExecution(Contract):
    output: str = Field(max_length=1000000)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    finish_reason: str = Field(default="completed", min_length=1, max_length=120)


LocalExecutor = Callable[[nn.Module, CaseSpec, GenerationSpec], LocalExecution]


class LocalTorchBackend(TorchBackend):
    """Wrap caller-owned native PyTorch execution; no imports, deserialization or code loading.

    ``revision`` and ``adapter_id`` are caller-declared identifiers, not verified code hashes.
    Model state is independently hashed and verified around every invocation. Callers must
    change their adapter identifier whenever their preprocessing or execution logic changes.
    This is a trusted in-process API, with no subprocess timeout or callback sandbox.
    """

    def __init__(
        self,
        model: nn.Module,
        execute: LocalExecutor,
        *,
        model_id: str,
        revision: str,
        adapter_id: str,
    ) -> None:
        if not adapter_id or len(adapter_id) > 200:
            raise NrlyError("invalid_request", "adapter_id must contain 1..200 characters")
        if not callable(execute):
            raise NrlyError("invalid_request", "execute must be a trusted Python callable")
        if next(model.parameters(), None) is None:
            raise NrlyError("unsupported_operation", "local model requires at least one parameter")
        self._executor = execute
        super().__init__(
            LocalModelSpec(model_id=model_id, revision=revision),
            model,
            {
                "adapter_id": adapter_id,
                "adapter_identity_source": "caller_declared",
                "trusted_python_callback": True,
                "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
                "fixture": False,
            },
        )

    def _execute(
        self, case: CaseSpec, generation: GenerationSpec
    ) -> tuple[str, int | None, int | None, str]:
        device = next(self.model.parameters()).device
        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(generation.seed)
            result = self._executor(self.model, case, generation)
        if not isinstance(result, LocalExecution):
            raise NrlyError("invalid_backend_result", "local executor must return LocalExecution")
        if result.output_tokens is not None and result.output_tokens > generation.max_new_tokens:
            raise NrlyError("budget_exceeded", "local executor exceeded requested output tokens")
        return result.output, result.input_tokens, result.output_tokens, result.finish_reason
