from __future__ import annotations

from typing import TYPE_CHECKING

from nrly.contracts import ModelSpec, NrlyError

if TYPE_CHECKING:
    from nrly.backends.torch_backend import TorchBackend


def load_backend(spec: ModelSpec) -> TorchBackend:
    try:
        if spec.backend == "tiny":
            from nrly.backends.tiny import TinyBackend

            return TinyBackend(spec)
        from nrly.backends.huggingface import HuggingFaceBackend

        return HuggingFaceBackend(spec)
    except ImportError as error:
        raise NrlyError("backend_unavailable", f"missing model dependency: {error.name}") from error
