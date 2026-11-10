from __future__ import annotations

import json
import math
from typing import cast

import torch
from torch import Tensor, nn

from nrly.backends.torch_backend import TorchBackend, model_device, model_dtype
from nrly.contracts import CaseSpec, GenerationSpec, ModelSpec, NrlyError


class TinySum(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 2, bias=False)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(2))
            self.output.weight.fill_(0.5)

    def forward(self, value: Tensor) -> Tensor:
        return cast(Tensor, self.output(self.relu(self.hidden(value))))


class TinyBackend(TorchBackend):
    def __init__(self, spec: ModelSpec) -> None:
        model = TinySum().to(device=model_device(spec), dtype=model_dtype(spec))
        super().__init__(
            spec,
            model,
            {
                "fixture": True,
                "task": "nonnegative sum with a deliberate output scale defect",
            },
        )

    def _execute(self, case: CaseSpec, generation: GenerationSpec) -> tuple[str, None, None, str]:
        del generation
        if case.input is None:
            raise NrlyError("unsupported_operation", "tiny backend does not accept chat messages")
        try:
            raw = json.loads(case.input)
            if (
                not isinstance(raw, list)
                or len(raw) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in raw)
            ):
                raise ValueError("invalid vector")
            values = [float(x) for x in raw]
            if any(not math.isfinite(x) or x < 0 for x in values):
                raise ValueError("invalid value")
        except (ValueError, TypeError, OverflowError) as error:
            raise NrlyError(
                "invalid_input",
                "tiny input must be a JSON array of two finite nonnegative numbers",
            ) from error
        parameter = next(self.model.parameters())
        value = torch.tensor([values], device=parameter.device, dtype=parameter.dtype)
        result = float(self.model(value).item())
        if not math.isfinite(result):
            raise NrlyError("nonfinite_tensor", "the fixture output is not finite")
        return format(result, ".10g"), None, None, "completed"
