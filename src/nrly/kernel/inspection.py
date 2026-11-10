from __future__ import annotations

import math

import torch
from torch import Tensor

from nrly.contracts import NrlyError, TensorCapture, TensorStatistics, WeightReadSpec
from nrly.kernel.interventions import _finite, _value_digest, flat_coordinate, tensor_slices


def inspect_tensor(value: Tensor, spec: WeightReadSpec) -> TensorCapture:
    """Read logical row-major coordinates, without running or changing the model."""
    value = value.detach()
    if not value.is_floating_point() or value.layout != torch.strided:
        raise NrlyError("unsupported_operation", "inspection requires dense real floating tensors")
    shape = list(value.shape)
    total = value.numel()
    selection_shape: list[int]
    if spec.indices:
        indices = spec.indices
        selection_shape = [len(indices)]
    elif spec.coordinates:
        indices = []
        for coordinate in spec.coordinates:
            if len(coordinate) != len(shape) or any(
                item >= width for item, width in zip(coordinate, shape, strict=True)
            ):
                raise NrlyError("target_mismatch", "coordinate is outside the tensor shape")
            flat = 0
            for item, width in zip(coordinate, shape, strict=True):
                flat = flat * width + item
            indices.append(flat)
        selection_shape = [len(indices)]
    elif spec.slices:
        tensor_slices(spec.slices, shape)
        selection_shape = [len(range(part.start, part.stop, part.step)) for part in spec.slices]
        count = math.prod(selection_shape)
        if count > spec.max_elements:
            raise NrlyError("invalid_request", "slice selection exceeds max_elements")
        indices = []
        for offset in range(count):
            coordinate = flat_coordinate(offset, selection_shape)
            flat = 0
            for item, part, width in zip(coordinate, spec.slices, shape, strict=True):
                flat = flat * width + part.start + item * part.step
            indices.append(flat)
    else:
        if spec.offset >= total and (total or spec.offset):
            raise NrlyError("target_mismatch", "offset is outside the tensor")
        indices = list(range(spec.offset, min(total, spec.offset + spec.max_elements)))
        selection_shape = [len(indices)]
    if any(index >= total for index in indices):
        raise NrlyError("target_mismatch", "index is outside the tensor")
    coordinates = [flat_coordinate(index, shape) for index in indices]
    if shape and coordinates:
        axes = tuple(
            torch.tensor(axis, device=value.device, dtype=torch.long)
            for axis in zip(*coordinates, strict=True)
        )
        selected = value[axes]
    else:
        selected = value.reshape(-1)[indices]
    selected = selected.double().cpu()
    _finite(selected)
    return TensorCapture(
        target=spec.target,
        shape=shape,
        dtype=str(value.dtype),
        values=selected.tolist(),
        truncated=len(indices) < total,
        flat_indices=indices,
        coordinates=coordinates,
        selection_shape=selection_shape,
        statistics=_statistics(selected),
        tensor_sha256=_value_digest(value),
    )


def _statistics(values: Tensor) -> TensorStatistics:
    if not values.numel():
        return TensorStatistics(count=0, zero_count=0)
    scale = float(values.abs().max().item())
    normalized = values / scale if scale else values
    return TensorStatistics(
        count=values.numel(),
        minimum=float(values.min().item()),
        maximum=float(values.max().item()),
        mean=float(normalized.mean().item()) * scale,
        std=float(normalized.std(correction=0).item()) * scale,
        rms=math.sqrt(float(normalized.square().mean().item())) * scale,
        zero_count=int(torch.count_nonzero(values == 0).item()),
    )
