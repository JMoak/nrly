"""Trusted local Python calculations that compile to bounded declarative programs.

These helpers are an authoring API, never a Python execution endpoint. Large local
arrays are useful for calculations, but emitted programs retain the kernel's
32-operation and 65,536-value limits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nrly.contracts import NrlyError, ProgramSpec, WeightOperationSpec

FloatArray = NDArray[np.float64]
MAX_AUTHORING_ELEMENTS = 16 * 1024 * 1024


def _array(values: ArrayLike) -> FloatArray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or not 1 <= array.size <= MAX_AUTHORING_ELEMENTS:
        raise NrlyError("invalid_authoring", "expected a bounded, nonempty real numeric array")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise NrlyError("invalid_authoring", "authoring values must be finite")
    return result


def unit_direction(values: ArrayLike) -> FloatArray:
    """Normalize a finite, nonzero vector without overflow in the norm."""
    value = _array(values)
    if value.ndim != 1:
        raise NrlyError("invalid_authoring", "a direction must be a vector")
    maximum = float(np.max(np.abs(value)))
    if maximum == 0:
        raise NrlyError("invalid_authoring", "a direction must be nonzero")
    scaled = value / maximum
    return scaled / np.linalg.norm(scaled)


def mean_direction(positive: ArrayLike, negative: ArrayLike) -> FloatArray:
    """Unit difference between two sets of row vectors; no quality claim implied."""
    left, right = _array(positive), _array(negative)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise NrlyError("invalid_authoring", "direction samples need matching feature widths")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = left.mean(axis=0) - right.mean(axis=0)
    return unit_direction(difference)


def project_out(values: ArrayLike, direction: ArrayLike) -> FloatArray:
    """Remove the direction from the last axis of vectors or matrices."""
    value, unit = _array(values), unit_direction(direction)
    if value.ndim not in {1, 2} or value.shape[-1] != unit.size:
        raise NrlyError("invalid_authoring", "projection requires a matching final dimension")
    with np.errstate(over="ignore", invalid="ignore"):
        projected = value - np.expand_dims(value @ unit, -1) * unit
    return _array(projected)


class WeightProgram:
    """Build ordered edits against one exact base fingerprint.

    Calculation inputs must come from the intended current weights. For a later
    repair, compute against the parent's edited weights and append the resulting
    delta with EditStore.create(parent_id=...). No code enters the saved program.
    """

    def __init__(self, expected_base: str) -> None:
        if not expected_base:
            raise NrlyError("invalid_authoring", "authoring requires an exact base fingerprint")
        self.expected_base = expected_base
        self._operations: list[WeightOperationSpec] = []

    def append(self, operation: WeightOperationSpec) -> Self:
        validated = WeightOperationSpec.model_validate(operation.model_dump())
        proposed = [*self._operations, validated]
        ProgramSpec(kind="pipeline", expected_base=self.expected_base, operations=proposed)
        self._operations = proposed
        return self

    def scale(self, target: str, factor: float) -> Self:
        return self.append(WeightOperationSpec(kind="weight_scale", target=target, factor=factor))

    def add(self, target: str, values: ArrayLike, *, factor: float = 1) -> Self:
        value = _array(values)
        if value.size > 65536:
            raise NrlyError("budget_exceeded", "compiled tensor payload exceeds 65536 values")
        return self.append(
            WeightOperationSpec(
                kind="weight_add", target=target, shape=list(value.shape),
                values=value.ravel().tolist(), factor=factor,
            )
        )

    def patch(
        self, target: str, shape: Sequence[int], indices: Sequence[int], values: ArrayLike,
        *, additive: bool = False, factor: float = 1,
    ) -> Self:
        value = _array(values)
        if value.size > 65536 or len(indices) > 65536:
            raise NrlyError("budget_exceeded", "compiled sparse payload exceeds 65536 values")
        if value.ndim != 1 or (not additive and factor != 1):
            raise NrlyError("invalid_authoring", "patch values must be a vector; scaling needs add")
        return self.append(
            WeightOperationSpec(
                kind="weight_sparse_add" if additive else "weight_patch", target=target,
                shape=list(shape), indices=list(indices), values=value.tolist(), factor=factor,
            )
        )

    def low_rank(
        self, target: str, left: ArrayLike, right: ArrayLike, *, factor: float = 1,
    ) -> Self:
        lhs, rhs = _array(left), _array(right)
        if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[0]:
            raise NrlyError("invalid_authoring", "low-rank factors require compatible matrices")
        if lhs.size + rhs.size > 65536:
            raise NrlyError("budget_exceeded", "compiled low-rank factors exceed 65536 values")
        return self.append(
            WeightOperationSpec(
                kind="weight_low_rank", target=target, shape=[lhs.shape[0], rhs.shape[1]],
                rank=lhs.shape[1], left=lhs.ravel().tolist(), right=rhs.ravel().tolist(),
                factor=factor,
            )
        )

    def remove_input_direction(
        self, target: str, weights: ArrayLike, direction: ArrayLike, *, strength: float = 1,
    ) -> Self:
        """Compile W <- W - strength * (W @ d) @ d.T as a rank-one edit."""
        weight, unit = _array(weights), unit_direction(direction)
        if weight.ndim != 2 or weight.shape[1] != unit.size:
            raise NrlyError("invalid_authoring", "weight columns must match the direction")
        with np.errstate(over="ignore", invalid="ignore"):
            left = (weight @ unit).reshape(-1, 1)
        return self.low_rank(target, left, unit.reshape(1, -1), factor=-strength)

    def steer_key(
        self, target: str, weights: ArrayLike, key: ArrayLike, desired: ArrayLike,
    ) -> Self:
        """Minimum-Frobenius rank-one delta satisfying (W + delta) @ key = desired.

        This is a local linear constraint, not a claim about downstream behavior.
        """
        weight, vector, output = _array(weights), _array(key), _array(desired)
        unit = unit_direction(vector)
        if (
            weight.ndim != 2 or vector.ndim != 1 or output.ndim != 1
            or weight.shape != (output.size, vector.size)
        ):
            raise NrlyError("invalid_authoring", "weight, key and desired output shapes disagree")
        scale = float(np.max(np.abs(vector)))
        normalized_norm = float(np.linalg.norm(vector / scale))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            left = ((output / scale) / normalized_norm - weight @ unit).reshape(-1, 1)
        return self.low_rank(target, left, unit.reshape(1, -1))

    def build(self) -> ProgramSpec:
        if not self._operations:
            return ProgramSpec(expected_base=self.expected_base)
        return ProgramSpec.model_validate(
            ProgramSpec(
                kind="pipeline", expected_base=self.expected_base, operations=self._operations,
            ).model_dump()
        )
