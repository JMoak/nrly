"""Matched controls: edits as large as a candidate that have no reason to matter.

Without an equally large edit somewhere unrelated, nobody can tell a lever from a
generic perturbation. These generators are trusted local authoring. They never read
weights, and they emit bounded declarative operations of the same kind and size as
the edit they shadow.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from nrly.authoring import FloatArray, unit_direction
from nrly.contracts import NrlyError, WeightOperationSpec

MAX_CONTROL_OPERATIONS = 32
_DRAWS = 8
_MIN_PIVOT = 1e-6
# Kinds whose size depends on the weights they overwrite or rescale, mapped to the
# additive kind that states the same edit as a delta a control can match.
_NEEDS_WEIGHTS = {
    "weight_patch": "weight_sparse_add",
    "weight_replace": "weight_add",
    "weight_scale": "weight_add",
    "weight_slice_scale": "weight_sparse_add",
}
_ELSEWHERE = "apply the same edit at another site as the control instead"


def _generator(seed: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise NrlyError("invalid_authoring", "a control seed must be an integer in 0..2**32-1")
    return np.random.Generator(np.random.PCG64(seed))


def _complement_frame(basis: FloatArray, generator: np.random.Generator) -> FloatArray:
    """Random orthonormal columns, one per orthonormal column of ``basis``, orthogonal to it."""
    for _ in range(_DRAWS):
        frame = generator.standard_normal(basis.shape)
        # The second pass removes the rounding that the first projection leaves behind.
        for _ in range(2):
            frame, triangle = np.linalg.qr(frame - basis @ (basis.T @ frame))
            pivots = np.diag(triangle)
            if float(np.abs(pivots).min()) <= _MIN_PIVOT:
                break
            # LAPACK fixes each column's sign; undoing that keeps the frame uniformly random.
            frame = frame * np.sign(pivots)
        else:
            return frame
    raise NrlyError(
        "unsupported_control", f"no orthogonal direction could be drawn; {_ELSEWHERE}",
    )


def random_orthogonal_direction(direction: ArrayLike, seed: int) -> FloatArray:
    """A deterministic unit vector orthogonal to ``direction``.

    Removing it with the same strength, at the same site, is the random-direction
    control for a removal of ``direction``.
    """
    generator = _generator(seed)
    unit = unit_direction(direction)
    if unit.size < 2:
        raise NrlyError(
            "unsupported_control",
            f"a one-dimensional direction has no orthogonal direction; {_ELSEWHERE}",
        )
    return _complement_frame(unit.reshape(-1, 1), generator).reshape(-1)


def _rotated_left(operation: WeightOperationSpec, generator: np.random.Generator) -> list[float]:
    rows, rank = operation.shape[0], cast(int, operation.rank)
    left = np.asarray(operation.left, dtype=np.float64).reshape(rows, rank)
    scale = float(np.abs(left).max())
    if rows < 2 * rank or scale == 0:
        raise NrlyError(
            "unsupported_control",
            "a low-rank control needs a nonzero left factor with at least two rows per rank; "
            + _ELSEWHERE,
        )
    # Moving the whole factor into the orthogonal complement of its own span keeps every
    # column norm and every angle between columns, so the update keeps its exact size.
    basis, triangle = np.linalg.qr(left / scale)
    frame = _complement_frame(basis, generator)
    with np.errstate(over="ignore", invalid="ignore"):
        control = frame @ triangle * scale
    if not np.isfinite(control).all():
        raise NrlyError("invalid_authoring", "control factors must be finite")
    return control.ravel().tolist()


def _moved_positions(
    operation: WeightOperationSpec, edited: NDArray[np.int64], generator: np.random.Generator,
) -> tuple[list[int], list[float]]:
    total, count = math.prod(operation.shape), len(operation.indices)
    occupied = edited[edited < total]
    free = total - int(occupied.size)
    if free < count:
        raise NrlyError(
            "unsupported_control",
            "the sparse edit leaves too few untouched positions for a disjoint control; "
            + _ELSEWHERE,
        )
    ranks = np.sort(generator.choice(free, size=count, replace=False))
    # The untouched position of rank r lies past every edited position that has at most r
    # untouched positions below it. This never materializes the tensor's positions.
    below = occupied - np.arange(occupied.size)
    positions = ranks + np.searchsorted(below, ranks, side="right")
    values = generator.permutation(np.asarray(operation.values, dtype=np.float64))
    return cast(list[int], positions.tolist()), cast(list[float], values.tolist())


def _permuted(operation: WeightOperationSpec, generator: np.random.Generator) -> list[float]:
    values = np.asarray(operation.values, dtype=np.float64)
    if values.min() == values.max():
        raise NrlyError(
            "unsupported_control",
            f"a dense edit that repeats one value has no distinct permutation; {_ELSEWHERE}",
        )
    shuffled = generator.permutation(values)
    if np.array_equal(shuffled, values):
        # A permutation that reproduces the edit is not a control: exchange two unequal entries.
        other = int(np.argmax(values != values[0]))
        shuffled[[0, other]] = shuffled[[other, 0]]
    return cast(list[float], shuffled.tolist())


def matched_random_control(
    operations: Sequence[WeightOperationSpec], seed: int,
) -> list[WeightOperationSpec]:
    """Shadow each operation with an equally large one that has no reason to matter.

    For edits whose direction is unknown. ``weight_low_rank`` keeps its right factor and
    moves its left factor to random output directions orthogonal to the original ones,
    with the same column norms and update norm. ``weight_sparse_add`` puts the same
    values, permuted, on as many positions that no sparse operation on that tensor
    touches. ``weight_add`` permutes its values. Kinds whose size depends on the current
    weights are refused. The input is never modified, and a seed fixes the result.
    """
    generator = _generator(seed)
    if len(operations) > MAX_CONTROL_OPERATIONS:
        raise NrlyError("budget_exceeded", "a control shadows at most 32 operations")
    checked = [WeightOperationSpec.model_validate(item.model_dump()) for item in operations]
    sparse: dict[str, list[int]] = {}
    for operation in checked:
        if operation.kind == "weight_sparse_add":
            sparse.setdefault(str(operation.target), []).extend(operation.indices)
    edited = {
        target: np.unique(np.asarray(indices, dtype=np.int64))
        for target, indices in sparse.items()
    }
    controls = []
    for operation in checked:
        content = operation.model_dump()
        if operation.kind == "weight_low_rank":
            content["left"] = _rotated_left(operation, generator)
        elif operation.kind == "weight_sparse_add":
            content["indices"], content["values"] = _moved_positions(
                operation, edited[str(operation.target)], generator,
            )
        elif operation.kind == "weight_add":
            content["values"] = _permuted(operation, generator)
        else:
            delta = _NEEDS_WEIGHTS.get(operation.kind, "weight_add")
            raise NrlyError(
                "unsupported_control",
                f"a matched control for {operation.kind} depends on the current weights; "
                f"restate the edit as a {delta} delta against them, or {_ELSEWHERE}",
            )
        controls.append(WeightOperationSpec.model_validate(content))
    return controls
