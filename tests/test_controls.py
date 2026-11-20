from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from nrly import controls
from nrly.backends.tiny import TinyBackend
from nrly.contracts import ModelSpec, NrlyError, ProgramSpec, TensorSlice, WeightOperationSpec
from nrly.controls import (
    MAX_CONTROL_OPERATIONS,
    matched_random_control,
    random_orthogonal_direction,
)
from nrly.kernel import intervene, tensor_digest
from nrly.lever_contracts import LeverSite
from nrly.recipes import compile_removal

TARGET = "blocks.1.mlp.out.weight"


def _low_rank(
    left: object, right: object, factor: float = 1.0, target: str = TARGET,
) -> WeightOperationSpec:
    lhs, rhs = np.asarray(left, float), np.asarray(right, float)
    return WeightOperationSpec(
        kind="weight_low_rank", target=target, shape=[lhs.shape[0], rhs.shape[1]],
        rank=lhs.shape[1], left=lhs.ravel().tolist(), right=rhs.ravel().tolist(), factor=factor,
    )


def _sparse(
    shape: list[int], indices: list[int], values: list[float], target: str = TARGET,
) -> WeightOperationSpec:
    return WeightOperationSpec(
        kind="weight_sparse_add", target=target, shape=shape, indices=indices, values=values,
    )


def _factors(operation: WeightOperationSpec) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = operation.shape
    left = np.array(operation.left).reshape(rows, operation.rank)
    return left, np.array(operation.right).reshape(operation.rank, columns)


def _delta(operation: WeightOperationSpec) -> np.ndarray:
    left, right = _factors(operation)
    return operation.factor * left @ right


def _refused(operations: list[WeightOperationSpec], seed: int = 0) -> NrlyError:
    with pytest.raises(NrlyError) as error:
        matched_random_control(operations, seed)
    assert error.value.code == "unsupported_control"
    return error.value


@pytest.mark.parametrize("width", [2, 3, 64, 4096])
def test_random_orthogonal_direction_is_a_deterministic_orthogonal_unit_vector(
    width: int,
) -> None:
    direction = 7 * np.random.default_rng(width).normal(size=width)
    snapshot = direction.copy()
    control = random_orthogonal_direction(direction, 5)
    assert control.shape == (width,) and control.dtype == np.float64
    assert abs(float(control @ direction)) < 1e-12 * np.linalg.norm(direction)
    assert float(np.linalg.norm(control)) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(direction, snapshot)
    np.testing.assert_array_equal(random_orthogonal_direction(direction, 5), control)
    # Only the direction matters, not its length or container.
    rescaled = random_orthogonal_direction((direction / 3).tolist(), 5)
    np.testing.assert_allclose(rescaled, control, atol=1e-12)
    if width > 2:
        assert np.abs(random_orthogonal_direction(direction, 6) - control).max() > 1e-3


def test_random_orthogonal_directions_cover_the_whole_complement() -> None:
    # In two dimensions the complement is one line; both of its unit vectors must occur.
    assert {
        round(float(random_orthogonal_direction([0.0, 1.0], seed)[0])) for seed in range(20)
    } == {-1, 1}
    # A QR factorization pins the sign of a column's leading entry; a control must not.
    controls = np.array([random_orthogonal_direction([0.0, 0.0, 1.0], s) for s in range(200)])
    np.testing.assert_allclose(controls[:, 2], 0, atol=1e-12)
    positive = (controls[:, :2] > 0).mean(axis=0)
    assert np.all(positive > 0.35) and np.all(positive < 0.65)


class _Draws:
    """Stands in for a generator so that an unlucky draw can be stated exactly."""

    def __init__(self, draws: list[np.ndarray]) -> None:
        self.draws, self.used = draws, 0

    def standard_normal(self, shape: tuple[int, ...]) -> np.ndarray:
        self.used += 1
        return self.draws[self.used - 1].reshape(shape).copy()


def test_unlucky_draws_still_give_exact_orthogonality_or_an_explicit_refusal() -> None:
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(64, 1))
    basis /= np.linalg.norm(basis)
    other = rng.normal(size=(64, 1))
    # A draw almost along the basis: one projection alone would leave an error near 1e-12.
    frame = controls._complement_frame(basis, _Draws([basis + 1e-6 * other]))  # type: ignore[arg-type]
    assert abs(float(frame[:, 0] @ basis[:, 0])) < 1e-15
    assert float(np.linalg.norm(frame)) == pytest.approx(1.0, abs=1e-12)
    # A draw exactly along the basis has no orthogonal part and is drawn again.
    retried = _Draws([3 * basis, other])
    frame = controls._complement_frame(basis, retried)  # type: ignore[arg-type]
    assert retried.used == 2 and abs(float(frame[:, 0] @ basis[:, 0])) < 1e-15
    exhausted = _Draws([basis * scale for scale in range(1, 9)])
    with pytest.raises(NrlyError) as error:
        controls._complement_frame(basis, exhausted)  # type: ignore[arg-type]
    assert error.value.code == "unsupported_control" and exhausted.used == 8


def test_random_direction_control_is_an_equally_strong_removal_of_something_else() -> None:
    rng = np.random.default_rng(2)
    weights, direction = rng.normal(size=(6, 9)), rng.normal(size=6)
    unit = direction / np.linalg.norm(direction)
    control = random_orthogonal_direction(direction, 1)
    site = LeverSite(parameter=TARGET, module="blocks.1.mlp")
    [edit] = compile_removal(site, weights, direction, 0.75)
    [shadow] = compile_removal(site, weights, control, 0.75)
    assert shadow.model_dump(exclude={"left", "right"}) == edit.model_dump(
        exclude={"left", "right"}
    )
    updated = weights + _delta(shadow)
    # The lever's own direction is written exactly as before; the control's is reduced.
    np.testing.assert_allclose(unit @ updated, unit @ weights, atol=1e-12)
    np.testing.assert_allclose(control @ updated, 0.25 * (control @ weights), atol=1e-12)
    assert np.linalg.norm(_delta(shadow)) > 0.1


@pytest.mark.parametrize(
    ("direction", "seed", "code"),
    [
        ([3.0], 0, "unsupported_control"),
        ([0.0, 0.0], 0, "invalid_authoring"),
        ([[1.0, 2.0]], 0, "invalid_authoring"),
        ([1.0, float("nan")], 0, "invalid_authoring"),
        ([1.0, 2.0], -1, "invalid_authoring"),
        ([1.0, 2.0], 2**32, "invalid_authoring"),
        ([1.0, 2.0], True, "invalid_authoring"),
        ([1.0, 2.0], 1.5, "invalid_authoring"),
        ([1.0, 2.0], "7", "invalid_authoring"),
    ],
)
def test_random_orthogonal_direction_refuses_bad_directions_and_seeds(
    direction: list[float], seed: int, code: str,
) -> None:
    with pytest.raises(NrlyError) as error:
        random_orthogonal_direction(direction, seed)
    assert error.value.code == code


def test_low_rank_control_keeps_input_sensitivity_and_size_in_a_new_output_direction() -> None:
    rng = np.random.default_rng(4)
    edit = _low_rank(rng.normal(size=(6, 1)), rng.normal(size=(1, 4)), factor=-0.7)
    [control] = matched_random_control([edit], 9)
    assert control.model_dump(exclude={"left"}) == edit.model_dump(exclude={"left"})
    (left, right), (moved, kept) = _factors(edit), _factors(control)
    np.testing.assert_array_equal(kept, right)
    assert abs(float(moved[:, 0] @ left[:, 0])) < 1e-12
    assert float(np.linalg.norm(moved)) == pytest.approx(float(np.linalg.norm(left)), rel=1e-12)
    assert float(np.linalg.norm(_delta(control))) == pytest.approx(
        float(np.linalg.norm(_delta(edit))), rel=1e-12
    )
    # Every input moves the output exactly as far as the edit moves it, somewhere else.
    probes = rng.normal(size=(4, 8))
    np.testing.assert_allclose(
        np.linalg.norm(_delta(control) @ probes, axis=0),
        np.linalg.norm(_delta(edit) @ probes, axis=0), rtol=1e-12,
    )
    np.testing.assert_allclose(left[:, 0] @ _delta(control), 0, atol=1e-12)
    assert np.abs(left[:, 0] @ _delta(edit)).max() > 0.1


def test_higher_rank_control_keeps_column_norms_angles_and_the_update_spectrum() -> None:
    rng = np.random.default_rng(6)
    left = rng.normal(size=(12, 3))
    left[:, 1] = 0.9 * left[:, 0] + 0.1 * left[:, 1]  # nearly parallel columns
    left[:, 2] *= 40
    edit = _low_rank(left, rng.normal(size=(3, 5)), factor=0.3)
    [control] = matched_random_control([edit], 21)
    moved, kept = _factors(control)
    np.testing.assert_array_equal(kept, _factors(edit)[1])
    scale = np.outer(np.linalg.norm(left, axis=0), np.linalg.norm(left, axis=0))
    # Each new column is orthogonal to its original, and to every other original column.
    assert np.abs(moved.T @ left / scale).max() < 1e-12
    np.testing.assert_allclose(
        np.linalg.norm(moved, axis=0), np.linalg.norm(left, axis=0), rtol=1e-12,
    )
    assert np.abs((moved.T @ moved - left.T @ left) / scale).max() < 1e-12
    np.testing.assert_allclose(
        np.linalg.svd(_delta(control), compute_uv=False)[:3],
        np.linalg.svd(_delta(edit), compute_uv=False)[:3], rtol=1e-10,
    )
    # Columns drawn independently would have lost the near-parallel pair's shared size.
    independent = np.linalg.norm(left[:, 0]) ** 2 + np.linalg.norm(left[:, 1]) ** 2
    assert np.linalg.norm(left[:, 0] + left[:, 1]) ** 2 > 1.9 * independent
    assert np.linalg.norm(moved[:, 0] + moved[:, 1]) ** 2 > 1.9 * independent


def test_low_rank_control_leaves_an_unused_column_unused() -> None:
    left = np.zeros((6, 2))
    left[:, 0] = np.random.default_rng(8).normal(size=6)
    edit = _low_rank(left, np.ones((2, 3)))
    [control] = matched_random_control([edit], 2)
    moved, _ = _factors(control)
    np.testing.assert_array_equal(moved[:, 1], 0)
    assert abs(float(moved[:, 0] @ left[:, 0])) < 1e-12
    assert float(np.linalg.norm(moved[:, 0])) == pytest.approx(float(np.linalg.norm(left[:, 0])))


@pytest.mark.parametrize(
    "edit",
    [
        _low_rank([[1.0]], [[0.5, 0.5]], target="output.weight"),
        _low_rank(np.ones((3, 2)), np.ones((2, 4))),
        _low_rank(np.zeros((6, 1)), np.ones((1, 4))),
    ],
    ids=["one-row tensor", "fewer than two rows per rank", "zero left factor"],
)
def test_low_rank_control_refuses_factors_without_an_orthogonal_complement(
    edit: WeightOperationSpec,
) -> None:
    assert "another site" in str(_refused([edit]))


def test_low_rank_control_refuses_a_factor_too_large_to_point_elsewhere() -> None:
    # Each entry is finite, but the column's norm is not, so no rotation of it can be stored.
    edit = _low_rank(np.full((16, 1), 1.79e308), np.ones((1, 2)))
    with pytest.raises(NrlyError) as error:
        matched_random_control([edit], 0)
    assert error.value.code == "invalid_authoring"


def test_sparse_control_moves_the_same_values_to_untouched_positions() -> None:
    edit = _sparse([4, 5], [0, 3, 7], [1.0, -2.0, 0.5]).model_copy(update={"factor": 2.0})
    placements, orders = set(), set()
    for seed in range(50):
        [control] = matched_random_control([edit], seed)
        assert control.model_dump(exclude={"indices", "values"}) == edit.model_dump(
            exclude={"indices", "values"}
        )
        assert len(set(control.indices)) == 3
        assert set(control.indices).isdisjoint(edit.indices)
        assert all(0 <= index < 20 for index in control.indices)
        assert sorted(control.values) == sorted(edit.values)
        placements.add(tuple(control.indices))
        orders.add(tuple(control.values))
    assert len(placements) > 20 and len(orders) > 3


def test_sparse_control_uses_exactly_the_free_positions_when_half_the_tensor_is_edited() -> None:
    edit = _sparse([6], [0, 2, 4], [1.0, 2.0, 3.0])
    for seed in range(10):
        [control] = matched_random_control([edit], seed)
        assert set(control.indices) == {1, 3, 5}
    assert "another site" in str(_refused([_sparse([5], [0, 1, 2], [1.0, 2.0, 3.0])]))


def test_sparse_control_avoids_every_sparse_edit_of_the_same_tensor() -> None:
    first, second = _sparse([6], [0, 1], [1.0, 2.0]), _sparse([6], [2, 3], [3.0, 4.0])
    elsewhere = _sparse([6], [0, 1], [5.0, 6.0], target="blocks.2.mlp.out.weight")
    for seed in range(10):
        controls = matched_random_control([first, second, elsewhere], seed)
        assert set(controls[0].indices) == {4, 5} == set(controls[1].indices)
        assert set(controls[2].indices) <= {2, 3, 4, 5}
    crowded = [_sparse([5], [0, 1], [1.0, 2.0]), _sparse([5], [2, 3], [3.0, 4.0])]
    _refused(crowded)


def test_sparse_control_never_enumerates_a_huge_tensor() -> None:
    side = 2**31 - 1
    edit = _sparse([side, side], [0, 5, 2**61], [1.0, 2.0, 3.0])
    [control] = matched_random_control([edit], 1)
    assert len(set(control.indices)) == 3 and set(control.indices).isdisjoint(edit.indices)
    assert all(0 <= index < side * side for index in control.indices)
    assert WeightOperationSpec.model_validate(control.model_dump()) == control


def test_dense_control_permutes_the_same_values_into_a_different_edit() -> None:
    edit = WeightOperationSpec(
        kind="weight_add", target=TARGET, shape=[2, 3], values=[1, 2, 3, 4, 5, 6], factor=0.5,
    )
    orders = set()
    for seed in range(30):
        [control] = matched_random_control([edit], seed)
        assert control.model_dump(exclude={"values"}) == edit.model_dump(exclude={"values"})
        assert sorted(control.values) == edit.values and control.values != edit.values
        orders.add(tuple(control.values))
    assert len(orders) > 20
    # With two values half of all shuffles reproduce the edit; a control never does.
    pair = WeightOperationSpec(kind="weight_add", target=TARGET, shape=[2], values=[1.0, 2.0])
    assert all(matched_random_control([pair], seed)[0].values == [2.0, 1.0] for seed in range(20))
    uniform = pair.model_copy(update={"values": [0.5, 0.5]})
    assert "another site" in str(_refused([uniform]))


@pytest.mark.parametrize(
    ("edit", "restated"),
    [
        (
            WeightOperationSpec(
                kind="weight_patch", target=TARGET, shape=[2, 2], indices=[0], values=[1.0],
            ),
            "weight_sparse_add",
        ),
        (
            WeightOperationSpec(
                kind="weight_replace", target=TARGET, shape=[2], values=[1.0, 2.0],
            ),
            "weight_add",
        ),
        (WeightOperationSpec(kind="weight_scale", target=TARGET, factor=2.0), "weight_add"),
        (
            WeightOperationSpec(
                kind="weight_slice_scale", target=TARGET, shape=[2, 2], factor=0.5,
                slices=[TensorSlice(stop=1), TensorSlice(stop=2)],
            ),
            "weight_sparse_add",
        ),
    ],
)
def test_edits_whose_size_depends_on_the_weights_are_refused_with_an_alternative(
    edit: WeightOperationSpec, restated: str,
) -> None:
    message = str(_refused([edit]))
    assert edit.kind in message and f"{restated} delta" in message and "another site" in message
    # One unsupported operation refuses the whole edit; there is no partial control.
    _refused([_sparse([4, 5], [0], [1.0]), edit])


def test_controls_are_deterministic_validated_and_never_touch_their_input() -> None:
    rng = np.random.default_rng(10)
    operations = [
        _low_rank(rng.normal(size=(8, 2)), rng.normal(size=(2, 3)), factor=-1.5),
        _sparse([4, 5], [1, 2, 19], [0.25, -0.5, 4.0]),
        WeightOperationSpec(kind="weight_add", target="bias", shape=[4], values=[1, 2, 3, 4]),
    ]
    snapshot = [operation.model_dump() for operation in operations]
    controls = matched_random_control(operations, 12)
    assert [control.kind for control in controls] == [op.kind for op in operations]
    assert [control.target for control in controls] == [op.target for op in operations]
    for control, operation in zip(controls, operations, strict=True):
        assert WeightOperationSpec.model_validate(control.model_dump()) == control
        assert control != operation
    assert matched_random_control(tuple(operations), 12) == controls
    assert matched_random_control(operations, 13) != controls
    # The results own their payloads: changing them cannot reach the edit they shadow.
    controls[0].right.clear()
    controls[1].indices.reverse()
    controls[2].values.append(9.0)
    assert [operation.model_dump() for operation in operations] == snapshot
    assert matched_random_control([], 3) == []


def test_control_inputs_are_bounded_and_revalidated() -> None:
    edit = _sparse([4, 5], [0], [1.0])
    assert len(matched_random_control([edit] * MAX_CONTROL_OPERATIONS, 0)) == 32
    with pytest.raises(NrlyError) as error:
        matched_random_control([edit] * (MAX_CONTROL_OPERATIONS + 1), 0)
    assert error.value.code == "budget_exceeded"
    for seed in (-1, 2**32, True, 1.5):
        with pytest.raises(NrlyError) as error:
            matched_random_control([edit], seed)  # type: ignore[arg-type]
        assert error.value.code == "invalid_authoring"
    edit.values.append(2.0)  # the contract is frozen, but its lists are not
    with pytest.raises(ValidationError):
        matched_random_control([edit], 0)


def test_controls_execute_through_the_kernel_as_equally_large_weight_changes() -> None:
    edit = [
        _low_rank([[0.6], [0.8]], [[0.5, -0.25]], factor=-1.0, target="hidden.weight"),
        WeightOperationSpec(
            kind="weight_add", target="output.weight", shape=[1, 2], values=[0.25, -0.5],
        ),
    ]
    control = matched_random_control(edit, 4)
    assert control[1].values == [-0.5, 0.25]
    with TinyBackend(ModelSpec()) as backend:
        base, before = backend.identity.fingerprint, tensor_digest(backend.model)
        original = backend.read_parameter("hidden.weight")
        deltas = []
        for operations in (edit, control):
            program = ProgramSpec(kind="pipeline", expected_base=base, operations=operations)
            with intervene(backend.model, program, base) as changes:
                assert [change.operation_kind for change in changes] == [
                    "weight_low_rank", "weight_add",
                ]
                assert changes[1].changed_elements == 2
                deltas.append(backend.read_parameter("hidden.weight") - original)
            assert tensor_digest(backend.model) == before
        applied, shadow = deltas
        assert float(np.linalg.norm(applied)) == pytest.approx(0.5590170, rel=1e-6)
        assert float(np.linalg.norm(shadow)) == pytest.approx(
            float(np.linalg.norm(applied)), rel=1e-6
        )
        # Same size, same input sensitivity, and no shared output direction.
        assert abs(float(np.sum(applied * shadow))) < 1e-6
        np.testing.assert_allclose(shadow.T @ [0.6, 0.8], 0, atol=1e-6)

        moved = matched_random_control([_sparse([2, 2], [0], [0.25], "hidden.weight")], 4)
        program = ProgramSpec(kind="pipeline", expected_base=base, operations=moved)
        with intervene(backend.model, program, base) as changes:
            assert changes[0].changed_elements == 1
            assert changes[0].edited_indices == moved[0].indices != [0]
            difference = backend.read_parameter("hidden.weight") - original
            assert float(np.abs(difference).sum()) == pytest.approx(0.25)
        assert tensor_digest(backend.model) == before
