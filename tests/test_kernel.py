from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from nrly.backends.tiny import TinySum
from nrly.contracts import CaptureSpec, NrlyError, ProgramSpec
from nrly.kernel import capture_activations, intervene, tensor_digest


def test_weight_scale_changes_computation_and_restores_base() -> None:
    model = TinySum()
    before = tensor_digest(model)
    value = torch.tensor([[2.0, 4.0]])
    assert model(value).item() == 3
    with intervene(
        model,
        ProgramSpec(kind="weight_scale", target="output.weight", factor=2),
        before,
    ):
        assert model(value).item() == 6
        assert tensor_digest(model) != before
    assert tensor_digest(model) == before
    assert model(value).item() == 3


def test_weight_replace_changes_exact_parameter() -> None:
    model = TinySum()
    before = tensor_digest(model)
    program = ProgramSpec(
        kind="weight_replace", target="output.weight", values=[1, 1], shape=[1, 2]
    )
    with intervene(model, program, before):
        assert model(torch.tensor([[3.0, 7.0]])).item() == 10
    assert tensor_digest(model) == before


def test_sparse_weight_patch_changes_only_selected_elements_and_restores() -> None:
    model = TinySum()
    before = tensor_digest(model)
    program = ProgramSpec(
        kind="weight_patch",
        target="output.weight",
        shape=[1, 2],
        indices=[0],
        values=[1.5],
    )
    with intervene(model, program, before) as changes:
        assert model(torch.tensor([[2.0, 4.0]])).item() == 5
        assert model.output.weight.tolist() == [[1.5, 0.5]]
        assert changes[0].changed_elements == 1
        assert changes[0].total_elements == 2
        assert changes[0].before_sha256 != changes[0].after_sha256
        assert changes[0].edited_indices == [0]
        assert changes[0].edited_before == [0.5]
        assert changes[0].edited_after == [1.5]
    assert tensor_digest(model) == before


def test_sparse_patch_uses_logical_flattening_for_noncontiguous_parameter() -> None:
    model = nn.Module()
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).T
    model.register_parameter("weight", nn.Parameter(original))
    before = tensor_digest(model)
    program = ProgramSpec(
        kind="weight_patch", target="weight", shape=[2, 2], indices=[1], values=[9]
    )
    with intervene(model, program, before):
        assert model.weight.tolist() == [[1, 9], [2, 4]]
    assert model.weight.tolist() == [[1, 3], [2, 4]]
    assert tensor_digest(model) == before


def test_sparse_patch_preview_shows_selected_values_beyond_prefix() -> None:
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.ones(40)))
    before = tensor_digest(model)
    program = ProgramSpec(
        kind="weight_patch", target="weight", shape=[40], indices=[39], values=[7]
    )
    with intervene(model, program, before) as changes:
        assert changes[0].before_preview == changes[0].after_preview == [1] * 32
        assert changes[0].edited_indices == [39]
        assert changes[0].edited_before == [1]
        assert changes[0].edited_after == [7]
    assert tensor_digest(model) == before


def test_changed_element_count_matches_byte_hash_for_signed_zero() -> None:
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.zeros(1)))
    before = tensor_digest(model)
    program = ProgramSpec(
        kind="weight_patch", target="weight", shape=[1], indices=[0], values=[-0.0]
    )
    with intervene(model, program, before) as changes:
        assert changes[0].changed_elements == 1
        assert changes[0].before_sha256 != changes[0].after_sha256
    assert tensor_digest(model) == before


@pytest.mark.parametrize("indices,values", [([0, 0], [1, 2]), ([-1], [1]), ([2], [1]), ([0], [])])
def test_sparse_patch_contract_rejects_invalid_indices(
    indices: list[int], values: list[float]
) -> None:
    with pytest.raises(ValidationError):
        ProgramSpec(
            kind="weight_patch",
            target="output.weight",
            shape=[1, 2],
            indices=indices,
            values=values,
        )


@pytest.mark.parametrize(
    "program",
    [
        ProgramSpec(kind="weight_scale", target="missing.weight", factor=2),
        ProgramSpec(kind="weight_replace", target="output.weight", values=[1, 1], shape=[2]),
        ProgramSpec(
            kind="weight_patch", target="output.weight", values=[1], indices=[0], shape=[2]
        ),
        ProgramSpec(
            kind="weight_patch", target="output.weight", values=[1], indices=[2], shape=[3]
        ),
        ProgramSpec(kind="activation_scale", target="missing", factor=2),
    ],
)
def test_unknown_or_misshaped_target_refuses_without_mutation(
    program: ProgramSpec,
) -> None:
    model = TinySum()
    before = tensor_digest(model)
    with pytest.raises(NrlyError) as error, intervene(model, program, before):
        pytest.fail("invalid target was admitted")
    assert error.value.code == "target_mismatch"
    assert tensor_digest(model) == before


def test_wrong_base_refuses_even_identity() -> None:
    with (
        pytest.raises(NrlyError) as error,
        intervene(TinySum(), ProgramSpec(expected_base="wrong"), "actual"),
    ):
        pytest.fail("wrong base was admitted")
    assert error.value.code == "base_mismatch"


@pytest.mark.parametrize(
    "program",
    [
        ProgramSpec(kind="weight_scale", target="output.weight", factor=2),
        ProgramSpec(kind="activation_scale", target="hidden", factor=2),
        ProgramSpec(
            kind="weight_patch", target="output.weight", shape=[1, 2], indices=[0, 1], values=[1, 1]
        ),
    ],
)
def test_interventions_restore_after_body_exception(program: ProgramSpec) -> None:
    model = TinySum()
    before = tensor_digest(model)
    with (
        pytest.raises(RuntimeError, match="injected"),
        intervene(model, program, before),
    ):
        assert model(torch.tensor([[2.0, 4.0]])).item() == 6
        raise RuntimeError("injected")
    assert tensor_digest(model) == before
    assert model(torch.tensor([[2.0, 4.0]])).item() == 3
    assert not model.hidden._forward_hooks


def test_tied_parameter_is_rejected() -> None:
    model = nn.Module()
    shared = nn.Linear(2, 2, bias=False)
    model.add_module("left", shared)
    model.add_module("right", shared)
    with (
        pytest.raises(NrlyError) as error,
        intervene(
            model,
            ProgramSpec(kind="weight_scale", target="left.weight", factor=2),
            "base",
        ),
    ):
        pytest.fail("tied edit was admitted")
    assert error.value.code == "unsupported_operation"


def test_distinct_parameters_sharing_storage_are_rejected() -> None:
    model = nn.Module()
    first = nn.Parameter(torch.ones(2, 2))
    model.register_parameter("first", first)
    model.register_parameter("second", nn.Parameter(first.detach()))
    with (
        pytest.raises(NrlyError) as error,
        intervene(model, ProgramSpec(kind="weight_scale", target="first", factor=2), "base"),
    ):
        pytest.fail("shared-storage edit was admitted")
    assert error.value.code == "unsupported_operation"


def test_nonfinite_replacement_is_rejected_before_mutation() -> None:
    model = TinySum()
    before = tensor_digest(model)
    with (
        pytest.raises(NrlyError) as error,
        intervene(
            model,
            ProgramSpec(kind="weight_scale", target="output.weight", factor=1e308),
            before,
        ),
    ):
        pytest.fail("non-finite replacement was admitted")
    assert error.value.code == "nonfinite_tensor"
    assert tensor_digest(model) == before


def test_last_token_operation_leaves_other_positions_intact() -> None:
    model = nn.Sequential()
    model.add_module("block", nn.Identity())
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    program = ProgramSpec(kind="activation_scale", target="block", factor=2, position="last")
    with intervene(model, program, "base"):
        changed = model(value)
    assert torch.equal(changed, torch.tensor([[[1.0, 2.0], [6.0, 8.0]]]))
    assert torch.equal(value, torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    assert torch.equal(model(value), value)


def test_activation_add_broadcast_is_limited_to_hidden_width() -> None:
    model = nn.Sequential()
    model.add_module("block", nn.Identity())
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    program = ProgramSpec(kind="activation_add", target="block", shape=[2], values=[10, 20])
    with intervene(model, program, "base"):
        assert torch.equal(model(value), torch.tensor([[[11.0, 22.0], [13.0, 24.0]]]))
    wrong = ProgramSpec(kind="activation_add", target="block", shape=[1], values=[10])
    with pytest.raises(NrlyError) as error, intervene(model, wrong, "base"):
        model(value)
    assert error.value.code == "target_mismatch"
    assert not model.block._forward_hooks


def test_capture_is_bounded_latest_and_post_intervention() -> None:
    model = TinySum()
    program = ProgramSpec(kind="activation_scale", target="hidden", factor=2)
    with (
        intervene(model, program, "base"),
        capture_activations(model, [CaptureSpec(target="hidden", max_elements=1)]) as captures,
    ):
        model(torch.tensor([[2.0, 4.0]]))
        model(torch.tensor([[3.0, 5.0]]))
    assert captures["hidden"].values == [6]
    assert captures["hidden"].shape == [1, 2]
    assert captures["hidden"].truncated
    assert not model.hidden._forward_hooks


def test_invalid_capture_removes_previously_installed_intervention() -> None:
    model = TinySum()
    with (
        pytest.raises(NrlyError),
        intervene(
            model,
            ProgramSpec(kind="activation_scale", target="hidden", factor=2),
            "base",
        ),
        capture_activations(model, [CaptureSpec(target="missing")]),
    ):
        pytest.fail("invalid capture was admitted")
    assert not model.hidden._forward_hooks
    assert model(torch.tensor([[2.0, 4.0]])).item() == 3


def test_tensor_digest_handles_scalar_buffers_and_detects_changes() -> None:
    model = TinySum()
    model.register_buffer("counter", torch.tensor(0, dtype=torch.int64))
    before = tensor_digest(model)
    model.counter.add_(1)
    assert tensor_digest(model) != before
