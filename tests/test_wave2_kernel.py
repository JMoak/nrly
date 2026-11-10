from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from nrly.backends import load_backend
from nrly.backends.tiny import TinySum
from nrly.contracts import (
    CaptureSpec,
    CaseSpec,
    ExperimentSpec,
    GenerationSpec,
    ModelSpec,
    NrlyError,
    ProgramSpec,
    RunRecord,
    TensorSlice,
    WeightOperationSpec,
    WeightReadSpec,
    canonical_json,
    digest,
)
from nrly.kernel import intervene, tensor_digest
from nrly.kernel.inspection import inspect_tensor
from nrly.worker import execute


def pipeline(model: nn.Module, *operations: WeightOperationSpec) -> ProgramSpec:
    return ProgramSpec(
        kind="pipeline", expected_base=tensor_digest(model), operations=list(operations)
    )


def test_ordered_pipeline_chains_same_target_evidence_and_restores_all_targets() -> None:
    model = TinySum()
    before = tensor_digest(model)
    program = pipeline(
        model,
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2),
        WeightOperationSpec(
            kind="weight_sparse_add", target="output.weight", shape=[1, 2], indices=[1], values=[1]
        ),
        WeightOperationSpec(kind="weight_scale", target="hidden.weight", factor=3),
    )
    with intervene(model, program, before) as changes:
        assert model(torch.tensor([[2.0, 4.0]])).item() == 30
        assert changes[0].after_sha256 == changes[1].before_sha256
        assert [item.operation_index for item in changes] == [0, 1, 2]
        assert changes[1].edited_indices == [1]
        assert changes[1].edited_coordinates == [[0, 1]]
        assert changes[1].edited_before == [1]
        assert changes[1].edited_after == [2]
    assert tensor_digest(model) == before


def test_pipeline_checks_every_target_before_any_write(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TinySum()
    before = tensor_digest(model)
    program = pipeline(
        model,
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2),
        WeightOperationSpec(kind="weight_scale", target="unknown", factor=2),
    )
    copies = 0

    def unexpected_copy(*args: object, **kwargs: object) -> None:
        nonlocal copies
        copies += 1
        raise AssertionError("a validation failure must not publish weights")

    monkeypatch.setattr(torch.Tensor, "copy_", unexpected_copy)
    with pytest.raises(NrlyError, match="unknown parameter"), intervene(model, program, before):
        pytest.fail("invalid program was admitted")
    assert copies == 0
    assert tensor_digest(model) == before


def test_pipeline_restores_even_partially_failed_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TinySum()
    before = tensor_digest(model)
    program = pipeline(
        model,
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2),
        WeightOperationSpec(kind="weight_scale", target="hidden.weight", factor=3),
    )
    original_copy = torch.Tensor.copy_
    copies = 0

    def failing_copy(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        nonlocal copies
        copies += 1
        result = original_copy(target, source)
        if copies == 2:
            raise RuntimeError("injected partial copy failure")
        return result

    monkeypatch.setattr(torch.Tensor, "copy_", failing_copy)
    with pytest.raises(RuntimeError, match="partial copy"), intervene(model, program, before):
        pytest.fail("failing apply entered the model body")
    assert copies == 4
    assert tensor_digest(model) == before


def test_pipeline_restores_on_body_exception_and_refuses_wrong_base() -> None:
    model = TinySum()
    before = tensor_digest(model)
    program = pipeline(
        model,
        WeightOperationSpec(kind="weight_scale", target="hidden.weight", factor=2),
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2),
    )
    with pytest.raises(RuntimeError, match="body"), intervene(model, program, before):
        assert model(torch.tensor([[2.0, 4.0]])).item() == 12
        raise RuntimeError("body")
    assert tensor_digest(model) == before
    with pytest.raises(NrlyError) as error, intervene(model, program, "different"):
        pytest.fail("wrong base accepted")
    assert error.value.code == "base_mismatch"


def test_later_nonfinite_operation_leaves_earlier_targets_untouched() -> None:
    model = TinySum()
    before = tensor_digest(model)
    program = pipeline(
        model,
        WeightOperationSpec(kind="weight_scale", target="hidden.weight", factor=2),
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=1e308),
    )
    with pytest.raises(NrlyError) as error, intervene(model, program, before):
        pytest.fail("overflow accepted")
    assert error.value.code == "nonfinite_tensor"
    assert tensor_digest(model) == before


def test_low_rank_edit_uses_fp32_factors_and_stored_dtype_evidence() -> None:
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.tensor([[3.0]], dtype=torch.bfloat16)))
    before = tensor_digest(model)
    program = ProgramSpec(
        kind="weight_low_rank",
        target="weight",
        shape=[1, 1],
        rank=2,
        left=[256, 257],
        right=[1, -1],
    )
    with intervene(model, program, before) as changes:
        # Premature BF16 conversion rounds 257 to 256 and loses this update.
        assert model.weight.item() == 2
        assert changes[0].changed_elements == 1
        assert changes[0].dtype == "torch.bfloat16"
        assert changes[0].after_preview == [2]
    assert tensor_digest(model) == before


def test_low_rank_edit_matches_dense_reference() -> None:
    model = nn.Module()
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    model.register_parameter("weight", nn.Parameter(original.clone()))
    left = torch.tensor([[1, 2], [3, 4], [-1, 1]], dtype=torch.float32)
    right = torch.tensor([[1, 2, 3, 4], [-2, 0, 1, 2]], dtype=torch.float32)
    program = ProgramSpec(
        kind="weight_low_rank",
        target="weight",
        shape=[3, 4],
        rank=2,
        left=left.flatten().tolist(),
        right=right.flatten().tolist(),
        factor=0.5,
    )
    before = tensor_digest(model)
    with intervene(model, program, before):
        assert torch.equal(model.weight, original + 0.5 * (left @ right))
    assert tensor_digest(model) == before


@pytest.mark.parametrize("kind", ["weight_add", "weight_sparse_add"])
def test_addition_accumulates_before_bf16_rounding(kind: str) -> None:
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.ones((1, 1), dtype=torch.bfloat16)))
    payload = {"kind": kind, "target": "weight", "shape": [1, 1], "values": [0.00391]}
    if kind == "weight_sparse_add":
        payload["indices"] = [0]
    before = tensor_digest(model)
    with intervene(model, ProgramSpec.model_validate(payload), before):
        assert model.weight.item() == 1.0078125
    assert tensor_digest(model) == before


@pytest.mark.parametrize(
    "program",
    [
        ProgramSpec(kind="weight_add", target="weight", shape=[1, 2], values=[0, 0]),
        ProgramSpec(
            kind="weight_sparse_add", target="weight", shape=[1, 2], indices=[0, 1], values=[0, 0]
        ),
        ProgramSpec(
            kind="weight_low_rank",
            target="weight",
            shape=[1, 2],
            rank=1,
            left=[1e308],
            right=[1e308, 1e308],
            factor=0,
        ),
    ],
)
def test_additive_noop_controls_preserve_signed_zero_bytes(program: ProgramSpec) -> None:
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.tensor([[-0.0, 1.0]])))
    before = tensor_digest(model)
    with intervene(model, program, before) as changes:
        assert changes[0].changed_elements == 0
        assert changes[0].before_sha256 == changes[0].after_sha256
        assert tensor_digest(model) == before
    assert tensor_digest(model) == before


def test_subprecision_low_rank_update_reports_actual_zero_changes() -> None:
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.ones((1, 2), dtype=torch.bfloat16)))
    program = ProgramSpec(
        kind="weight_low_rank",
        target="weight",
        shape=[1, 2],
        rank=1,
        left=[0.0001],
        right=[1, 1],
    )
    with intervene(model, program, tensor_digest(model)) as changes:
        assert changes[0].changed_elements == 0
        assert changes[0].before_sha256 == changes[0].after_sha256


def test_block_scale_edits_only_selected_coordinates_beyond_prefix() -> None:
    model = nn.Module()
    original = torch.ones((64, 128))
    model.register_parameter("weight", nn.Parameter(original.clone()))
    before = tensor_digest(model)
    program = ProgramSpec(
        kind="weight_slice_scale",
        target="weight",
        shape=[64, 128],
        factor=-2,
        slices=[TensorSlice(start=40, stop=42), TensorSlice(start=8, stop=12, step=2)],
    )
    with intervene(model, program, before) as changes:
        expected = original.clone()
        expected[40:42, 8:12:2] = -2
        assert torch.equal(model.weight, expected)
        assert changes[0].changed_elements == 4
        assert changes[0].edited_indices == [5128, 5130, 5256, 5258]
        assert changes[0].edited_coordinates == [[40, 8], [40, 10], [41, 8], [41, 10]]
        assert changes[0].edited_before == [1] * 4
        assert changes[0].edited_after == [-2] * 4
        assert changes[0].before_preview == changes[0].after_preview
    assert tensor_digest(model) == before


def test_weight_parameter_aliasing_buffer_is_rejected() -> None:
    model = TinySum()
    model.register_buffer("shared", model.output.weight.detach())
    before = tensor_digest(model)
    with (
        pytest.raises(NrlyError) as error,
        intervene(
            model, ProgramSpec(kind="weight_scale", target="output.weight", factor=2), before
        ),
    ):
        pytest.fail("shared storage accepted")
    assert error.value.code == "unsupported_operation"
    assert tensor_digest(model) == before


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "pipeline", "operations": []},
        {"kind": "pipeline", "operations": [{"kind": "weight_scale", "target": "a"}]},
        {"kind": "pipeline", "expected_base": "base", "operations": [{"kind": "identity"}]},
        {
            "kind": "weight_low_rank",
            "target": "a",
            "shape": [2, 2],
            "rank": 1,
            "left": [1],
            "right": [1, 2],
        },
        {
            "kind": "weight_low_rank",
            "target": "a",
            "shape": [2, 2],
            "rank": True,
            "left": [1, 2],
            "right": [1, 2],
        },
        {"kind": "weight_patch", "target": "a", "shape": [2**31, 1], "indices": [0], "values": [1]},
        {
            "kind": "weight_patch",
            "target": "a",
            "shape": [2**31 - 1] * 3,
            "indices": [0],
            "values": [1],
        },
        {
            "kind": "weight_low_rank",
            "target": "a",
            "shape": [1, 1],
            "rank": 1,
            "left": [float("inf")],
            "right": [1],
        },
    ],
)
def test_program_contract_rejects_ambiguous_or_unbounded_programs(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ProgramSpec.model_validate(payload)


def test_extended_read_is_exact_beyond_4096_and_statistics_describe_selection() -> None:
    value = torch.arange(8192, dtype=torch.float32).reshape(64, 128)
    capture = inspect_tensor(value, WeightReadSpec(target="weight", offset=5123, max_elements=3))
    assert capture.values == [5123, 5124, 5125]
    assert capture.flat_indices == [5123, 5124, 5125]
    assert capture.coordinates == [[40, 3], [40, 4], [40, 5]]
    assert capture.selection_shape == [3]
    assert capture.statistics is not None
    assert capture.statistics.scope == "selection"
    assert capture.statistics.mean == 5124
    assert capture.statistics.std == pytest.approx((2 / 3) ** 0.5)
    assert capture.tensor_sha256 and len(capture.tensor_sha256) == 64
    assert capture.truncated


def test_matrix_slice_and_coordinate_reads_match_and_preserve_tensor() -> None:
    value = torch.arange(8192, dtype=torch.float32).reshape(64, 128)
    original = value.clone()
    sliced = inspect_tensor(
        value,
        WeightReadSpec(
            target="weight",
            max_elements=4,
            slices=[TensorSlice(start=40, stop=42), TensorSlice(start=8, stop=12, step=2)],
        ),
    )
    indexed = inspect_tensor(
        value, WeightReadSpec(target="weight", indices=[5128, 5130, 5256, 5258])
    )
    coordinates = inspect_tensor(
        value, WeightReadSpec(target="weight", coordinates=[[40, 8], [40, 10], [41, 8], [41, 10]])
    )
    assert sliced.values == indexed.values == coordinates.values == [5128, 5130, 5256, 5258]
    assert sliced.tensor_sha256 == indexed.tensor_sha256 == coordinates.tensor_sha256
    assert sliced.selection_shape == [2, 2]
    assert torch.equal(value, original)


def test_index_order_and_noncontiguous_logical_coordinates_are_preserved() -> None:
    value = torch.arange(8192, dtype=torch.float32).reshape(64, 128).T
    capture = inspect_tensor(value, WeightReadSpec(target="weight", indices=[5160, 0, 100]))
    assert capture.coordinates == [[80, 40], [0, 0], [1, 36]]
    assert capture.values == [5200, 0, 4609]


@pytest.mark.parametrize(
    "read",
    [
        WeightReadSpec(target="weight", offset=8192),
        WeightReadSpec(target="weight", indices=[8192]),
        WeightReadSpec(target="weight", coordinates=[[64, 0]]),
        WeightReadSpec(target="weight", coordinates=[[1]]),
        WeightReadSpec(target="weight", slices=[TensorSlice(stop=64)]),
        WeightReadSpec(target="weight", slices=[TensorSlice(stop=65), TensorSlice(stop=1)]),
        WeightReadSpec(target="weight", slices=[TensorSlice(stop=64), TensorSlice(stop=128)]),
    ],
)
def test_read_refuses_bounds_and_oversized_selections(read: WeightReadSpec) -> None:
    value = torch.zeros(64, 128)
    with pytest.raises(NrlyError):
        inspect_tensor(value, read)
    assert torch.count_nonzero(value) == 0


@pytest.mark.parametrize(
    "selection",
    [
        {"indices": [-1]},
        {"indices": [True]},
        {"indices": [1, 1]},
        {"indices": [1], "offset": 1},
        {"coordinates": [[1], [1]]},
        {"indices": [0, 1], "max_elements": 1},
        {"coordinates": [[1] * 9]},
        {"slices": [{"start": 5, "stop": 3}]},
    ],
)
def test_read_contract_rejects_invalid_selectors(selection: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WeightReadSpec.model_validate({"target": "weight", **selection})


def test_statistics_handles_empty_singleton_and_large_float64_values() -> None:
    empty = inspect_tensor(torch.empty(0), WeightReadSpec(target="empty"))
    assert empty.statistics is not None and empty.statistics.count == 0
    assert empty.statistics.mean is None
    scalar = inspect_tensor(torch.tensor(3.0), WeightReadSpec(target="scalar"))
    assert scalar.values == [3]
    assert scalar.statistics is not None and scalar.statistics.std == 0
    huge = inspect_tensor(
        torch.tensor([-1e308, 1e308], dtype=torch.float64), WeightReadSpec(target="huge")
    )
    assert huge.statistics is not None
    assert huge.statistics.mean == 0
    assert huge.statistics.std == 1e308
    with pytest.raises(NrlyError) as error:
        inspect_tensor(torch.tensor([float("nan")]), WeightReadSpec(target="bad"))
    assert error.value.code == "nonfinite_tensor"


def test_backend_executes_pinned_pipeline_and_refuses_modified_base() -> None:
    with load_backend(ModelSpec()) as backend:
        program = ProgramSpec(
            kind="pipeline",
            expected_base=backend.identity.fingerprint,
            operations=[
                WeightOperationSpec(kind="weight_scale", target="hidden.weight", factor=2),
                WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2),
            ],
        )
        case = CaseSpec(id="sum", input="[2,4]")
        result = backend.run_case(case, program, [], GenerationSpec())
        assert result.output == "12"
        assert len(result.weight_changes) == 2
        assert backend.inspect_weight_spec(
            WeightReadSpec(target="hidden.weight", indices=[3])
        ).values == [1]
        with torch.no_grad():
            backend.model.output.weight.add_(1)
        with pytest.raises(NrlyError) as error:
            backend.run_case(case, program, [], GenerationSpec())
        assert error.value.code == "restoration_failed"


@pytest.mark.parametrize("name", ["baseline", "identity-weight", "sparse-patch"])
def test_real_legacy_bundle_preserves_exact_canonical_digests(name: str) -> None:
    bundle = Path(__file__).parents[1] / "docs/validation/qwen-runner/bundles" / name
    raw_spec = json.loads((bundle / "spec.json").read_text(encoding="utf-8"))
    raw_run = json.loads((bundle / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    spec = ExperimentSpec.model_validate(raw_spec)
    run = RunRecord.model_validate(raw_run)
    assert spec.schema_version == run.schema_version == "0.1"
    assert canonical_json(spec) == canonical_json(raw_spec)
    assert canonical_json(run) == canonical_json(raw_run)
    assert digest(spec) == manifest["spec_sha256"] == run.spec_digest
    assert digest(run) == manifest["run_sha256"]


def test_legacy_capture_constructor_and_explicit_schema_gate() -> None:
    spec = ExperimentSpec(
        name="legacy",
        schema_version="0.1",
        purpose="inspection",
        weight_reads=[CaptureSpec(target="weight")],
    )
    assert spec.weight_reads[0] == WeightReadSpec(target="weight")
    assert spec.model_dump()["weight_reads"] == [{"target": "weight", "max_elements": 16}]
    with pytest.raises(ValidationError, match="schema_version 0.2"):
        ExperimentSpec(
            name="extended",
            schema_version="0.1",
            purpose="inspection",
            weight_reads=[WeightReadSpec(target="weight", offset=1)],
        )
    with pytest.raises(ValidationError, match="schema_version 0.2"):
        ExperimentSpec(
            name="extended",
            schema_version="0.1",
            cases=[CaseSpec(id="x", input="[1,2]")],
            program=ProgramSpec(kind="weight_add", target="weight", shape=[1], values=[1]),
        )


def test_cancelling_pipeline_has_zero_final_net_edit_with_ordered_step_evidence() -> None:
    model = TinySum()
    before = tensor_digest(model)
    program = pipeline(
        model,
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2),
        WeightOperationSpec(kind="weight_scale", target="hidden.weight", factor=3),
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=0.5),
    )
    with intervene(model, program, before) as changes:
        assert changes[0].changed_elements == changes[2].changed_elements == 2
        assert changes[0].net_changed_elements is None
        assert changes[1].net_changed_elements == 2
        assert changes[2].net_changed_elements == 0
        assert changes[2].net_before_sha256 == changes[2].net_after_sha256
        assert changes[2].net_before_sha256 == changes[0].before_sha256
        assert model.output.weight.tolist() == [[0.5, 0.5]]
    assert tensor_digest(model) == before


def test_distinct_target_storage_budget_is_checked_before_cloning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nrly.kernel.interventions as kernel

    model = TinySum()
    before = tensor_digest(model)
    monkeypatch.setattr(kernel, "MAX_EDITED_WEIGHT_BYTES", 8)
    repeated = pipeline(
        model,
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2),
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=0.5),
    )
    with intervene(model, repeated, before):
        assert tensor_digest(model) == before
    too_large = pipeline(
        model, WeightOperationSpec(kind="weight_scale", target="hidden.weight", factor=2)
    )

    def unexpected_clone(*args: object, **kwargs: object) -> None:
        pytest.fail("budget must be checked before snapshots allocate")

    monkeypatch.setattr(torch.Tensor, "clone", unexpected_clone)
    with pytest.raises(NrlyError) as error, intervene(model, too_large, before):
        pytest.fail("storage budget was ignored")
    assert error.value.code == "budget_exceeded"
    assert tensor_digest(model) == before


def test_pipeline_operation_and_combined_payload_limits() -> None:
    with pytest.raises(ValidationError):
        ProgramSpec(
            kind="pipeline",
            expected_base="base",
            operations=[
                WeightOperationSpec(kind="weight_scale", target="weight") for _ in range(33)
            ],
        )
    operation = WeightOperationSpec(
        kind="weight_add", target="weight", shape=[32768], values=[0] * 32768
    )
    with pytest.raises(ValidationError, match="pipeline payload"):
        ProgramSpec(kind="pipeline", expected_base="base", operations=[operation] * 3)


def test_worker_persists_ordered_pipeline_net_and_read_evidence(tmp_path: Path) -> None:
    with load_backend(ModelSpec()) as backend:
        fingerprint = backend.identity.fingerprint
    spec = ExperimentSpec(
        name="atomic-worker",
        program=ProgramSpec(
            kind="pipeline",
            expected_base=fingerprint,
            operations=[
                WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2),
                WeightOperationSpec(
                    kind="weight_sparse_add",
                    target="output.weight",
                    shape=[1, 2],
                    indices=[1],
                    values=[0.5],
                ),
            ],
        ),
        cases=[CaseSpec(id="sum", input="[2,4]")],
        weight_reads=[WeightReadSpec(target="output.weight", coordinates=[[0, 1]])],
    )
    spec_path = tmp_path / "spec.json"
    result_path = tmp_path / "result.json"
    spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
    assert execute(spec_path, result_path) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["samples"][0]["output"] == "8"
    assert result["samples"][0]["weight_changes"][1]["net_changed_elements"] == 2
    assert result["samples"][0]["weight_changes"][1]["edited_after"] == [1.5]
    assert result["inspection"]["weights"][0]["flat_indices"] == [1]
    assert result["inspection"]["weights"][0]["values"] == [0.5]
    assert result["capture_bytes"] > 0


@pytest.mark.parametrize("kind", ["weight_add", "weight_sparse_add"])
def test_additive_factor_scales_delta_and_zero_factor_is_exact(kind: str) -> None:
    model = TinySum()
    before = tensor_digest(model)
    payload = {"kind": kind, "target": "output.weight", "shape": [1, 2], "values": [1, -1]}
    if kind == "weight_sparse_add":
        payload["indices"] = [0, 1]
    scaled = ProgramSpec.model_validate({**payload, "factor": 0.5})
    with intervene(model, scaled, before):
        assert model.output.weight.tolist() == [[1, 0]]
    noop = ProgramSpec.model_validate({**payload, "values": [1e308, -1e308], "factor": 0})
    with intervene(model, noop, before) as changes:
        assert tensor_digest(model) == before
        assert changes[0].changed_elements == 0
    assert tensor_digest(model) == before


def test_rollback_attempts_all_targets_when_one_restore_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TinySum()
    before = tensor_digest(model)
    program = pipeline(
        model,
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2),
        WeightOperationSpec(kind="weight_scale", target="hidden.weight", factor=3),
    )
    original_copy = torch.Tensor.copy_
    copies = 0

    def failing_restore(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        nonlocal copies
        copies += 1
        if copies == 3:
            raise RuntimeError("injected restoration failure")
        return original_copy(target, source)

    monkeypatch.setattr(torch.Tensor, "copy_", failing_restore)
    with pytest.raises(NrlyError) as error, intervene(model, program, before):
        assert model.output.weight.tolist() == [[1, 1]]
    assert error.value.code == "restoration_failed"
    assert copies == 4
    assert model.output.weight.tolist() == [[0.5, 0.5]]
    # The failed target remains corrupted; an honest failure is required.
    assert model.hidden.weight.tolist() == [[3, 0], [0, 3]]
