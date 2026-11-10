from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch import nn

from nrly.artifacts import (
    MAX_TENSOR_BYTES,
    TensorArtifact,
    TensorArtifactStore,
    read_local_file,
)
from nrly.authoring import WeightProgram, mean_direction, project_out, unit_direction
from nrly.backends.tiny import TinyBackend
from nrly.contracts import (
    CaseSpec,
    ExperimentSpec,
    GenerationSpec,
    ModelSpec,
    NrlyError,
    ProgramSpec,
    RunRecord,
    SampleResult,
    canonical_json,
    digest,
)
from nrly.deployment import apply_patch_package, export_patch, read_patch
from nrly.kernel import intervene, tensor_digest
from nrly.lineage import EditStore, compare_sequential, conflict_footprint


def test_directions_projection_and_rank_one_compile_to_real_weight_edits() -> None:
    direction = mean_direction([[2, 0], [4, 0]], [[0, 0]])
    np.testing.assert_allclose(direction, [1, 0])
    np.testing.assert_allclose(project_out([[2, 3]], direction), [[0, 3]])
    np.testing.assert_allclose(unit_direction([1e308, 1e308]), [2**-0.5, 2**-0.5])
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
    before = tensor_digest(model)
    weights = model.weight.detach().numpy().copy()
    program = WeightProgram(before).remove_input_direction("weight", weights, direction).build()
    assert program.operations[0].kind == "weight_low_rank"
    assert len(program.operations[0].left) + len(program.operations[0].right) == 4
    with intervene(model, program, before):
        np.testing.assert_allclose(model.weight.detach().numpy(), [[0, 3], [0, 5]])
    assert tensor_digest(model) == before


def test_steer_key_satisfies_linear_constraint_and_preserves_orthogonal_key() -> None:
    weights = np.array([[2.0, 3.0], [4.0, 5.0]])
    program = WeightProgram("base").steer_key("weight", weights, [2, 0], [12, 14]).build()
    operation = program.operations[0]
    left = np.array(operation.left).reshape(2, 1)
    right = np.array(operation.right).reshape(1, 2)
    updated = weights + left @ right
    np.testing.assert_allclose(updated @ [2, 0], [12, 14])
    np.testing.assert_allclose(updated @ [0, 1], weights @ [0, 1])


@pytest.mark.parametrize("values", [[0, 0], [float("nan"), 1], [True, False], [[1, 2]]])
def test_invalid_directions_are_rejected(values: object) -> None:
    with pytest.raises((NrlyError, TypeError)):
        unit_direction(values)  # type: ignore[arg-type]


def test_program_bound_and_failed_append_leave_builder_unchanged() -> None:
    builder = WeightProgram("base").scale("weight", 2)
    with pytest.raises(NrlyError, match="65536"):
        builder.add("weight", np.ones(65537))
    for _ in range(31):
        builder.scale("weight", 1)
    with pytest.raises(ValidationError):
        builder.scale("weight", 3)
    assert len(builder.build().operations) == 32
    snapshot = builder.build()
    snapshot.operations[0].values.append(99)
    assert not builder.build().operations[0].values


def test_tensor_artifacts_round_trip_dtype_shape_and_copy_isolation(tmp_path: Path) -> None:
    store = TensorArtifactStore(tmp_path)
    value = np.arange(12).reshape(3, 4)
    reference = store.put(value, dtype="float64")
    assert store.inspect(reference.id) == reference
    assert store.put(value, dtype="float64") == reference
    loaded = store.get(reference)
    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(loaded, value)
    loaded[0, 0] = 99
    np.testing.assert_array_equal(store.get(reference), value)
    assert store.put(value, dtype="float32").id != reference.id


def test_tensor_blob_and_metadata_tampering_and_missing_member_are_rejected(tmp_path: Path) -> None:
    store = TensorArtifactStore(tmp_path)
    reference = store.put([1, 2])
    blob = tmp_path / f"{reference.id}.bin"
    blob.write_bytes(b"x" * reference.byte_length)
    with pytest.raises(NrlyError, match="do not match"):
        store.inspect(reference.id)
    blob.unlink()
    with pytest.raises(NrlyError):
        store.get(reference)
    manifest = tmp_path / f"{reference.id}.json"
    data = json.loads(manifest.read_text())
    data["shape"] = [1, 2]
    manifest.write_text(json.dumps(data))
    with pytest.raises(NrlyError, match="manifest"):
        store.inspect(reference.id)


def test_artifacts_reject_traversal_oversize_objects_and_nonfinite(tmp_path: Path) -> None:
    store = TensorArtifactStore(tmp_path)
    with pytest.raises(NrlyError):
        store.inspect("../outside")
    with pytest.raises(NrlyError):
        read_local_file(tmp_path, "../outside", 100)
    for value in ([float("inf")], [1e308], np.array([object()], dtype=object)):
        with pytest.raises(NrlyError):
            store.put(value)
    with pytest.raises(NrlyError, match="16 MiB"):
        store.put(np.zeros(MAX_TENSOR_BYTES // 4 + 1, dtype=np.float32))


def test_forged_checksums_do_not_allow_nonfinite_tensor_data(tmp_path: Path) -> None:
    raw = np.array([np.nan], dtype="<f4").tobytes()
    content = {
        "schema_version": "nrly-tensor/1", "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": "float32", "shape": [1], "byte_length": 4,
    }
    reference = TensorArtifact(id=digest(content), **content)
    (tmp_path / f"{reference.id}.bin").write_bytes(raw)
    with pytest.raises(NrlyError, match="nonfinite"):
        TensorArtifactStore(tmp_path).get(reference)


def test_lineage_is_cumulative_immutable_and_rollback_selects_ancestor(tmp_path: Path) -> None:
    store = EditStore(tmp_path)
    first = store.create("A", WeightProgram("base").scale("weight", 2).build())
    second = store.create("B", WeightProgram("base").scale("weight", 3).build(), first.id)
    assert [item.id for item in store.ancestry(second.id)] == [first.id, second.id]
    assert [op.factor for op in second.program.operations] == [2, 3]
    assert store.select_ancestor(second.id, first.id) == first.program
    assert store.select_ancestor(second.id).kind == "identity"
    assert store.get(second.id) == second
    assert conflict_footprint(second).overlapping_targets == ["weight"]
    with pytest.raises(NrlyError, match="exact base"):
        store.create("wrong", WeightProgram("other").scale("weight", 1).build(), first.id)
    with pytest.raises(NrlyError, match="not an ancestor"):
        store.select_ancestor(first.id, second.id)


def test_lineage_verifies_prefix_even_if_attacker_rehashes_child(tmp_path: Path) -> None:
    store = EditStore(tmp_path)
    first = store.create("A", WeightProgram("base").scale("weight", 2).build())
    second = store.create("B", WeightProgram("base").scale("weight", 3).build(), first.id)
    content = second.model_dump(mode="json", exclude={"id"})
    content["program"]["operations"][0]["factor"] = 5.0
    new_id = digest(content)
    (tmp_path / f"{new_id}.json").write_text(canonical_json({"id": new_id, **content}))
    with pytest.raises(NrlyError, match="parent prefix"):
        store.get(new_id)
    first.program.operations.clear()
    with pytest.raises(ValidationError):
        conflict_footprint(first)


def test_disjoint_sparse_footprint_makes_no_behavioral_independence_claim(tmp_path: Path) -> None:
    store = EditStore(tmp_path)
    first = store.create("A", WeightProgram("base").patch("weight", [2], [0], [1]).build())
    second = store.create(
        "B", WeightProgram("base").patch("weight", [2], [1], [1]).build(), first.id,
    )
    footprint = conflict_footprint(second)
    assert footprint.disjoint_sparse_targets == ["weight"]
    assert not footprint.overlapping_targets
    assert "no_behavioral_independence" in footprint.semantics


def _run(backend: TinyBackend, program: ProgramSpec, identifier: str) -> RunRecord:
    cases = [
        CaseSpec(id="target", input="[2,4]", expected="6", scorer="numeric"),
        CaseSpec(id="regression", input="[0,0]", expected="0", role="regression"),
    ]
    spec = ExperimentSpec(name=identifier, program=program, cases=cases)
    samples = []
    for case in cases:
        result = backend.run_case(case, program, [], GenerationSpec())
        samples.append(SampleResult(
            case_id=case.id, status="completed", provenance="torch_fixture",
            output=result.output, weight_changes=result.weight_changes,
        ))
    return RunRecord(
        id=identifier, status="succeeded", spec=spec, spec_digest=digest(spec),
        created_at="2026-11-10T00:00:00Z", samples=samples, model_identity=backend.identity,
    )


def test_sequential_validation_detects_repair_b_destroying_a(tmp_path: Path) -> None:
    with TinyBackend(ModelSpec()) as backend:
        store = EditStore(tmp_path)
        base = backend.identity.fingerprint
        first = store.create("repair A", WeightProgram(base).scale("output.weight", 2).build())
        second = store.create(
            "repair B", WeightProgram(base).scale("output.weight", 0.5).build(), first.id,
        )
        original = _run(backend, ProgramSpec(), "original")
        parent = _run(backend, first.program, "parent")
        candidate = _run(backend, second.program, "candidate")
        report = compare_sequential(original, parent, candidate, first, second)
        assert report.comparable
        assert report.verdict == "observed_regression"
        assert report.lost_prior_improvements == ["target"]
        assert report.original_to_candidate.verdict == "no_change"
        failed = candidate.model_copy(update={"status": "failed"})
        report = compare_sequential(original, parent, failed, first, second)
        assert not report.comparable and report.verdict == "inconclusive"
        assert not report.lost_prior_improvements
        wrong_program = candidate.model_copy(update={"spec": parent.spec})
        assert not compare_sequential(original, parent, wrong_program, first, second).comparable


def test_patch_package_runs_inference_and_restores_on_success_and_exception(tmp_path: Path) -> None:
    with TinyBackend(ModelSpec()) as backend:
        before = tensor_digest(backend.model)
        program = WeightProgram(backend.identity.fingerprint).scale("output.weight", 2).build()
        manifest = export_patch(tmp_path / "patch", program, backend.identity)
        assert read_patch(tmp_path / "patch") == manifest
        with apply_patch_package(backend.model, backend.identity, tmp_path / "patch") as changes:
            assert backend.model(torch.tensor([[2.0, 4.0]])).item() == 6
            assert changes[0].changed_elements == 2
        assert tensor_digest(backend.model) == before
        with pytest.raises(RuntimeError, match="inference failure"):
            with apply_patch_package(backend.model, backend.identity, tmp_path / "patch"):
                raise RuntimeError("inference failure")
        assert tensor_digest(backend.model) == before
        with pytest.raises(NrlyError, match="absent or empty"):
            export_patch(tmp_path / "patch", program, backend.identity)


def test_patch_rejects_wrong_identity_weights_and_manifest(tmp_path: Path) -> None:
    with TinyBackend(ModelSpec()) as backend:
        program = WeightProgram(backend.identity.fingerprint).scale("output.weight", 2).build()
        export_patch(tmp_path, program, backend.identity)
        wrong = backend.identity.model_copy(update={"fingerprint": "other"})
        with pytest.raises(NrlyError, match="identity differs"):
            with apply_patch_package(backend.model, wrong, tmp_path):
                pytest.fail("wrong identity admitted")
        with torch.no_grad():
            backend.model.output.weight.fill_(7)
        with pytest.raises(NrlyError, match="weights differ"):
            with apply_patch_package(backend.model, backend.identity, tmp_path):
                pytest.fail("wrong weight bytes admitted")
        manifest_file = tmp_path / "manifest.json"
        content = json.loads(manifest_file.read_text())
        content["program"]["operations"][0]["factor"] = 99
        manifest_file.write_text(json.dumps(content))
        with pytest.raises(NrlyError, match="manifest"):
            read_patch(tmp_path)
