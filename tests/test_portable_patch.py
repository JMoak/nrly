from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from torch import nn

import nrly
from nrly.artifacts import MAX_MANIFEST_BYTES
from nrly.backends.tiny import TinyBackend
from nrly.contracts import (
    CaseSpec,
    GenerationSpec,
    ModelIdentity,
    ModelSpec,
    NrlyError,
    PortableBase,
    ProgramSpec,
    WeightOperationSpec,
    canonical_json,
    digest,
)
from nrly.deployment import (
    PatchManifest,
    PortablePatchManifest,
    apply_patch_package,
    bind_operations,
    export_patch,
    export_portable_patch,
    read_any_patch,
    read_patch,
    read_portable_patch,
)
from nrly.kernel import intervene, tensor_digest

REPAIR = (WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2),)
ANY_BASE = PortableBase(
    backend="tiny", model_id="nrly/tiny-sum-v1", revision="1", dtype="torch.float32",
    weights_sha256="a" * 64,
)


def runtime_fingerprint(identity: ModelIdentity) -> str:
    """The fingerprint TorchBackend derives from a model id, revision and runtime details."""
    manifest = {
        "model_id": identity.model_id, "revision": identity.revision, "details": identity.details,
    }
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def on_another_machine(identity: ModelIdentity) -> ModelIdentity:
    """The same checkpoint as a runtime with another GPU and library build records it."""
    details = {**identity.details, "torch": "9.9.9+elsewhere", "device_name": "Elsewhere GPU"}
    moved = identity.model_copy(update={"details": details})
    return moved.model_copy(update={"fingerprint": runtime_fingerprint(moved)})


def state_bytes(model: nn.Module) -> dict[str, bytes]:
    return {
        name: value.detach().cpu().numpy().tobytes() for name, value in model.state_dict().items()
    }


def tiny_sum(backend: TinyBackend) -> float:
    return float(backend.model(torch.tensor([[2.0, 4.0]])).item())


def rewrite_manifest(
    package: Path, pointer: tuple[str | int, ...], value: Any, *, rehash: bool,
) -> None:
    path = package / "manifest.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    parent = content
    for key in pointer[:-1]:
        parent = parent[key]
    parent[pointer[-1]] = value
    if rehash:
        del content["id"]
        content["id"] = digest(content)
    path.write_text(canonical_json(content), encoding="utf-8")


def test_portable_package_round_trips_and_pins_weights_not_a_runtime(tmp_path: Path) -> None:
    with TinyBackend(ModelSpec()) as backend:
        here = backend.identity
    # The simulated machine is only meaningful while this mirrors the backend's own digest.
    assert runtime_fingerprint(here) == here.fingerprint
    elsewhere = on_another_machine(here)
    assert elsewhere.fingerprint != here.fingerprint

    base = PortableBase.from_identity(here)
    manifest = export_portable_patch(tmp_path / "patch", REPAIR, base)
    assert type(manifest) is PortablePatchManifest
    assert manifest.schema_version == "nrly-patch/2"
    assert manifest.base == base and manifest.operations == list(REPAIR)
    assert manifest.base.weights_sha256 == here.details["weights_sha256"]
    assert manifest.id == digest(manifest.model_dump(mode="json", exclude={"id"}))

    raw = (tmp_path / "patch" / "manifest.json").read_bytes()
    assert raw == canonical_json(manifest).encode("utf-8")
    stored = json.loads(raw)
    assert set(stored) == {"schema_version", "id", "base", "operations"}
    assert set(stored["base"]) == {"backend", "model_id", "revision", "dtype", "weights_sha256"}
    assert here.fingerprint.encode() not in raw
    assert str(here.details["torch"]).encode() not in raw
    assert read_portable_patch(tmp_path / "patch") == manifest
    assert read_any_patch(tmp_path / "patch") == manifest
    assert type(read_any_patch(tmp_path / "patch")) is PortablePatchManifest

    # The same edit packaged from another runtime's record is the same package, byte for byte.
    moved = export_portable_patch(tmp_path / "moved", REPAIR, PortableBase.from_identity(elsewhere))
    assert moved.id == manifest.id
    assert (tmp_path / "moved" / "manifest.json").read_bytes() == raw

    with pytest.raises(NrlyError, match="absent or empty") as occupied:
        export_portable_patch(tmp_path / "patch", [REPAIR[0], REPAIR[0]], base)
    assert occupied.value.code == "destination_exists"
    assert (tmp_path / "patch" / "manifest.json").read_bytes() == raw


def test_portable_export_enforces_program_bounds_before_writing(tmp_path: Path) -> None:
    def addition(count: int) -> WeightOperationSpec:
        return WeightOperationSpec(
            kind="weight_add", target="output.weight", shape=[count], values=[0.0] * count,
        )

    invalid = {
        "none": [],
        "too-many": [REPAIR[0]] * 33,
        "payload": [addition(32768), addition(32769)],
    }
    for name, operations in invalid.items():
        with pytest.raises(ValidationError):
            export_portable_patch(tmp_path / name, operations, ANY_BASE)
        assert not (tmp_path / name).exists()
    assert len(export_portable_patch(tmp_path / "32", [REPAIR[0]] * 32, ANY_BASE).operations) == 32
    full = export_portable_patch(tmp_path / "full", [addition(32768), addition(32768)], ANY_BASE)
    assert read_portable_patch(tmp_path / "full") == full

    occupied = tmp_path / "file"
    occupied.write_text("not a package", encoding="utf-8")
    with pytest.raises(NrlyError) as refused:
        export_portable_patch(occupied, REPAIR, ANY_BASE)
    assert refused.value.code == "destination_exists"
    assert occupied.read_text(encoding="utf-8") == "not a package"


@pytest.mark.parametrize(
    ("pointer", "value", "rehash"),
    [
        pytest.param(("operations", 0, "factor"), 99.0, False, id="edited-operation"),
        pytest.param(("base", "weights_sha256"), "0" * 64, False, id="edited-base-weights"),
        pytest.param(("base", "dtype"), "torch.float16", False, id="edited-base-dtype"),
        pytest.param(("id",), "0" * 64, False, id="forged-id"),
        pytest.param(("schema_version",), "nrly-patch/1", True, id="relabelled-runtime-pinned"),
        pytest.param(("schema_version",), "nrly-patch/3", True, id="unknown-version"),
        pytest.param(("fingerprint",), "f" * 64, True, id="rehashed-runtime-fingerprint"),
        pytest.param(("base", "fingerprint"), "f" * 64, True, id="rehashed-base-fingerprint"),
        pytest.param(("operations",), [], True, id="rehashed-without-operations"),
        pytest.param(("operations", 0, "kind"), "activation_scale", True, id="rehashed-activation"),
        pytest.param(("base", "weights_sha256"), "unknown", True, id="rehashed-without-digest"),
    ],
)
def test_tampered_portable_manifests_are_rejected_by_readers_and_application(
    tmp_path: Path, pointer: tuple[str | int, ...], value: Any, rehash: bool,
) -> None:
    with TinyBackend(ModelSpec()) as backend:
        export_portable_patch(tmp_path, REPAIR, PortableBase.from_identity(backend.identity))
        rewrite_manifest(tmp_path, pointer, value, rehash=rehash)
        for reader in (read_portable_patch, read_any_patch):
            with pytest.raises(NrlyError, match="manifest") as rejected:
                reader(tmp_path)
            assert rejected.value.code == "invalid_artifact"
        before = state_bytes(backend.model)
        with pytest.raises(NrlyError) as refused:
            with apply_patch_package(backend.model, backend.identity, tmp_path):
                pytest.fail("a tampered package was applied")
        assert refused.value.code == "invalid_artifact"
        assert state_bytes(backend.model) == before


def test_rehashed_content_is_a_different_package_not_the_original(tmp_path: Path) -> None:
    # Control for the rejections above: the rewrite helper produces well-formed manifests,
    # and a checksum identifies content without vouching for who published it.
    original = export_portable_patch(tmp_path, REPAIR, ANY_BASE)
    rewrite_manifest(tmp_path, ("operations", 0, "factor"), 99.0, rehash=True)
    changed = read_any_patch(tmp_path)
    assert type(changed) is PortablePatchManifest
    assert changed.operations[0].factor == 99 and changed.id != original.id


@pytest.mark.parametrize(
    "content",
    [
        None,
        b"",
        b"not json",
        b"[]",
        b'"nrly-patch/2"',
        b'{"schema_version": "nrly-patch/2"}',
        b'{"schema_version": 2}',
        b"[" * 100000,
        b" " * (MAX_MANIFEST_BYTES + 1),
    ],
    ids=["absent", "empty", "text", "array", "string", "header-only", "numeric-version", "deep",
         "oversize"],
)
def test_malformed_manifests_are_invalid_artifacts_for_every_reader(
    tmp_path: Path, content: bytes | None,
) -> None:
    if content is not None:
        (tmp_path / "manifest.json").write_bytes(content)
    for reader in (read_patch, read_portable_patch, read_any_patch):
        with pytest.raises(NrlyError) as rejected:
            reader(tmp_path)
        assert rejected.value.code == "invalid_artifact"


def test_same_weights_on_another_machine_are_refused_by_v1_and_accepted_by_portable(
    tmp_path: Path,
) -> None:
    with TinyBackend(ModelSpec()) as backend:
        here = backend.identity
        program = ProgramSpec(
            kind="pipeline", expected_base=here.fingerprint, operations=list(REPAIR),
        )
        export_patch(tmp_path / "pinned", program, here)
        export_portable_patch(tmp_path / "portable", REPAIR, PortableBase.from_identity(here))
        base_digest = tensor_digest(backend.model)
        before = state_bytes(backend.model)
        assert tiny_sum(backend) == 3

        # Control: the runtime-pinned package is valid for the runtime that produced it.
        with apply_patch_package(backend.model, here, tmp_path / "pinned"):
            assert tiny_sum(backend) == 6
        assert state_bytes(backend.model) == before

        # The very same loaded weights, as a machine with another GPU and Torch records them.
        elsewhere = on_another_machine(here)
        assert elsewhere.fingerprint != here.fingerprint
        assert elsewhere.details["weights_sha256"] == here.details["weights_sha256"]
        with pytest.raises(NrlyError, match="identity differs") as refused:
            with apply_patch_package(backend.model, elsewhere, tmp_path / "pinned"):
                pytest.fail("a runtime-pinned package was admitted on another machine")
        assert refused.value.code == "base_mismatch"
        assert state_bytes(backend.model) == before

        with apply_patch_package(backend.model, elsewhere, tmp_path / "portable") as changes:
            assert tiny_sum(backend) == 6
            assert tensor_digest(backend.model) != base_digest
            assert [change.target for change in changes] == ["output.weight"]
            assert changes[0].changed_elements == 2
            assert changes[0].net_after_sha256 == changes[0].after_sha256
        assert state_bytes(backend.model) == before
        assert tensor_digest(backend.model) == base_digest
        assert tiny_sum(backend) == 3
        backend.verify_base_unchanged()


def test_portable_package_refuses_weights_that_differ_from_its_base(tmp_path: Path) -> None:
    with TinyBackend(ModelSpec()) as backend:
        recorded = backend.identity
        export_portable_patch(tmp_path, REPAIR, PortableBase.from_identity(recorded))
        with torch.no_grad():
            backend.model.hidden.weight[0, 0] = 1.5
        drifted = state_bytes(backend.model)

        # The stale identity still claims the base digest; only rehashing the tensors sees it.
        with pytest.raises(NrlyError, match="weights differ") as stale:
            with apply_patch_package(backend.model, recorded, tmp_path):
                pytest.fail("a package was applied to weights it was not written for")
        assert stale.value.code == "base_mismatch"
        assert state_bytes(backend.model) == drifted

        details = {**recorded.details, "weights_sha256": tensor_digest(backend.model)}
        honest = recorded.model_copy(update={"details": details})
        honest = honest.model_copy(update={"fingerprint": runtime_fingerprint(honest)})
        with pytest.raises(NrlyError, match="base in weights_sha256$") as declared:
            with apply_patch_package(backend.model, honest, tmp_path):
                pytest.fail("a package was applied to weights it was not written for")
        assert declared.value.code == "base_mismatch"
        assert state_bytes(backend.model) == drifted


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", "torch.float16"),
        ("revision", "2"),
        ("model_id", "nrly/another-model"),
        ("backend", "huggingface"),
    ],
)
def test_portable_package_refuses_another_dtype_revision_or_model(
    tmp_path: Path, field: str, value: str,
) -> None:
    with TinyBackend(ModelSpec()) as backend:
        base = PortableBase.from_identity(backend.identity)
        export_portable_patch(tmp_path, REPAIR, base.model_copy(update={field: value}))
        before = state_bytes(backend.model)
        with pytest.raises(NrlyError, match=f"base in {field}$") as refused:
            with apply_patch_package(backend.model, backend.identity, tmp_path):
                pytest.fail("a package for another base was applied")
        assert refused.value.code == "base_mismatch"
        assert state_bytes(backend.model) == before


def test_portable_package_written_at_float32_is_refused_by_a_float16_load(tmp_path: Path) -> None:
    with TinyBackend(ModelSpec()) as full, TinyBackend(ModelSpec(dtype="float16")) as half:
        export_portable_patch(tmp_path / "full", REPAIR, PortableBase.from_identity(full.identity))
        before = state_bytes(half.model)
        with pytest.raises(NrlyError, match="base in dtype, weights_sha256$") as refused:
            with apply_patch_package(half.model, half.identity, tmp_path / "full"):
                pytest.fail("a float32 package was applied to float16 weights")
        assert refused.value.code == "base_mismatch"
        assert state_bytes(half.model) == before

        export_portable_patch(tmp_path / "half", REPAIR, PortableBase.from_identity(half.identity))
        with apply_patch_package(half.model, half.identity, tmp_path / "half") as changes:
            assert changes[0].dtype == "torch.float16" and changes[0].changed_elements == 2
        assert state_bytes(half.model) == before


def test_portable_application_restores_when_the_body_raises_and_rejects_foreign_writes(
    tmp_path: Path,
) -> None:
    with TinyBackend(ModelSpec()) as backend:
        elsewhere = on_another_machine(backend.identity)
        export_portable_patch(tmp_path, REPAIR, PortableBase.from_identity(backend.identity))
        before = state_bytes(backend.model)
        with pytest.raises(RuntimeError, match="inference failure"):
            with apply_patch_package(backend.model, elsewhere, tmp_path):
                assert state_bytes(backend.model) != before
                raise RuntimeError("inference failure")
        assert state_bytes(backend.model) == before
        backend.verify_base_unchanged()

        # The kernel restores what the patch touched; the digest after restoration must
        # still expose a write the patch never made.
        with pytest.raises(NrlyError, match="after patch restoration") as failed:
            with apply_patch_package(backend.model, elsewhere, tmp_path):
                with torch.no_grad():
                    backend.model.hidden.weight.add_(1)
        assert failed.value.code == "restoration_failed"


def test_bound_operations_run_through_the_real_kernel_on_exactly_one_runtime() -> None:
    operations = [
        WeightOperationSpec(kind="weight_scale", target="output.weight", factor=4),
        WeightOperationSpec(
            kind="weight_sparse_add", target="output.weight", shape=[1, 2], indices=[0, 1],
            values=[-1.0, -1.0],
        ),
    ]
    case = CaseSpec(id="target", input="[2,4]", expected="6", scorer="numeric")
    with TinyBackend(ModelSpec()) as backend:
        here = backend.identity
        elsewhere = on_another_machine(here)
        base = PortableBase.from_identity(here)
        program = bind_operations(operations, base, elsewhere)
        assert program.kind == "pipeline"
        assert program.expected_base == elsewhere.fingerprint
        assert program.operations == operations

        before = state_bytes(backend.model)
        with intervene(backend.model, program, elsewhere.fingerprint) as changes:
            assert tiny_sum(backend) == 6
            assert [change.operation_index for change in changes] == [0, 1]
            assert [change.operation_kind for change in changes] == [
                "weight_scale", "weight_sparse_add",
            ]
        assert state_bytes(backend.model) == before

        with pytest.raises(NrlyError, match="expected_base") as other_runtime:
            with intervene(backend.model, program, here.fingerprint):
                pytest.fail("a bound program ran on a runtime it was not bound to")
        assert other_runtime.value.code == "base_mismatch"

        local = bind_operations(operations, base, here)
        assert local.expected_base == here.fingerprint
        result = backend.run_case(case, local, [], GenerationSpec())
        assert result.output == "6" and result.base_unchanged
        assert result.weight_changes[-1].net_changed_elements == 2
        assert backend.run_case(case, ProgramSpec(), [], GenerationSpec()).output == "3"


def test_binding_refuses_mismatched_bases_and_unusable_inputs() -> None:
    with TinyBackend(ModelSpec()) as backend:
        here = backend.identity
    base = PortableBase.from_identity(here)
    for field, value in (
        ("weights_sha256", "0" * 64), ("dtype", "torch.bfloat16"), ("revision", "2"),
    ):
        with pytest.raises(NrlyError, match=f"base in {field}$") as refused:
            bind_operations(REPAIR, base.model_copy(update={field: value}), here)
        assert refused.value.code == "base_mismatch"

    undescribed = here.model_copy(update={"details": {}})
    with pytest.raises(NrlyError, match="base in dtype, weights_sha256$") as unverifiable:
        bind_operations(REPAIR, base, undescribed)
    assert unverifiable.value.code == "base_mismatch"

    with pytest.raises(NrlyError, match="fingerprint") as unpinned:
        bind_operations(REPAIR, base, here.model_copy(update={"fingerprint": ""}))
    assert unpinned.value.code == "invalid_identity"

    for operations in ([], [REPAIR[0]] * 33):
        with pytest.raises(ValidationError):
            bind_operations(operations, base, here)


def test_runtime_pinned_packages_still_read_and_apply_as_before(tmp_path: Path) -> None:
    with TinyBackend(ModelSpec()) as backend:
        here = backend.identity
        program = ProgramSpec(
            kind="weight_scale", target="output.weight", factor=2, expected_base=here.fingerprint,
        )
        manifest = export_patch(tmp_path / "pinned", program, here)
        assert type(manifest) is PatchManifest
        assert manifest.schema_version == "nrly-patch/1"
        assert manifest.identity == here and manifest.program == program
        assert manifest.id == digest(manifest.model_dump(mode="json", exclude={"id"}))
        file = tmp_path / "pinned" / "manifest.json"
        raw = file.read_bytes()
        assert raw == canonical_json(manifest).encode("utf-8")
        assert set(json.loads(raw)) == {"schema_version", "id", "identity", "program", "revision"}
        assert read_patch(tmp_path / "pinned") == manifest
        assert read_any_patch(tmp_path / "pinned") == manifest
        assert type(read_any_patch(tmp_path / "pinned")) is PatchManifest

        # Each strict reader refuses the other format instead of reinterpreting it.
        export_portable_patch(tmp_path / "portable", REPAIR, PortableBase.from_identity(here))
        for reader, package in ((read_portable_patch, "pinned"), (read_patch, "portable")):
            with pytest.raises(NrlyError, match="manifest") as wrong_format:
                reader(tmp_path / package)
            assert wrong_format.value.code == "invalid_artifact"

        before = state_bytes(backend.model)
        with apply_patch_package(backend.model, here, tmp_path / "pinned") as changes:
            assert tiny_sum(backend) == 6
            assert changes[0].changed_elements == 2
            # The recorded single operation executes as written, not as a bound pipeline.
            assert changes[0].operation_index is None
        assert state_bytes(backend.model) == before
        with pytest.raises(RuntimeError, match="inference failure"):
            with apply_patch_package(backend.model, here, tmp_path / "pinned"):
                raise RuntimeError("inference failure")
        assert state_bytes(backend.model) == before

        # A manifest that declares no version was always read as nrly-patch/1; it still is.
        undeclared = json.loads(raw)
        del undeclared["schema_version"]
        file.write_text(json.dumps(undeclared), encoding="utf-8")
        assert read_patch(tmp_path / "pinned") == manifest
        assert read_any_patch(tmp_path / "pinned") == manifest


def invoke(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(nrly.__file__).resolve().parents[1])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "nrly.cli", "--root", str(tmp_path / "runs"), *arguments],
        cwd=tmp_path, env=environment, capture_output=True, text=True, encoding="utf-8",
        timeout=60, check=False,
    )


def success(tmp_path: Path, *arguments: str) -> Any:
    result = invoke(tmp_path, *arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def refusal(tmp_path: Path, *arguments: str) -> dict[str, str]:
    result = invoke(tmp_path, *arguments)
    assert result.returncode == 2 and result.stdout == "", result.stdout + result.stderr
    error: dict[str, str] = json.loads(result.stderr)["error"]
    return error


def write_json(tmp_path: Path, name: str, content: Any) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def experiment(program: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": "CLI tiny portable patch acceptance",
        "program": program or {"kind": "identity"},
        "cases": [
            {"id": "target", "input": "[2,4]", "expected": "6", "scorer": "numeric"},
            {"id": "regression", "input": "[0,0]", "expected": "0", "role": "regression"},
        ],
    }


def added_runs(tmp_path: Path, before: list[dict[str, Any]]) -> list[tuple[str, str]]:
    known = {run["id"] for run in before}
    return [
        (run["purpose"], run["status"])
        for run in success(tmp_path, "list")
        if run["id"] not in known
    ]


def test_actual_cli_portable_export_inspect_and_run_bind_to_the_local_runtime(
    tmp_path: Path,
) -> None:
    repaired_spec = write_json(tmp_path, "repaired", experiment({
        "kind": "weight_scale", "target": "output.weight", "factor": 2,
    }))
    initial = success(tmp_path, "run", str(repaired_spec))["run"]
    identity = initial["model_identity"]
    assert initial["status"] == "succeeded" and initial["samples"][0]["output"] == "6"

    package = tmp_path / "portable"
    manifest = success(tmp_path, "patch", "export", initial["id"], str(package), "--portable")
    assert set(manifest) == {"schema_version", "id", "base", "operations"}
    assert manifest["schema_version"] == "nrly-patch/2"
    assert manifest["base"] == {
        "backend": "tiny", "model_id": "nrly/tiny-sum-v1", "revision": "1",
        "dtype": identity["details"]["dtype"],
        "weights_sha256": identity["details"]["weights_sha256"],
    }
    assert [(op["kind"], op["target"], op["factor"]) for op in manifest["operations"]] == [
        ("weight_scale", "output.weight", 2.0),
    ]
    package_bytes = (package / "manifest.json").read_bytes()
    assert identity["fingerprint"].encode() not in package_bytes
    assert success(tmp_path, "patch", "inspect", str(package)) == manifest

    base_spec = write_json(tmp_path, "base", experiment())
    replay = success(tmp_path, "patch", "run", str(package), str(base_spec))
    assert set(replay) == {"patch_id", "bound_fingerprint", "inspection_run", "run", "evaluation"}
    assert replay["patch_id"] == manifest["id"]
    assert replay["bound_fingerprint"] == identity["fingerprint"]
    run = replay["run"]
    assert run["status"] == "succeeded" and run["model_identity"] == identity
    assert run["spec"]["program"]["kind"] == "pipeline"
    assert run["spec"]["program"]["expected_base"] == replay["bound_fingerprint"]
    assert [sample["output"] for sample in run["samples"]] == ["6", "0"]
    assert replay["evaluation"]["summary"]["passed"] == 2
    for key in ("before_sha256", "after_sha256", "changed_elements"):
        assert (
            run["samples"][0]["weight_changes"][0][key]
            == initial["samples"][0]["weight_changes"][0][key]
        )
    inspection = success(tmp_path, "show", replay["inspection_run"])
    assert inspection["spec"]["purpose"] == "inspection"
    assert inspection["status"] == "succeeded" and not inspection["samples"]
    assert inspection["model_identity"] == identity

    occupied = refusal(tmp_path, "patch", "export", initial["id"], str(package), "--portable")
    assert occupied["code"] == "destination_exists"
    assert (package / "manifest.json").read_bytes() == package_bytes

    # Without the flag the same commands still produce and execute a runtime-pinned package.
    pinned = tmp_path / "pinned"
    pinned_manifest = success(tmp_path, "patch", "export", initial["id"], str(pinned))
    assert pinned_manifest["schema_version"] == "nrly-patch/1"
    assert pinned_manifest["identity"] == identity
    assert pinned_manifest["program"]["expected_base"] == identity["fingerprint"]
    assert success(tmp_path, "patch", "inspect", str(pinned)) == pinned_manifest
    before = success(tmp_path, "list")
    pinned_replay = success(tmp_path, "patch", "run", str(pinned), str(base_spec))
    assert set(pinned_replay) == {"patch_id", "run", "evaluation"}
    assert pinned_replay["patch_id"] == pinned_manifest["id"]
    assert pinned_replay["run"]["spec"]["program"] == pinned_manifest["program"]
    assert pinned_replay["run"]["samples"][0]["output"] == "6"
    assert added_runs(tmp_path, before) == [("experiment", "succeeded")]


def test_actual_cli_refuses_portable_export_of_identity_and_activation_runs(
    tmp_path: Path,
) -> None:
    programs = {
        "identity": {"kind": "identity"},
        "activation": {"kind": "activation_scale", "target": "relu", "factor": 2},
    }
    for name, program in programs.items():
        spec = write_json(tmp_path, name, experiment(program))
        run = success(tmp_path, "run", str(spec))["run"]
        assert run["status"] == "succeeded" and len(run["samples"]) == 2
        destination = tmp_path / f"{name}-patch"
        error = refusal(tmp_path, "patch", "export", run["id"], str(destination), "--portable")
        assert error["code"] == "invalid_program"
        assert not destination.exists()


def test_actual_cli_runs_a_patch_from_another_machine_only_when_it_is_portable(
    tmp_path: Path,
) -> None:
    local = ModelIdentity.model_validate(success(tmp_path, "inspect")["model_identity"])
    assert runtime_fingerprint(local) == local.fingerprint
    elsewhere = on_another_machine(local)
    program = ProgramSpec(
        kind="pipeline", expected_base=elsewhere.fingerprint, operations=list(REPAIR),
    )
    pinned = export_patch(tmp_path / "pinned", program, elsewhere)
    portable = export_portable_patch(
        tmp_path / "portable", REPAIR, PortableBase.from_identity(elsewhere),
    )
    spec = write_json(tmp_path, "base", experiment())

    rejected = invoke(tmp_path, "patch", "run", str(tmp_path / "pinned"), str(spec))
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    failed = json.loads(rejected.stdout)
    assert failed["patch_id"] == pinned.id
    assert failed["run"]["status"] == "failed"
    assert failed["run"]["error"]["code"] == "base_mismatch"
    assert not failed["run"]["samples"]
    assert failed["evaluation"]["summary"]["passed"] == 0

    accepted = success(tmp_path, "patch", "run", str(tmp_path / "portable"), str(spec))
    assert accepted["patch_id"] == portable.id
    assert accepted["bound_fingerprint"] == local.fingerprint
    assert accepted["bound_fingerprint"] != elsewhere.fingerprint
    assert accepted["run"]["status"] == "succeeded"
    assert accepted["run"]["model_identity"]["fingerprint"] == local.fingerprint
    assert [sample["output"] for sample in accepted["run"]["samples"]] == ["6", "0"]
    assert accepted["evaluation"]["summary"]["passed"] == 2


def test_actual_cli_patch_run_refuses_a_portable_base_the_local_model_does_not_hold(
    tmp_path: Path,
) -> None:
    local = ModelIdentity.model_validate(success(tmp_path, "inspect")["model_identity"])
    base = PortableBase.from_identity(local)
    spec = write_json(tmp_path, "base", experiment())

    # Learned only by loading the model: the refusal follows a recorded inspection.
    for field, value in (("dtype", "torch.float16"), ("weights_sha256", "0" * 64)):
        export_portable_patch(tmp_path / field, REPAIR, base.model_copy(update={field: value}))
        before = success(tmp_path, "list")
        error = refusal(tmp_path, "patch", "run", str(tmp_path / field), str(spec))
        assert error["code"] == "base_mismatch"
        assert error["message"].endswith(f"base in {field}")
        assert added_runs(tmp_path, before) == [("inspection", "succeeded")]

    # Visible from the request alone: refused before any model is loaded.
    export_portable_patch(tmp_path / "revision", REPAIR, base.model_copy(update={"revision": "2"}))
    before = success(tmp_path, "list")
    error = refusal(tmp_path, "patch", "run", str(tmp_path / "revision"), str(spec))
    assert error["code"] == "base_mismatch"
    assert added_runs(tmp_path, before) == []

    # An inspection that cannot finish binds nothing and starts no experiment.
    export_portable_patch(tmp_path / "matching", REPAIR, base)
    starved = experiment()
    starved["budget"] = {"wall_seconds": 0.001}
    error = refusal(
        tmp_path, "patch", "run", str(tmp_path / "matching"),
        str(write_json(tmp_path, "starved", starved)),
    )
    assert error["code"] == "budget_exceeded"
    assert "did not establish the local model identity" in error["message"]
    assert added_runs(tmp_path, before) == [("inspection", "failed")]
