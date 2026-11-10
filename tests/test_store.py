from __future__ import annotations

import json
from pathlib import Path

import pytest

from nrly.contracts import CaseSpec, ExperimentSpec, NrlyError, digest
from nrly.store import Store, read_bundle


def specification(name: str = "store fixture") -> ExperimentSpec:
    return ExperimentSpec(name=name, cases=[CaseSpec(id="sum", input="[2,4]", expected="6")])


def terminal(store: Store) -> str:
    record, _ = store.create(specification())
    store.save(record.model_copy(update={"status": "cancelled"}))
    return record.id


def test_idempotent_submission_and_conflicting_request(tmp_path: Path) -> None:
    store = Store(tmp_path)
    first, created = store.create(specification(), request_id="fixture-request")
    same, repeated = store.create(specification(), request_id="fixture-request")
    assert created and not repeated
    assert first == same
    assert len(store.list()) == 1
    with pytest.raises(NrlyError, match="already names another spec") as error:
        store.create(specification("changed"), request_id="fixture-request")
    assert error.value.code == "request_conflict"


@pytest.mark.parametrize("handle", ["../outside", "A" * 32, "a" * 31, "a/b", ""])
def test_run_handles_cannot_escape_store(tmp_path: Path, handle: str) -> None:
    with pytest.raises(NrlyError) as error:
        Store(tmp_path).get(handle)
    assert error.value.code == "invalid_run_id"


def test_unknown_valid_handle_has_typed_error(tmp_path: Path) -> None:
    with pytest.raises(NrlyError) as error:
        Store(tmp_path).get("a" * 32)
    assert error.value.code == "run_not_found"


def test_experiment_and_lineage_are_immutable(tmp_path: Path) -> None:
    store = Store(tmp_path)
    record, _ = store.create(specification())
    changed = specification("changed")
    with pytest.raises(NrlyError) as error:
        store.save(record.model_copy(update={"spec": changed, "spec_digest": digest(changed)}))
    assert error.value.code == "spec_mismatch"
    with pytest.raises(NrlyError) as error:
        store.save(record.model_copy(update={"parent_run": "a" * 32}))
    assert error.value.code == "identity_mismatch"


def test_terminal_records_cannot_be_rewritten(tmp_path: Path) -> None:
    store = Store(tmp_path)
    run_id = terminal(store)
    record = store.get(run_id)
    store.save(record)
    with pytest.raises(NrlyError) as error:
        store.save(record.model_copy(update={"status": "succeeded"}))
    assert error.value.code == "immutable_run"


def test_export_roundtrip_is_checksummed_and_complete(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    run_id = terminal(store)
    destination = store.export(run_id, tmp_path / "export")
    assert read_bundle(destination) == store.get(run_id)
    assert {path.name for path in destination.iterdir()} == {
        "run.json",
        "spec.json",
        "manifest.json",
    }


@pytest.mark.parametrize("filename", ["run.json", "spec.json", "manifest.json"])
def test_bundle_tampering_is_rejected(tmp_path: Path, filename: str) -> None:
    store = Store(tmp_path / "workspace")
    run_id = terminal(store)
    destination = store.export(run_id, tmp_path / "export")
    path = destination / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    if filename == "manifest.json":
        data["run_sha256"] = "0" * 64
    elif filename == "spec.json":
        data["name"] = "altered"
    else:
        data["status"] = "succeeded"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(NrlyError) as error:
        read_bundle(destination)
    assert error.value.code == "artifact_mismatch"


def test_nonobject_manifest_is_a_typed_bundle_error(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    run_id = terminal(store)
    destination = store.export(run_id, tmp_path / "export")
    (destination / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(NrlyError) as error:
        read_bundle(destination)
    assert error.value.code == "invalid_bundle"


def test_missing_bundle_member_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(NrlyError) as error:
        read_bundle(tmp_path)
    assert error.value.code == "invalid_bundle"


def test_export_rejects_existing_file_and_nonempty_directory(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    run_id = terminal(store)
    occupied = tmp_path / "occupied"
    occupied.write_text("preserve me", encoding="utf-8")
    for destination in (occupied, store.run_dir(run_id)):
        with pytest.raises(NrlyError) as error:
            store.export(run_id, destination)
        assert error.value.code == "destination_exists"
    assert occupied.read_text(encoding="utf-8") == "preserve me"


def test_active_run_cannot_be_exported(tmp_path: Path) -> None:
    store = Store(tmp_path / "workspace")
    record, _ = store.create(specification())
    with pytest.raises(NrlyError) as error:
        store.export(record.id, tmp_path / "export")
    assert error.value.code == "run_incomplete"


def test_unfinished_query_is_not_limited_by_history_page(tmp_path: Path) -> None:
    store = Store(tmp_path)
    old, _ = store.create(specification("old active"))
    terminal(store)
    assert store.list(1)[0].status == "cancelled"
    assert [record.id for record in store.unfinished()] == [old.id]
