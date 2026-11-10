from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import nrly


def invoke(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(nrly.__file__).resolve().parents[1])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "nrly.cli", "--root", str(tmp_path / "runs"), *arguments],
        cwd=tmp_path, env=environment, capture_output=True, text=True, encoding="utf-8",
        timeout=30, check=False,
    )


def success(tmp_path: Path, *arguments: str) -> Any:
    result = invoke(tmp_path, *arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def write_json(tmp_path: Path, name: str, content: Any) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def experiment(program: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": "CLI tiny authoring acceptance",
        "program": program or {"kind": "identity"},
        "cases": [
            {"id": "target", "input": "[2,4]", "expected": "6", "scorer": "numeric"},
            {"id": "regression", "input": "[0,0]", "expected": "0", "role": "regression"},
        ],
    }


def test_actual_cli_patch_export_inspect_run_and_machine_readable_failures(tmp_path: Path) -> None:
    repaired_spec = write_json(tmp_path, "repaired", experiment({
        "kind": "weight_scale", "target": "output.weight", "factor": 2,
    }))
    initial = success(tmp_path, "run", str(repaired_spec))
    assert initial["run"]["status"] == "succeeded"
    assert initial["run"]["samples"][0]["output"] == "6"
    assert initial["evaluation"]["scope"] == "engineering_fixture"
    assert initial["evaluation"]["summary"]["passed"] == 2

    package = tmp_path / "patch"
    manifest = success(tmp_path, "patch", "export", initial["run"]["id"], str(package))
    verified = success(tmp_path, "patch", "inspect", str(package))
    assert verified == manifest
    assert manifest["program"]["expected_base"] == initial["run"]["model_identity"]["fingerprint"]
    package_bytes = (package / "manifest.json").read_bytes()
    base_spec = write_json(tmp_path, "base", experiment())
    replay = success(tmp_path, "patch", "run", str(package), str(base_spec))
    assert replay["patch_id"] == manifest["id"]
    assert replay["run"]["model_identity"] == initial["run"]["model_identity"]
    assert replay["run"]["samples"][0]["output"] == "6"
    assert (
        replay["run"]["samples"][0]["weight_changes"]
        == initial["run"]["samples"][0]["weight_changes"]
    )
    assert success(tmp_path, "run", str(base_spec))["run"]["samples"][0]["output"] == "3"

    existing = invoke(tmp_path, "patch", "export", initial["run"]["id"], str(package))
    assert existing.returncode == 2 and existing.stdout == ""
    assert json.loads(existing.stderr)["error"]["code"] == "destination_exists"
    assert (package / "manifest.json").read_bytes() == package_bytes

    mismatch = experiment()
    mismatch["model"] = {
        "backend": "huggingface", "model_id": "different/model", "revision": "a" * 40,
    }
    mismatch_spec = write_json(tmp_path, "wrong-model", mismatch)
    before_runs = success(tmp_path, "list")
    refused = invoke(tmp_path, "patch", "run", str(package), str(mismatch_spec))
    assert refused.returncode == 2 and refused.stdout == ""
    assert json.loads(refused.stderr)["error"]["code"] == "base_mismatch"
    assert success(tmp_path, "list") == before_runs
    assert (package / "manifest.json").read_bytes() == package_bytes


def test_actual_cli_sequential_a_then_harmful_b_reports_interference(tmp_path: Path) -> None:
    original_spec = write_json(tmp_path, "original", experiment())
    original = success(tmp_path, "run", str(original_spec))["run"]
    base = original["model_identity"]["fingerprint"]
    first_delta = write_json(tmp_path, "delta-a", {
        "kind": "weight_scale", "target": "output.weight", "factor": 2,
        "expected_base": base,
    })
    first = success(tmp_path, "edit", "create", "repair A", str(first_delta))
    first_spec = write_json(tmp_path, "first", experiment(first["program"]))
    parent_run = success(tmp_path, "run", str(first_spec))["run"]
    second_delta = write_json(tmp_path, "delta-b", {
        "kind": "weight_scale", "target": "output.weight", "factor": 0.5,
        "expected_base": base,
    })
    second = success(
        tmp_path, "edit", "create", "harmful B", str(second_delta), "--parent", first["id"],
    )
    second_spec = write_json(tmp_path, "second", experiment(second["program"]))
    candidate_run = success(tmp_path, "run", str(second_spec))["run"]
    report = success(
        tmp_path, "edit", "compare", original["id"], parent_run["id"], candidate_run["id"],
        first["id"], second["id"],
    )
    assert report["comparable"] is True
    assert report["verdict"] == "observed_regression"
    assert report["lost_prior_improvements"] == ["target"]
    assert report["original_to_parent"]["verdict"] == "observed_improvement"
    assert report["original_to_candidate"]["verdict"] == "no_change"
    assert report["footprint"]["overlapping_targets"] == ["output.weight"]
    history = success(tmp_path, "edit", "history", second["id"])
    assert [revision["id"] for revision in history] == [first["id"], second["id"]]
    shown = success(tmp_path, "edit", "show", second["id"])
    assert shown["revision"] == second
    assert success(
        tmp_path, "edit", "select", second["id"], "--ancestor", first["id"],
    ) == first["program"]
    assert success(tmp_path, "edit", "select", second["id"])["kind"] == "identity"
    package = tmp_path / "edit-package"
    exported = success(tmp_path, "edit", "export", first["id"], parent_run["id"], str(package))
    assert exported["revision"] == first
    invalid = invoke(
        tmp_path, "edit", "export", second["id"], parent_run["id"], str(tmp_path / "wrong"),
    )
    assert invalid.returncode == 2
    assert json.loads(invalid.stderr)["error"]["code"] == "lineage_mismatch"
    assert not (tmp_path / "wrong").exists()
