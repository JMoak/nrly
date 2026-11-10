from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import TracebackType

import pytest

import nrly
from nrly import cli
from nrly.contracts import (
    CaseSpec,
    ExperimentSpec,
    ModelIdentity,
    ModelInspection,
    RunRecord,
    SampleResult,
    TensorCapture,
    WeightChange,
    digest,
)
from nrly.store import read_bundle


def invoke(tmp_path: Path, *arguments: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(nrly.__file__).resolve().parents[1])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "nrly.cli", "--root", str(tmp_path / "runs"), *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )


@pytest.mark.parametrize("kind", ["experiment", "model", "program"])
def test_cli_emits_parseable_public_schemas_without_creating_store(
    tmp_path: Path, kind: str
) -> None:
    result = invoke(tmp_path, "schema", kind)
    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    if kind == "program":
        assert "weight_patch" in schema["properties"]["kind"]["enum"]
        assert schema["properties"]["indices"]["items"]["type"] == "integer"
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("content", ["{invalid", '{"name":"bad","cases":[]}'])
def test_cli_rejects_malformed_specs_with_machine_readable_error(
    tmp_path: Path, content: str
) -> None:
    spec_file = tmp_path / "invalid.json"
    spec_file.write_text(content, encoding="utf-8")
    result = invoke(tmp_path, "run", str(spec_file))
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]["code"] == "invalid_request"
    assert not list((tmp_path / "runs").glob("runs/*/worker.log"))


def replay_record(*, inspection: bool = False) -> RunRecord:
    spec = (
        ExperimentSpec(name="Inspection replay fixture", purpose="inspection")
        if inspection
        else ExperimentSpec(name="Replay fixture", cases=[CaseSpec(id="sum", input="[2,4]")])
    )
    return RunRecord(
        id="a" * 32,
        status="succeeded",
        spec=spec,
        spec_digest=digest(spec),
        created_at="2026-11-10T00:00:00+00:00",
        model_identity=ModelIdentity(
            backend="tiny", model_id="nrly/tiny-sum-v1", revision="1", fingerprint="b" * 64
        ),
        samples=[]
        if inspection
        else [
            SampleResult(case_id="sum", status="completed", provenance="torch_fixture", output="3")
        ],
        inspection=ModelInspection(
            weights=[
                TensorCapture(
                    target="output.weight",
                    shape=[1, 2],
                    dtype="torch.float32",
                    values=[0.5, 0.5],
                    truncated=False,
                )
            ]
        )
        if inspection
        else None,
    )


@pytest.mark.parametrize(
    "mode",
    [
        "same",
        "output",
        "identity",
        "inspection",
        "inspection_same",
        "inspection_identity",
        "weight_hash",
        "weight_count",
        "weight_removed",
        "weight_enriched",
        "activation_values",
        "activation_enriched",
        "sample_status",
        "sample_identity",
        "sample_timing",
        "token_accounting",
        "experiment_inspection_values",
        "experiment_inspection_enriched",
        "inspection_enriched",
    ],
)
def test_cli_replay_checks_evidence_and_returns_failure_on_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
) -> None:
    original = replay_record(inspection=mode.startswith("inspection"))
    if mode.startswith("weight"):
        change = WeightChange(
            target="output.weight",
            shape=[1, 2],
            dtype="torch.float32",
            before_sha256="e" * 64,
            after_sha256="f" * 64,
            total_elements=2,
            changed_elements=1,
            edited_indices=[0],
            edited_before=[0.5],
            edited_after=[1],
        )
        original = original.model_copy(
            update={
                "samples": [original.samples[0].model_copy(update={"weight_changes": [change]})]
            }
        )
    if mode.startswith("activation"):
        capture = TensorCapture(
            target="hidden",
            shape=[1, 2],
            dtype="torch.float32",
            values=[2, 4],
            truncated=False,
        )
        original = original.model_copy(
            update={"samples": [original.samples[0].model_copy(update={"captures": [capture]})]}
        )
    if mode.startswith("experiment_inspection"):
        original = original.model_copy(
            update={"inspection": replay_record(inspection=True).inspection}
        )
    replay = original.model_copy(update={"id": "c" * 32})
    if mode == "output":
        replay = replay.model_copy(
            update={"samples": [original.samples[0].model_copy(update={"output": "4"})]}
        )
    elif mode in {"identity", "inspection_identity"}:
        assert original.model_identity is not None
        replay = replay.model_copy(
            update={
                "model_identity": original.model_identity.model_copy(
                    update={"fingerprint": "d" * 64}
                )
            }
        )
    elif mode == "inspection":
        assert original.inspection is not None
        replay = replay.model_copy(
            update={"inspection": original.inspection.model_copy(update={"weights": []})}
        )
    elif mode.startswith("weight"):
        changed = original.samples[0].weight_changes[0]
        if mode == "weight_hash":
            changed = changed.model_copy(update={"after_sha256": "0" * 64})
        elif mode == "weight_count":
            changed = changed.model_copy(update={"changed_elements": 0})
        elif mode == "weight_enriched":
            changed = changed.model_copy(update={"edited_coordinates": [[0, 0]]})
        replay = replay.model_copy(
            update={
                "samples": [
                    original.samples[0].model_copy(
                        update={
                            "weight_changes": [] if mode == "weight_removed" else [changed],
                        }
                    )
                ]
            }
        )
    elif mode.startswith("activation"):
        changed_capture = (
            original.samples[0]
            .captures[0]
            .model_copy(
                update=(
                    {"values": [2, 5]}
                    if mode == "activation_values"
                    else {"flat_indices": [0, 1], "coordinates": [[0, 0], [0, 1]]}
                )
            )
        )
        replay = replay.model_copy(
            update={
                "samples": [original.samples[0].model_copy(update={"captures": [changed_capture]})]
            }
        )
    elif mode in {"sample_status", "sample_identity", "sample_timing", "token_accounting"}:
        field, value = {
            "sample_status": ("status", "failed"),
            "sample_identity": ("case_id", "different"),
            "sample_timing": ("elapsed_ms", 12345.0),
            "token_accounting": ("output_tokens", 7),
        }[mode]
        replay = replay.model_copy(
            update={"samples": [original.samples[0].model_copy(update={field: value})]}
        )
    elif mode in {
        "experiment_inspection_values",
        "experiment_inspection_enriched",
        "inspection_enriched",
    }:
        assert original.inspection is not None
        changed_weight = original.inspection.weights[0].model_copy(
            update=(
                {"values": [0.5, 1]}
                if mode == "experiment_inspection_values"
                else {"flat_indices": [0, 1], "coordinates": [[0, 0], [0, 1]]}
            )
        )
        replay = replay.model_copy(
            update={
                "inspection": original.inspection.model_copy(update={"weights": [changed_weight]})
            }
        )

    class ReplayRunner:
        def __init__(self, _root: Path, _cache: Path | None) -> None:
            pass

        def __enter__(self) -> ReplayRunner:
            return self

        def __exit__(
            self,
            _kind: type[BaseException] | None,
            _error: BaseException | None,
            _trace: TracebackType | None,
        ) -> None:
            pass

        def submit(self, spec: ExperimentSpec, *, parent_run: str) -> RunRecord:
            assert spec.program.expected_base == "b" * 64
            assert parent_run == original.id
            return replay

        def wait(self, run_id: str) -> RunRecord:
            assert run_id == replay.id
            return replay

    monkeypatch.setattr(cli, "Runner", ReplayRunner)
    monkeypatch.setattr(cli, "read_bundle", lambda _path: original)
    arguments = cli.parser().parse_args(
        ["--root", str(tmp_path), "replay", str(tmp_path / "bundle")]
    )
    matching = mode in {
        "same",
        "inspection_same",
        "weight_enriched",
        "activation_enriched",
        "sample_timing",
        "experiment_inspection_enriched",
        "inspection_enriched",
    }
    assert cli.execute(arguments) == (0 if matching else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["replay"]["original_run"] == original.id
    assert report["replay"]["replay_matches"] is matching
    if mode == "inspection":
        assert report["replay"]["inspection_matches"] is False
    if mode in {"weight_hash", "weight_count", "weight_removed"}:
        assert report["replay"]["outputs_match"] is True
        assert report["replay"]["weight_evidence_matches"] is False
    if mode == "activation_values":
        assert report["replay"]["outputs_match"] is True
        assert report["replay"]["activation_evidence_matches"] is False
    if mode == "experiment_inspection_values":
        assert report["replay"]["outputs_match"] is True
        assert report["replay"]["inspection_matches"] is False


def test_cli_actual_torch_demo_export_and_replay(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    bundle = tmp_path / "bundle"
    demo = invoke(tmp_path, "demo", "--export", str(bundle), timeout=60)
    assert demo.returncode == 0, demo.stderr + demo.stdout
    result = json.loads(demo.stdout)
    assert result["scope"] == "engineering_fixture"
    assert result["base_restored"] is True
    assert result["comparison"]["comparable"] is True
    exported = read_bundle(bundle)
    assert exported.spec.program.kind == "weight_scale"
    assert exported.status == "succeeded"
    assert exported.samples[1].output == "6"
    assert exported.samples[1].weight_changes[0].changed_elements == 2
    replay = invoke(tmp_path, "replay", str(bundle), timeout=30)
    assert replay.returncode == 0, replay.stderr + replay.stdout
    assert json.loads(replay.stdout)["replay"]["outputs_match"] is True


@pytest.mark.parametrize("content", ["{broken", "{}", "null", "42"])
def test_inspection_rejects_malformed_read_selection_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "reads.json"
    path.write_text(content, encoding="utf-8")
    result = invoke(tmp_path, "inspect", "--read-spec", str(path))
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"]["code"] == "invalid_request"


def test_legacy_inspection_replay_allows_enrichment_but_checks_original_evidence() -> None:
    old = {"weights": [{"values": [0.5, 0.5], "shape": [1, 2]}]}
    enriched = {"weights": [{**old["weights"][0], "coordinates": [[0, 0], [0, 1]]}]}
    assert cli.contains_recorded_evidence(old, enriched)
    enriched["weights"][0]["values"] = [0.5, 1.0]
    assert not cli.contains_recorded_evidence(old, enriched)
    assert not cli.contains_recorded_evidence(old, {"weights": []})


def test_cli_redirected_unicode_uses_utf8_despite_inherited_windows_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    recipes = Path(__file__).parents[1] / "recipes/studies"
    spec = json.loads((recipes / "tiny-study.json").read_text(encoding="utf-8"))
    spec["name"] = "Étude — 🧪"
    spec_file = tmp_path / "study.json"
    spec_file.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    created = invoke(
        tmp_path,
        "study",
        "create",
        str(spec_file),
        "--holdout",
        str(recipes / "tiny-holdout.operator.json"),
    )
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["spec"]["name"] == "Étude — 🧪"
    missing_file = tmp_path / "missing — 🧪.json"
    failed = invoke(tmp_path, "run", str(missing_file))
    assert failed.returncode == 2
    assert "missing — 🧪.json" in json.loads(failed.stderr)["error"]["message"]
