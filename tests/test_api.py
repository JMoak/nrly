from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from nrly.api import RESPONSE_LIMIT, Application, bounded
from nrly.contracts import (
    CaseSpec,
    ErrorInfo,
    ExperimentSpec,
    ModelIdentity,
    ModelInspection,
    NrlyError,
    ProgramSpec,
    RunRecord,
    SampleResult,
    TensorCapture,
    TensorInfo,
    WeightChange,
)
from nrly.mcp import public_errors
from nrly.study_contracts import StudySpec


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Application]:
    application = Application(tmp_path / "runs")
    try:
        yield application
    finally:
        application.close()


def saved_run(
    app: Application,
    *,
    inspection: ModelInspection | None = None,
    output: str = "3",
) -> RunRecord:
    spec = (
        ExperimentSpec(name="Inspection API fixture", purpose="inspection")
        if inspection is not None
        else ExperimentSpec(name="Sample API fixture", cases=[CaseSpec(id="sum", input="[2,4]")])
    )
    queued, _created = app.store.create(spec)
    samples = (
        []
        if inspection is not None
        else [
            SampleResult(
                case_id="sum",
                status="completed",
                provenance="torch_fixture",
                output=output,
                captures=[
                    TensorCapture(
                        target="hidden",
                        shape=[64],
                        dtype="torch.float32",
                        values=list(range(64)),
                        truncated=False,
                    )
                ],
            )
        ]
    )
    record = queued.model_copy(
        update={
            "status": "succeeded",
            "samples": samples,
            "inspection": inspection,
            "model_identity": ModelIdentity(
                backend="tiny", model_id="nrly/tiny-sum-v1", revision="1", fingerprint="a" * 64
            ),
        }
    )
    app.store.save(record)
    return record


def test_large_run_has_explicit_summary_and_recoverable_sample_pages(app: Application) -> None:
    output = "abcd\U0001f9e0" * 3000
    record = saved_run(app, output=output)
    summary = bounded(app.get_run(record.id))
    assert summary["truncated"] is True
    assert summary["artifact"] == f"nrly://runs/{record.id}"
    assert "samples" not in summary
    parts = []
    for offset in range(0, len(output), 4000):
        page = bounded(app.get_sample(record.id, "sum", offset, 4000))
        parts.append(page["output"])
        assert page["output_characters"] == len(output)
        assert len(page["captures"][0]["values"]) == 16
        assert page["captures"][0]["truncated"] is True
        assert len(json.dumps(page, ensure_ascii=False)) <= RESPONSE_LIMIT
    assert "".join(parts) == output
    assert page["more_output"] is False
    assert app.store.get(record.id).samples[0].captures[0].truncated is False


def test_inspection_pages_catalog_and_selected_weight_values(app: Application) -> None:
    inspection = ModelInspection(
        tensors=[
            TensorInfo(name=f"layer.{i}.weight", kind="parameter", shape=[128]) for i in range(43)
        ],
        weights=[
            TensorCapture(
                target="layer.0.weight",
                shape=[256],
                dtype="torch.float32",
                values=list(range(128)),
                truncated=True,
            )
        ],
    )
    record = saved_run(app, inspection=inspection)
    first = bounded(app.get_inspection(record.id, limit=40, value_limit=64))
    second = bounded(app.get_inspection(record.id, offset=40, value_offset=64, value_limit=64))
    assert first["next_offset"] == 40
    assert second["next_offset"] is None
    assert len(first["tensors"]) == 40 and len(second["tensors"]) == 3
    assert first["weight_reads"][0]["values"] + second["weight_reads"][0]["values"] == list(
        range(128)
    )
    assert first["weight_reads"][0]["more_values"] is True
    assert second["weight_reads"][0]["more_values"] is False
    assert second["weight_reads"][0]["truncated"] is True
    with pytest.raises(NrlyError) as error:
        app.get_inspection(record.id, weight_target="uncaptured.weight")
    assert error.value.code == "weight_not_captured"


def test_oversized_inspection_has_public_error_and_can_be_retried_with_smaller_page(
    app: Application,
) -> None:
    inspection = ModelInspection(
        tensors=[
            TensorInfo(
                name=f"model.layers.{i}.mlp.down_proj.weight",
                kind="parameter",
                shape=[1024, 3072],
                dtype="torch.float32",
            )
            for i in range(40)
        ],
        weights=[
            TensorCapture(
                target=f"model.layers.{i}.mlp.down_proj.weight",
                shape=[1024, 3072],
                dtype="torch.float32",
                values=[0.12345678901234567] * 64,
                truncated=True,
            )
            for i in range(8)
        ],
    )
    record = saved_run(app, inspection=inspection)
    read = public_errors(app.get_inspection)
    with pytest.raises(ToolError, match="response_too_large"):
        read(record.id, value_limit=64)
    smaller = read(record.id, limit=10, weight_target=inspection.weights[0].target, value_limit=16)
    assert len(smaller["weight_reads"]) == 1
    assert len(json.dumps(smaller, ensure_ascii=False)) <= RESPONSE_LIMIT


def test_domain_errors_remain_identifiable_at_public_tool_boundary(app: Application) -> None:
    with pytest.raises(ToolError, match="invalid_run_id"):
        public_errors(app.get_run)("../outside")
    with pytest.raises(ToolError, match="invalid_request"):
        public_errors(app.list_runs)(21)


@pytest.mark.parametrize("value", [True, False, 1.0, "1"])
@pytest.mark.parametrize("field", ["shape", "indices"])
def test_parameter_addresses_reject_coercion(value: object, field: str) -> None:
    data = {
        "kind": "weight_patch",
        "target": "output.weight",
        "shape": [1, 2],
        "indices": [1],
        "values": [0],
    }
    data[field] = [value, 2] if field == "shape" else [value]
    with pytest.raises(ValidationError):
        ProgramSpec.model_validate(data)


def test_large_failure_message_is_bounded_in_run_summary(app: Application) -> None:
    spec = ExperimentSpec(name="Failed fixture", cases=[CaseSpec(id="sum", input="[2,4]")])
    queued, _created = app.store.create(spec)
    record = queued.model_copy(
        update={"status": "failed", "error": ErrorInfo(code="backend_failed", message="x" * 20000)}
    )
    app.store.save(record)
    result = bounded(app.get_run(record.id))
    assert result["error"]["truncated"] is True
    assert len(result["error"]["message"]) == 300


def test_all_pipeline_changes_are_accessible_through_bounded_pages(app: Application) -> None:
    spec = ExperimentSpec(name="Pipeline evidence", cases=[CaseSpec(id="sum", input="[2,4]")])
    queued, _ = app.store.create(spec)
    changes = [
        WeightChange(
            target="output.weight",
            shape=[1, 2],
            dtype="torch.float32",
            before_sha256=str(index) * 64,
            after_sha256=str(index + 1) * 64,
            total_elements=2,
            changed_elements=2,
            before_preview=[0.5, 0.5],
            after_preview=[1, 1],
            operation_index=index,
            operation_kind="weight_scale",
        )
        for index in range(32)
    ]
    app.store.save(
        queued.model_copy(
            update={
                "status": "succeeded",
                "samples": [
                    SampleResult(
                        case_id="sum",
                        status="completed",
                        provenance="torch_fixture",
                        output="6",
                        weight_changes=changes,
                    )
                ],
            }
        )
    )
    recovered = []
    for offset in range(0, 32, 4):
        page = bounded(app.get_sample(queued.id, "sum", change_offset=offset))
        assert page["change_offset"] == offset
        assert page["weight_change_count"] == 32
        recovered.extend(page["weight_changes"])
    assert recovered == [change.model_dump(mode="json") for change in changes]
    assert page["next_change_offset"] is None
    with pytest.raises(NrlyError, match="pagination"):
        app.get_sample(queued.id, "sum", change_limit=5)


def test_study_mutable_targets_are_all_recoverable_without_holdout_data(app: Application) -> None:
    targets = [f"model.layers.{i}.mlp.down_proj.weight" for i in range(45)]
    study = app.studies.create(
        StudySpec(
            name="Large mutable interface",
            mutable_targets=targets,
            development_cases=[
                CaseSpec(id="dev-target", input="[2,4]", expected="6"),
                CaseSpec(id="dev-regression", input="[0,0]", expected="0", role="regression"),
            ],
        ),
        [
            CaseSpec(id="holdout-secret", input="[9,9]", expected="18", split="test"),
            CaseSpec(
                id="holdout-regression",
                input="[0.0,0.0]",
                expected="0",
                split="test",
                role="regression",
            ),
        ],
    )
    recovered = []
    for offset in range(0, len(targets), 16):
        page = bounded(app.get_study(study.id, target_offset=offset))
        assert page["mutable_target_offset"] == offset
        assert page["mutable_target_count"] == len(targets)
        assert "holdout-secret" not in json.dumps(page)
        recovered.extend(page["mutable_targets"])
    assert recovered == targets
    assert page["next_target_offset"] is None
    with pytest.raises(NrlyError, match="pagination"):
        app.get_study(study.id, target_limit=17)
