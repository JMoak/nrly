from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from nrly.api import Application, bounded
from nrly.contracts import (
    CaseSpec,
    ContinuationScore,
    DiagnosticSpec,
    ExperimentSpec,
    ModelSpec,
    NrlyError,
    RunRecord,
    SampleResult,
    TensorCapture,
    TokenAlternative,
    TokenObservation,
)
from nrly.evidence import capture_page, diagnostics_page
from nrly.review import render_run_report


@pytest.fixture
def evidence_app(tmp_path: Path) -> Iterator[tuple[Application, RunRecord]]:
    app = Application(tmp_path / "runs")
    spec = ExperimentSpec(
        schema_version="0.3",
        name="Recorded diagnostic evidence pagination",
        model=ModelSpec(backend="huggingface", model_id="local-fixture", revision="a" * 40),
        diagnostics=DiagnosticSpec(top_k=5),
        cases=[CaseSpec(id="probe", input="prefix", continuation=" continuation")],
    )
    queued, _created = app.store.create(spec)
    tokens = [
        TokenObservation(
            position=i + 2,
            token_id=i,
            text="x" * 300,
            logprob=-1,
            rank=2,
            top_k=[TokenAlternative(token_id=j, text=str(j), logprob=-0.5 - j) for j in range(5)],
        )
        for i in range(17)
    ]
    score = ContinuationScore(
        prompt_tokens=2,
        continuation_tokens=17,
        sum_logprob=-17,
        mean_logprob=-1,
        tokens=tokens,
    )
    sample = SampleResult(
        case_id="probe",
        status="completed",
        provenance="huggingface",
        output="",
        finish_reason="teacher_forced",
        input_tokens=19,
        output_tokens=0,
        diagnostics=score,
        captures=[
            TensorCapture(
                target="mlp",
                shape=[1, 2, 48],
                dtype="torch.float32",
                values=list(range(96)),
                truncated=False,
                token_positions=[0, 2],
                invocation_index=0,
                source="input",
            )
        ],
    )
    record = queued.model_copy(update={"status": "succeeded", "samples": [sample]})
    app.store.save(record)
    try:
        yield app, record
    finally:
        app.close()


def test_diagnostic_pages_recover_tokens_and_alternatives_without_mutating_evidence(
    evidence_app: tuple[Application, RunRecord],
) -> None:
    app, run = evidence_app
    observed = []
    for offset in (0, 8, 16):
        page = bounded(app.get_diagnostics(run.id, "probe", offset=offset))
        observed.extend(page["tokens"])
        assert page["evidence_kind"] == "teacher_forced_diagnostic_not_task_accuracy"
    assert [token["position"] for token in observed] == list(range(2, 19))
    assert page["next_offset"] is None
    assert all(token["text_truncated"] and token["text_characters"] == 300 for token in observed)
    assert len(observed[0]["top_k"]) == 3
    remainder = app.get_diagnostics(run.id, "probe", limit=1, top_k_offset=3)
    assert [item["token_id"] for item in remainder["tokens"][0]["top_k"]] == [3, 4]
    assert remainder["tokens"][0]["next_top_k_offset"] is None
    saved = app.store.get(run.id).samples[0].diagnostics
    assert saved is not None and len(saved.tokens[0].text) == 300
    assert len(saved.tokens[0].top_k) == 5


def test_sample_summary_does_not_inline_large_token_evidence(
    evidence_app: tuple[Application, RunRecord],
) -> None:
    app, run = evidence_app
    data = bounded(app.get_sample(run.id, "probe"))
    assert data["diagnostic_tokens_omitted"] is True
    assert data["diagnostic_token_count"] == 17
    assert "tokens" not in data["diagnostics"]


def test_capture_pages_preserve_selection_metadata(
    evidence_app: tuple[Application, RunRecord],
) -> None:
    app, run = evidence_app
    first = bounded(app.get_capture(run.id, "probe", "mlp"))
    last = bounded(app.get_capture(run.id, "probe", "mlp", offset=64))
    assert first["values"] + last["values"] == list(range(96))
    assert first["capture"]["token_positions"] == [0, 2]
    assert first["capture"]["source"] == "input"
    assert first["capture"]["truncated"] is False
    assert last["next_offset"] is None


@pytest.mark.parametrize("offset,limit,top_limit", [(-1, 1, 1), (0, 9, 1), (0, 1, 6)])
def test_diagnostic_bounds_reject_invalid_pages(
    evidence_app: tuple[Application, RunRecord],
    offset: int,
    limit: int,
    top_limit: int,
) -> None:
    _app, run = evidence_app
    with pytest.raises(NrlyError, match="pagination"):
        diagnostics_page(run, "probe", offset, limit, top_k_limit=top_limit)


def test_unknown_evidence_never_falls_back_to_another_sample(
    evidence_app: tuple[Application, RunRecord],
) -> None:
    _app, run = evidence_app
    with pytest.raises(NrlyError, match="case id"):
        diagnostics_page(run, "missing")
    with pytest.raises(NrlyError, match="target"):
        capture_page(run, "probe", "missing")


def test_offline_diagnostic_report_separates_probabilities_from_task_accuracy_and_escapes_text(
    evidence_app: tuple[Application, RunRecord],
) -> None:
    _app, run = evidence_app
    sample = run.samples[0]
    assert sample.diagnostics is not None
    score = sample.diagnostics.model_copy(
        update={
            "tokens": [
                sample.diagnostics.tokens[0].model_copy(
                    update={"text": "<script>alert(1)</script>"}
                ),
                *sample.diagnostics.tokens[1:],
            ]
        }
    )
    run = run.model_copy(update={"samples": [sample.model_copy(update={"diagnostics": score})]})
    html = render_run_report(run)
    assert "Continuation diagnostics" in html
    assert "not generated answers or task accuracy" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
