"""Bounded views of recorded evidence; these functions never execute a model."""

from __future__ import annotations

from typing import Any

from nrly.contracts import NrlyError, RunRecord, SampleResult


def sample_record(run: RunRecord, case_id: str) -> SampleResult:
    sample = next((item for item in run.samples if item.case_id == case_id), None)
    if sample is None:
        raise NrlyError("sample_not_found", "no recorded sample with that case id")
    return sample


def _token_text(data: dict[str, Any]) -> dict[str, Any]:
    text = str(data["text"])
    return {
        **data,
        "text": text[:256],
        "text_characters": len(text),
        "text_truncated": len(text) > 256,
    }


def diagnostics_page(
    run: RunRecord,
    case_id: str,
    offset: int = 0,
    limit: int = 8,
    top_k_offset: int = 0,
    top_k_limit: int = 3,
) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 8 or top_k_offset < 0 or not 0 <= top_k_limit <= 5:
        raise NrlyError("invalid_request", "invalid diagnostic pagination bounds")
    score = sample_record(run, case_id).diagnostics
    if score is None:
        raise NrlyError("diagnostics_unavailable", "sample has no teacher-forced observations")
    tokens = []
    for token in score.tokens[offset : offset + limit]:
        data = _token_text(token.model_dump(mode="json", exclude={"top_k"}))
        data.update(
            top_k=[
                _token_text(item.model_dump(mode="json"))
                for item in token.top_k[top_k_offset : top_k_offset + top_k_limit]
            ],
            top_k_offset=top_k_offset,
            top_k_total=len(token.top_k),
            next_top_k_offset=(
                top_k_offset + top_k_limit
                if top_k_limit and top_k_offset + top_k_limit < len(token.top_k)
                else None
            ),
        )
        tokens.append(data)
    return {
        "run_id": run.id,
        "case_id": case_id,
        "diagnostics": score.model_dump(mode="json", exclude={"tokens"}),
        "offset": offset,
        "total": len(score.tokens),
        "next_offset": offset + limit if offset + limit < len(score.tokens) else None,
        "tokens": tokens,
        "evidence_kind": "teacher_forced_diagnostic_not_task_accuracy",
    }


def capture_page(
    run: RunRecord, case_id: str, target: str, offset: int = 0, limit: int = 64
) -> dict[str, Any]:
    if offset < 0 or not 1 <= limit <= 64:
        raise NrlyError("invalid_request", "invalid capture pagination bounds")
    capture = next(
        (item for item in sample_record(run, case_id).captures if item.target == target), None
    )
    if capture is None:
        raise NrlyError("capture_not_found", "no recorded capture for that target")
    return {
        "run_id": run.id,
        "case_id": case_id,
        "capture": capture.model_dump(mode="json", exclude={"values"}),
        "values": capture.values[offset : offset + limit],
        "offset": offset,
        "total": len(capture.values),
        "next_offset": offset + limit if offset + limit < len(capture.values) else None,
    }
