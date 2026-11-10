from __future__ import annotations

import hashlib
import importlib.metadata
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nrly import __version__
from nrly.contracts import (
    CaptureSpec,
    ExperimentSpec,
    ModelSpec,
    NrlyError,
    RunRecord,
    WeightReadSpec,
    digest,
)
from nrly.evaluation import compare_runs, evaluate_run
from nrly.runner import Runner
from nrly.studies import StudyEngine, StudyReader
from nrly.study_contracts import AttemptRecord, CandidateSpec, StudyAssessment, StudyRecord

RESPONSE_LIMIT = 12000


def assessment_preview(value: StudyAssessment | None) -> dict[str, Any] | None:
    if value is None:
        return None
    data = value.model_dump(mode="json")
    if len(json.dumps(data, ensure_ascii=False)) <= 5000:
        return data
    return {
        **data,
        "arms": [],
        "comparisons": [],
        "issues": [str(item)[:200] for item in data["issues"][:5]],
        "arm_count": len(data["arms"]),
        "comparison_count": len(data["comparisons"]),
        "truncated": True,
        "next": "Use get_study_assessment for the complete JSON record.",
    }


def bounded(data: dict[str, Any]) -> dict[str, Any]:
    if len(json.dumps(data, ensure_ascii=False)) > RESPONSE_LIMIT:
        raise NrlyError(
            "response_too_large",
            "Response exceeds 12000 characters. Reduce page/output limits, select one weight "
            "target, or use CLI show/compare/export for complete evidence.",
        )
    return data


def capabilities() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "transformers", "mcp"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "name": "nrly",
        "version": __version__,
        "field": "weight-space programming",
        "schema_version": "0.3",
        "supported_schema_versions": ["0.1", "0.2", "0.3"],
        "backends": ["tiny", "huggingface"],
        "weight_operations": [
            "weight_scale",
            "weight_replace",
            "weight_patch",
            "weight_add",
            "weight_sparse_add",
            "weight_low_rank",
            "weight_slice_scale",
        ],
        "composition": "pinned atomic pipelines of up to 32 ordered weight operations",
        "diagnostic_operations": ["activation_scale", "activation_add"],
        "observations": ["teacher_forced_token_logprobs", "selected_activation_captures"],
        "authoring": "trusted Python builders, tensor artifacts, edit ancestry and patch packages",
        "python_adapter": "caller-owned PyTorch model and callable; no code loading over MCP",
        "control_operations": ["identity"],
        "arbitrary_code": False,
        "execution": "one isolated model worker at a time; asynchronous submission",
        "inspection": (
            "offsets, indices, coordinates and slices; selection statistics; zero forwards"
        ),
        "studies": (
            "operator-provisioned holdout; adaptive candidates; cumulative reservations; "
            "one final nominee"
        ),
        "artifacts": "SHA256-verified declarative programs and run bundles; CLI export/replay",
        "limits": {"response_characters": RESPONSE_LIMIT, "weight_payload_elements": 65536},
        "packages": packages,
        "evidence": "tiny backend is an engineering fixture, never a real-model capability result",
    }


def run_summary(run: RunRecord) -> dict[str, Any]:
    return {
        "id": run.id,
        "status": run.status,
        "name": run.spec.name,
        "purpose": run.spec.purpose,
        "model_id": run.spec.model.model_id,
        "created_at": run.created_at,
        "finished_at": run.finished_at,
        "samples_completed": sum(sample.status == "completed" for sample in run.samples),
        "case_count": len(run.spec.cases),
        "elapsed_ms": run.elapsed_ms,
        "error": {
            "code": run.error.code[:100],
            "message": run.error.message[:300],
            "truncated": len(run.error.message) > 300,
        }
        if run.error
        else None,
    }


class Application:
    """Shared application boundary for CLI and MCP; model imports stay in workers."""

    def __init__(self, root: Path, model_cache: Path | None = None) -> None:
        self.runner = Runner(root, model_cache)
        self.store = self.runner.store
        self.study_reader = StudyReader(self.store.root)
        self._studies: StudyEngine | None = None
        self._study_guard = threading.Lock()
        self._closed = False

    @property
    def studies(self) -> StudyEngine:
        with self._study_guard:
            if self._closed:
                raise NrlyError("runner_closed", "application is shutting down")
            if self._studies is None:
                self._studies = StudyEngine(self.runner)
            return self._studies

    def close(self) -> None:
        with self._study_guard:
            self._closed = True
        try:
            if self._studies is not None:
                self._studies.close()
        finally:
            self.runner.close()

    def submit(self, spec: ExperimentSpec, request_id: str | None = None) -> dict[str, Any]:
        record = self.runner.submit(spec, request_id)
        return {"run_id": record.id, "status": record.status}

    def submit_inspection(
        self,
        model: ModelSpec,
        weight_reads: Sequence[CaptureSpec],
        name: str = "Model inspection",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return self.submit(
            ExperimentSpec(
                name=name,
                purpose="inspection",
                model=model,
                weight_reads=[
                    WeightReadSpec.model_validate(read.model_dump()) for read in weight_reads
                ],
            ),
            request_id,
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        record = self.store.get(run_id)
        data = record.model_dump(mode="json")
        if len(json.dumps(data, ensure_ascii=False)) <= RESPONSE_LIMIT:
            return data
        return {
            **run_summary(record),
            "truncated": True,
            "artifact": f"nrly://runs/{record.id}",
            "next": "Use get_samples/get_sample/get_inspection for details, or CLI export.",
        }

    def get_samples(self, run_id: str, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 20:
            raise NrlyError("invalid_request", "invalid sample-list pagination bounds")
        record = self.store.get(run_id)
        return {
            "run_id": run_id,
            "status": record.status,
            "offset": offset,
            "total": len(record.samples),
            "expected_cases": len(record.spec.cases),
            "next_offset": offset + limit if offset + limit < len(record.samples) else None,
            "samples": [
                {
                    "case_id": sample.case_id,
                    "status": sample.status,
                    "provenance": sample.provenance,
                    "elapsed_ms": sample.elapsed_ms,
                    "output_characters": len(sample.output or ""),
                    "output_preview": (sample.output or "")[:160],
                    "error_code": sample.error.code if sample.error else None,
                }
                for sample in record.samples[offset : offset + limit]
            ],
        }

    def get_inspection(
        self,
        run_id: str,
        offset: int = 0,
        limit: int = 40,
        weight_target: str | None = None,
        value_offset: int = 0,
        value_limit: int = 32,
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 40 or value_offset < 0 or not 1 <= value_limit <= 64:
            raise NrlyError("invalid_request", "invalid inspection pagination bounds")
        record = self.store.get(run_id)
        if record.inspection is None:
            raise NrlyError(
                "inspection_unavailable", "inspection is not available; check run status"
            )
        inspection = record.inspection
        reads = []
        for weight in inspection.weights:
            if weight_target is None or weight.target == weight_target:
                data = weight.model_dump(mode="json")
                data.update(
                    values=weight.values[value_offset : value_offset + value_limit],
                    value_offset=value_offset,
                    captured_elements=len(weight.values),
                    more_values=value_offset + value_limit < len(weight.values),
                )
                for key in ("flat_indices", "coordinates"):
                    if key in data:
                        data[key] = data[key][value_offset : value_offset + value_limit]
                reads.append(data)
        if weight_target and not reads:
            raise NrlyError(
                "weight_not_captured", "submit another inspection requesting that target"
            )
        return {
            "run_id": run_id,
            "status": record.status,
            "model_identity": record.model_identity.model_dump(mode="json")
            if record.model_identity
            else None,
            "tensors": [item.model_dump() for item in inspection.tensors[offset : offset + limit]],
            "total": len(inspection.tensors),
            "offset": offset,
            "limit": limit,
            "next_offset": offset + limit if offset + limit < len(inspection.tensors) else None,
            "weight_reads": reads,
        }

    def get_sample(
        self,
        run_id: str,
        case_id: str,
        output_offset: int = 0,
        output_limit: int = 2000,
        change_offset: int = 0,
        change_limit: int = 4,
    ) -> dict[str, Any]:
        if (
            output_offset < 0
            or not 1 <= output_limit <= 4000
            or change_offset < 0
            or not 1 <= change_limit <= 4
        ):
            raise NrlyError("invalid_request", "invalid sample pagination bounds")
        run = self.store.get(run_id)
        sample = next((item for item in run.samples if item.case_id == case_id), None)
        if sample is None:
            raise NrlyError("sample_not_found", "no recorded sample with that case id")
        data = sample.model_dump(mode="json")
        output = sample.output or ""
        data.update(
            output=output[output_offset : output_offset + output_limit],
            output_offset=output_offset,
            output_characters=len(output),
            more_output=output_offset + output_limit < len(output),
            weight_changes=data["weight_changes"][change_offset : change_offset + change_limit],
            change_offset=change_offset,
            weight_change_count=len(sample.weight_changes),
            next_change_offset=(
                change_offset + change_limit
                if change_offset + change_limit < len(sample.weight_changes)
                else None
            ),
        )
        data["captures"] = [
            {
                **capture.model_dump(mode="json"),
                "values": capture.values[:16],
                "truncated": capture.truncated or len(capture.values) > 16,
            }
            for capture in sample.captures
        ]
        if sample.diagnostics is not None:
            data["diagnostics"] = sample.diagnostics.model_dump(mode="json", exclude={"tokens"})
            data["diagnostic_token_count"] = len(sample.diagnostics.tokens)
            data["diagnostic_tokens_omitted"] = True
        return data

    def get_diagnostics(
        self,
        run_id: str,
        case_id: str,
        offset: int = 0,
        limit: int = 8,
        top_k_offset: int = 0,
        top_k_limit: int = 3,
    ) -> dict[str, Any]:
        from nrly.evidence import diagnostics_page

        return diagnostics_page(
            self.store.get(run_id), case_id, offset, limit, top_k_offset, top_k_limit
        )

    def get_capture(
        self, run_id: str, case_id: str, target: str, offset: int = 0, limit: int = 64
    ) -> dict[str, Any]:
        from nrly.evidence import capture_page

        return capture_page(self.store.get(run_id), case_id, target, offset, limit)

    def list_runs(self, limit: int = 20) -> dict[str, Any]:
        if not 1 <= limit <= 20:
            raise NrlyError("invalid_request", "limit must be between 1 and 20")
        return {"runs": [run_summary(record) for record in self.store.list(limit)]}

    def cancel(self, run_id: str) -> dict[str, Any]:
        record = self.runner.cancel(run_id)
        return {"run_id": record.id, "status": record.status}

    def compare(self, baseline_id: str, candidate_id: str) -> dict[str, Any]:
        report = compare_runs(self.store.get(baseline_id), self.store.get(candidate_id))
        data = report.model_dump(mode="json")
        if len(json.dumps(data, ensure_ascii=False)) <= RESPONSE_LIMIT:
            return data
        for name in ("baseline", "candidate"):
            data[name].pop("cases", None)
            data[name]["issues"] = data[name]["issues"][:10]
        for name in ("improved_cases", "regressed_cases", "unchanged_cases", "issues"):
            data[name] = data[name][:20]
        data["truncated"] = True
        data["next"] = "Use CLI compare for the complete report."
        return data

    def evaluate(self, run_id: str) -> dict[str, Any]:
        return evaluate_run(self.store.get(run_id)).model_dump(mode="json")

    def get_study(
        self, study_id: str, target_offset: int = 0, target_limit: int = 16
    ) -> dict[str, Any]:
        if target_offset < 0 or not 1 <= target_limit <= 16:
            raise NrlyError("invalid_request", "invalid mutable-target pagination bounds")
        return study_summary(self.study_reader.get(study_id), target_offset, target_limit)

    def list_studies(self, limit: int = 20) -> dict[str, Any]:
        if not 1 <= limit <= 20:
            raise NrlyError("invalid_request", "study limit must be between 1 and 20")
        return {
            "studies": [
                {
                    "id": record.id,
                    "name": record.spec.name,
                    "status": record.status,
                    "attempts": record.usage.attempts,
                    "created_at": record.created_at,
                    "verdict": record.final_result.verdict if record.final_result else None,
                }
                for record in self.study_reader.list(limit)
            ]
        }

    def submit_study_candidate(
        self,
        study_id: str,
        candidate: CandidateSpec,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        record = self.studies.submit_candidate(study_id, candidate, request_id)
        return {"study_id": record.study_id, "attempt_id": record.id, "status": record.status}

    def get_study_attempt(self, study_id: str, attempt_id: str) -> dict[str, Any]:
        record = self.study_reader.get_attempt(study_id, attempt_id)
        return {
            **attempt_summary(record),
            "assessment": assessment_preview(record.assessment),
        }

    def get_study_assessment(
        self,
        study_id: str,
        attempt_id: str | None = None,
        offset: int = 0,
        limit: int = 4000,
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 4000:
            raise NrlyError("invalid_request", "invalid assessment pagination")
        assessment = (
            self.study_reader.get_attempt(study_id, attempt_id).assessment
            if attempt_id is not None
            else self.study_reader.get(study_id).final_result
        )
        if assessment is None:
            raise NrlyError("assessment_unavailable", "assessment is not available yet")
        text = json.dumps(assessment.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return {
            "study_id": study_id,
            "attempt_id": attempt_id,
            "format": "json",
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "offset": offset,
            "total": len(text),
            "content": text[offset : offset + limit],
            "next_offset": offset + limit if offset + limit < len(text) else None,
        }

    def get_study_comparator(
        self,
        study_id: str,
        comparator: str,
        offset: int = 0,
        limit: int = 4000,
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 4000:
            raise NrlyError("invalid_request", "invalid comparator pagination")
        study = self.study_reader.get(study_id)
        registered = next(
            (item for item in study.spec.comparators if item.name == comparator), None
        )
        if registered is None:
            raise NrlyError("comparator_not_found", "no comparator registered with that name")
        identity = study.comparator_identities.get(comparator)
        if registered.kind == "prompt":
            identity = study.model_identity
        text = json.dumps(
            {
                "registration": registered.model_dump(mode="json"),
                "registration_digest": digest(registered),
                "model_identity": identity.model_dump(mode="json") if identity else None,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return {
            "study_id": study_id,
            "comparator": comparator,
            "format": "json",
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "offset": offset,
            "total": len(text),
            "content": text[offset : offset + limit],
            "next_offset": offset + limit if offset + limit < len(text) else None,
        }

    def list_study_attempts(
        self, study_id: str, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 20:
            raise NrlyError("invalid_request", "invalid study attempt pagination")
        study = self.study_reader.get(study_id)
        return {
            "study_id": study_id,
            "offset": offset,
            "total": study.usage.attempts,
            "next_offset": offset + limit if offset + limit < study.usage.attempts else None,
            "attempts": [
                attempt_summary(record)
                for record in self.study_reader.attempts(
                    study_id,
                    offset,
                    limit,
                )
            ],
        }

    def finalize_study(self, study_id: str, attempt_id: str) -> dict[str, Any]:
        record = self.studies.finalize(study_id, attempt_id)
        return {"study_id": record.id, "status": record.status}

    def cancel_study_attempt(self, study_id: str, attempt_id: str) -> dict[str, Any]:
        record = self.studies.cancel(study_id, attempt_id)
        return {"study_id": study_id, "attempt_id": attempt_id, "status": record.status}

    def cancel_study_final(self, study_id: str) -> dict[str, Any]:
        record = self.studies.cancel_final(study_id)
        return {"study_id": record.id, "status": record.status}

    def get_study_cases(
        self,
        study_id: str,
        offset: int = 0,
        limit: int = 5,
        input_offset: int = 0,
        input_limit: int = 1000,
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 5 or input_offset < 0 or not 1 <= input_limit <= 1000:
            raise NrlyError("invalid_request", "invalid development case pagination")
        cases = self.study_reader.get(study_id).spec.development_cases
        return {
            "study_id": study_id,
            "offset": offset,
            "total": len(cases),
            "next_offset": offset + limit if offset + limit < len(cases) else None,
            "cases": [
                {
                    **case.model_dump(mode="json", exclude={"messages", "continuation", "adapter"}),
                    "input": (case.input or "")[input_offset : input_offset + input_limit],
                    "expected": (case.expected or "")[input_offset : input_offset + input_limit],
                    "input_offset": input_offset,
                    "input_characters": len(case.input or ""),
                    "expected_characters": len(case.expected or ""),
                    "message_count": len(case.messages),
                    "adapter_kind": case.adapter.kind if case.adapter else None,
                    "case_spec_available": True,
                    "more_input": input_offset + input_limit
                    < max(len(case.input or ""), len(case.expected or "")),
                }
                for case in cases[offset : offset + limit]
            ],
        }

    def get_study_case_spec(
        self,
        study_id: str,
        case_id: str,
        offset: int = 0,
        limit: int = 4000,
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 4000:
            raise NrlyError("invalid_request", "invalid case specification pagination")
        case = next(
            (
                item
                for item in self.study_reader.get(study_id).spec.development_cases
                if item.id == case_id
            ),
            None,
        )
        if case is None:
            raise NrlyError("case_not_found", "no development case with that id")
        text = json.dumps(case.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return {
            "study_id": study_id,
            "case_id": case_id,
            "format": "json",
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "offset": offset,
            "total": len(text),
            "content": text[offset : offset + limit],
            "next_offset": offset + limit if offset + limit < len(text) else None,
        }

    def get_study_case_messages(
        self,
        study_id: str,
        case_id: str,
        offset: int = 0,
        limit: int = 4,
        content_offset: int = 0,
        content_limit: int = 1000,
    ) -> dict[str, Any]:
        if (
            offset < 0
            or not 1 <= limit <= 4
            or content_offset < 0
            or not 1 <= content_limit <= 1000
        ):
            raise NrlyError("invalid_request", "invalid development message pagination")
        case = next(
            (
                item
                for item in self.study_reader.get(study_id).spec.development_cases
                if item.id == case_id
            ),
            None,
        )
        if case is None:
            raise NrlyError("case_not_found", "no development case with that id")
        return {
            "study_id": study_id,
            "case_id": case_id,
            "offset": offset,
            "total": len(case.messages),
            "next_offset": offset + limit if offset + limit < len(case.messages) else None,
            "messages": [
                {
                    "role": item.role,
                    "content": item.content[content_offset : content_offset + content_limit],
                    "content_offset": content_offset,
                    "content_characters": len(item.content),
                    "more_content": content_offset + content_limit < len(item.content),
                }
                for item in case.messages[offset : offset + limit]
            ],
        }

    def get_study_samples(
        self,
        study_id: str,
        attempt_id: str,
        arm: str,
        offset: int = 0,
        limit: int = 5,
        output_limit: int = 512,
        output_offset: int = 0,
    ) -> dict[str, Any]:
        return self.study_reader.development_samples(
            study_id, attempt_id, arm, offset, limit, output_limit, output_offset
        )


def study_summary(
    record: StudyRecord, target_offset: int = 0, target_limit: int = 16
) -> dict[str, Any]:
    return {
        "id": record.id,
        "name": record.spec.name,
        "status": record.status,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
        "spec_digest": record.spec_digest,
        "holdout": record.holdout.model_dump(),
        "model": record.spec.model.model_dump(),
        "generation": record.spec.generation.model_dump(),
        "model_identity": record.model_identity.model_dump(mode="json")
        if record.model_identity
        else None,
        "development_case_count": len(record.spec.development_cases),
        "controls": [control.name for control in record.spec.controls],
        "seeds": record.spec.seeds,
        "comparators": [
            {
                "name": item.name,
                "kind": item.kind,
                "registration_digest": digest(item),
                "model_fingerprint": record.comparator_identities[item.name].fingerprint
                if item.name in record.comparator_identities
                else record.model_identity.fingerprint
                if item.kind == "prompt" and record.model_identity
                else None,
            }
            for item in record.spec.comparators
        ],
        "mutable_targets": record.spec.mutable_targets[target_offset : target_offset + target_limit]
        if record.spec.mutable_targets
        else None,
        "mutable_target_offset": target_offset,
        "next_target_offset": (
            target_offset + target_limit
            if record.spec.mutable_targets
            and target_offset + target_limit < len(record.spec.mutable_targets)
            else None
        ),
        "mutable_target_count": len(record.spec.mutable_targets)
        if record.spec.mutable_targets
        else None,
        "usage": record.usage.model_dump(),
        "budget": record.spec.budget.model_dump(),
        "budget_semantics": "admitted ceilings including reserved final evaluation; no refunds",
        "nominee_attempt_id": record.nominee_attempt_id,
        "nominee_digest": record.nominee_digest,
        "final_result": assessment_preview(record.final_result),
        "final_error_code": record.final_error_code,
    }


def attempt_summary(record: AttemptRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "attempt_id": record.id,
        "study_id": record.study_id,
        "sequence": record.sequence,
        "name": record.candidate.name,
        "status": record.status,
        "program_kind": record.candidate.program.kind,
        "candidate_digest": record.candidate_digest,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
        "error_code": record.error_code,
        "verdict": record.assessment.verdict if record.assessment else None,
        "recommendation": record.assessment.recommendation if record.assessment else None,
    }
