"""Durable controlled studies built on Runner, with an operator-provisioned holdout.

The boundary protects the ordinary application/MCP run and study APIs. The private
SQLite database and run directory contain secrets: this is deliberately not a
security boundary against the local filesystem owner or an agent with shell access.
Budget accounting charges admitted ceilings without refunds, including the final
evaluation reserved at creation. Wall ceilings cover worker execution, not queue
waiting, operator think time, or process termination overhead.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nrly.contracts import (
    CaseSpec,
    ExperimentSpec,
    ModelIdentity,
    ModelSpec,
    NrlyError,
    ProgramSpec,
    RunRecord,
    canonical_json,
    digest,
)
from nrly.evaluation.reports import SCORER_VERSION
from nrly.evaluation.study_reports import assess_study_arms
from nrly.runner import Runner
from nrly.store import TERMINAL, now
from nrly.study_contracts import (
    AttemptRecord,
    CandidateSpec,
    HoldoutCommitment,
    StudyAssessment,
    StudyEvent,
    StudyRecord,
    StudySpec,
    StudyUsage,
    has_scoring_contract,
    study_arm_name,
    study_arm_names,
    study_seeds,
    validate_targets,
)


def _handle(value: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise NrlyError("invalid_study_handle", "expected a 32-character hexadecimal handle")


def _request(value: str | None) -> None:
    if value is not None and not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,160}", value):
        raise NrlyError("invalid_request", "request id has invalid characters or length")


def _inconclusive(spec: StudySpec, issue: str) -> StudyAssessment:
    return StudyAssessment(
        scorer_version=SCORER_VERSION,
        verdict="inconclusive",
        scope="engineering_fixture" if spec.model.backend == "tiny" else "real_model",
        issues=[issue],
    )


def _has_comparable_evidence(spec: StudySpec, assessment: StudyAssessment) -> bool:
    """Execution completeness is independent of whether the candidate beats controls."""
    names = study_arm_names(spec)
    expected_arms = [
        study_arm_name(spec, name, seed) for seed in study_seeds(spec) for name in names
    ]
    return (
        [arm.arm for arm in assessment.arms] == expected_arms
        and all(arm.complete for arm in assessment.arms)
        and [pair.reference for pair in assessment.comparisons] == names[:-1]
        and all(pair.comparable for pair in assessment.comparisons)
    )


class StudyReader:
    """Read public study records without acquiring a model workspace lease."""

    def __init__(self, root: Path) -> None:
        self.database = root.resolve() / "studies" / "studies.sqlite3"

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        if not self.database.is_file():
            raise NrlyError("study_not_found", "this workspace has no studies")
        connection = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=10)
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _study(connection: sqlite3.Connection, study_id: str) -> StudyRecord:
        _handle(study_id)
        row = connection.execute("SELECT payload FROM studies WHERE id=?", (study_id,)).fetchone()
        if row is None:
            raise NrlyError("study_not_found", "no study has that handle")
        return StudyRecord.model_validate_json(row[0])

    @staticmethod
    def _attempt(connection: sqlite3.Connection, study_id: str, attempt_id: str) -> AttemptRecord:
        _handle(study_id)
        _handle(attempt_id)
        row = connection.execute(
            "SELECT payload FROM attempts WHERE id=? AND study_id=?", (attempt_id, study_id)
        ).fetchone()
        if row is None:
            raise NrlyError("attempt_not_found", "no attempt has those handles")
        return AttemptRecord.model_validate_json(row[0])

    def get(self, study_id: str) -> StudyRecord:
        with self._read() as connection:
            return self._study(connection, study_id)

    def list(self, limit: int = 20) -> list[StudyRecord]:
        if not 1 <= limit <= 100:
            raise NrlyError("invalid_request", "study limit must be between 1 and 100")
        if not self.database.is_file():
            return []
        with self._read() as connection:
            rows = connection.execute(
                "SELECT payload FROM studies ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [StudyRecord.model_validate_json(row[0]) for row in rows]

    def get_attempt(self, study_id: str, attempt_id: str) -> AttemptRecord:
        with self._read() as connection:
            return self._attempt(connection, study_id, attempt_id)

    def attempts(
        self, study_id: str, offset: int = 0, limit: int = 20
    ) -> builtins.list[AttemptRecord]:
        if offset < 0 or not 1 <= limit <= 100:
            raise NrlyError("invalid_request", "invalid attempt pagination")
        with self._read() as connection:
            self._study(connection, study_id)
            rows = connection.execute(
                "SELECT payload FROM attempts WHERE study_id=? ORDER BY sequence LIMIT ? OFFSET ?",
                (study_id, limit, offset),
            ).fetchall()
        return [AttemptRecord.model_validate_json(row[0]) for row in rows]

    def events(self, study_id: str, offset: int = 0, limit: int = 100) -> builtins.list[StudyEvent]:
        if offset < 0 or not 1 <= limit <= 1000:
            raise NrlyError("invalid_request", "invalid ledger pagination")
        with self._read() as connection:
            self._study(connection, study_id)
            rows = connection.execute(
                "SELECT payload FROM events WHERE study_id=? ORDER BY sequence LIMIT ? OFFSET ?",
                (study_id, limit, offset),
            ).fetchall()
        return [StudyEvent.model_validate_json(row[0]) for row in rows]

    def development_cases(self, study_id: str, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 20:
            raise NrlyError("invalid_request", "invalid development case pagination")
        cases = self.get(study_id).spec.development_cases
        return {
            "study_id": study_id,
            "cases": [case.model_dump(mode="json") for case in cases[offset : offset + limit]],
            "total": len(cases),
            "offset": offset,
            "next_offset": offset + limit if offset + limit < len(cases) else None,
        }

    def context_snapshot(self, study_id: str) -> dict[str, Any]:
        """Atomic public handoff; never read sealed tables or candidate tensor payloads."""
        with self._read() as connection:
            connection.execute("BEGIN")
            study = self._study(connection, study_id)
            rows = connection.execute(
                "SELECT id,sequence,json_extract(payload,'$.candidate.name'),"
                "json_extract(payload,'$.candidate.program.kind'),candidate_digest,"
                "json_extract(payload,'$.status'),json_extract(payload,'$.assessment.verdict'),"
                "json_extract(payload,'$.assessment.recommendation'),"
                "json_extract(payload,'$.error_code') FROM attempts "
                "WHERE study_id=? ORDER BY sequence LIMIT 1001",
                (study_id,),
            ).fetchall()
        if len(rows) > 1000 or len(rows) != study.usage.attempts:
            raise NrlyError("study_integrity_error", "attempt count does not match study admission")
        keys = (
            "id",
            "sequence",
            "name",
            "program_kind",
            "candidate_digest",
            "status",
            "verdict",
            "recommendation",
            "error_code",
        )
        identity = study.model_identity
        return {
            "study_id": study.id,
            "name": study.spec.name,
            "status": study.status,
            "evidence_scope": "engineering_fixture"
            if study.spec.model.backend == "tiny"
            else "real_model",
            "spec_digest": study.spec_digest,
            "created_at": study.created_at,
            "finished_at": study.finished_at,
            "model": study.spec.model.model_dump(mode="json"),
            "model_fingerprint": identity.fingerprint if identity else None,
            "generation": study.spec.generation.model_dump(mode="json"),
            "seeds": study.spec.seeds,
            "comparators": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "comparator_digest": digest(item),
                    "model": item.model.model_dump(mode="json") if item.model else None,
                }
                for item in study.spec.comparators
            ],
            "comparator_fingerprints": {
                name: identity.fingerprint for name, identity in study.comparator_identities.items()
            },
            "mutable_targets": study.spec.mutable_targets,
            "development_case_count": len(study.spec.development_cases),
            "holdout": study.holdout.model_dump(mode="json"),
            "controls": [
                {"name": item.name, "program_digest": digest(item.program)}
                for item in study.spec.controls
            ],
            "usage": study.usage.model_dump(mode="json"),
            "budget": study.spec.budget.model_dump(mode="json"),
            "budget_semantics": "admitted ceilings including reserved final; no refunds",
            "nominee_attempt_id": study.nominee_attempt_id,
            "nominee_digest": study.nominee_digest,
            "final_result": study.final_result.model_dump(mode="json")
            if study.final_result
            else None,
            "final_error_code": study.final_error_code,
            "attempts": [dict(zip(keys, row, strict=True)) for row in rows],
            "coverage": {
                "attempts_included": len(rows),
                "attempts_total": study.usage.attempts,
                "attempts_truncated": False,
                "omitted": [
                    "case text and answers",
                    "raw inference outputs",
                    "private run handles",
                    "candidate tensor payloads",
                    "worker logs",
                    "raw event ledger",
                ],
            },
        }

    def development_samples(
        self,
        study_id: str,
        attempt_id: str,
        arm: str,
        offset: int = 0,
        limit: int = 10,
        output_limit: int = 1000,
        output_offset: int = 0,
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 10 or not 1 <= output_limit <= 2000 or output_offset < 0:
            raise NrlyError("invalid_request", "invalid development sample pagination")
        with self._read() as connection:
            self._attempt(connection, study_id, attempt_id)
            row = connection.execute(
                "SELECT run_id FROM development_runs WHERE study_id=? AND attempt_id=? AND arm=?",
                (study_id, attempt_id, arm),
            ).fetchone()
        if row is None:
            raise NrlyError("study_arm_unavailable", "that development arm has no recorded run")
        database = self.database.parent / "private_runs" / "runs.sqlite3"
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=10)
        try:
            run_row = connection.execute(
                "SELECT payload FROM runs WHERE id=?", (row[0],)
            ).fetchone()
        finally:
            connection.close()
        if run_row is None:
            raise NrlyError("study_arm_unavailable", "development arm result is unavailable")
        run = RunRecord.model_validate_json(run_row[0])
        # This is defense in depth in addition to the development-only mapping.
        study_spec = self.get(study_id).spec
        expected_cases = study_spec.development_cases
        allowed_arms = [
            study_arm_name(study_spec, name, seed)
            for seed in study_seeds(study_spec)
            for name in [
                "baseline",
                "no_op",
                "candidate",
                *("control:" + c.name for c in study_spec.controls),
                *("comparator:" + c.name for c in study_spec.comparators),
            ]
        ]
        if arm not in allowed_arms:
            raise NrlyError("study_arm_unavailable", "unregistered development arm")
        for comparator in study_spec.comparators:
            if any(
                arm == study_arm_name(study_spec, "comparator:" + comparator.name, seed)
                for seed in study_seeds(study_spec)
            ):
                expected_cases = comparator.transform_cases(expected_cases)
                break
        if run.spec.cases != expected_cases:
            raise NrlyError("study_arm_unavailable", "development arm case identity mismatch")
        samples = []
        for sample in run.samples[offset : offset + limit]:
            output = sample.output or ""
            samples.append(
                {
                    "case_id": sample.case_id,
                    "status": sample.status,
                    "provenance": sample.provenance,
                    "output": output[output_offset : output_offset + output_limit],
                    "output_offset": output_offset,
                    "output_characters": len(output),
                    "more_output": output_offset + output_limit < len(output),
                    "truncated": output_offset > 0 or len(output) > output_limit,
                    "inference_truncated": sample.truncated,
                    "finish_reason": sample.finish_reason,
                    "error_code": sample.error.code if sample.error else None,
                }
            )
        return {
            "study_id": study_id,
            "attempt_id": attempt_id,
            "arm": arm,
            "samples": samples,
            "total": len(run.samples),
            "offset": offset,
            "next_offset": offset + limit if offset + limit < len(run.samples) else None,
        }


class StudyEngine(StudyReader):
    def __init__(self, public_runner: Runner) -> None:
        super().__init__(public_runner.store.root)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.guard = threading.RLock()
        self.closed = False
        self.futures: dict[str, Future[None]] = {}
        self.cancellations: dict[str, threading.Event] = {}
        # A separate Store means ordinary get_run/list_runs/export cannot retrieve
        # these artifacts, even if a private handle is guessed or learned elsewhere.
        self.private_runner = Runner(
            self.database.parent / "private_runs",
            public_runner.model_cache,
            execution_gate=public_runner.execution_gate,
        )
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nrly-studies")
        try:
            self._initialize()
            self._recover()
        except Exception:
            self.pool.shutdown(wait=True)
            self.private_runner.close()
            raise

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=10)
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._write() as connection:
            connection.executescript(
                "PRAGMA journal_mode=WAL;"
                "CREATE TABLE IF NOT EXISTS studies (id TEXT PRIMARY KEY, request_id TEXT UNIQUE,"
                " spec_digest TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS holdouts (study_id TEXT PRIMARY KEY, payload TEXT "
                "NOT NULL);"
                "CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, study_id TEXT NOT NULL,"
                " sequence INTEGER NOT NULL, request_id TEXT, candidate_digest TEXT NOT NULL,"
                " payload TEXT NOT NULL, UNIQUE(study_id, request_id), UNIQUE(study_id, sequence));"
                "CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY AUTOINCREMENT,"
                " study_id TEXT NOT NULL, payload TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS development_runs (study_id TEXT NOT NULL,"
                " attempt_id TEXT NOT NULL, arm TEXT NOT NULL, run_id TEXT NOT NULL,"
                " PRIMARY KEY(study_id,attempt_id,arm));"
                "CREATE TRIGGER IF NOT EXISTS immutable_event_update BEFORE UPDATE ON events "
                "BEGIN SELECT RAISE(ABORT,'study events are append-only'); END;"
                "CREATE TRIGGER IF NOT EXISTS immutable_event_delete BEFORE DELETE ON events "
                "BEGIN SELECT RAISE(ABORT,'study events are append-only'); END;"
                "CREATE TRIGGER IF NOT EXISTS immutable_holdout_update BEFORE UPDATE ON holdouts "
                "BEGIN SELECT RAISE(ABORT,'holdouts are immutable'); END;"
                "CREATE TRIGGER IF NOT EXISTS immutable_holdout_delete BEFORE DELETE ON holdouts "
                "BEGIN SELECT RAISE(ABORT,'holdouts are immutable'); END;"
            )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        study_id: str,
        kind: str,
        detail_digest: str,
        attempt_id: str | None = None,
    ) -> None:
        row = connection.execute(
            "SELECT payload FROM events WHERE study_id=? ORDER BY sequence DESC LIMIT 1",
            (study_id,),
        ).fetchone()
        previous = StudyEvent.model_validate_json(row[0]).event_digest if row else "0" * 64
        sequence = connection.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM events").fetchone()[
            0
        ]
        data = {
            "sequence": sequence,
            "study_id": study_id,
            "attempt_id": attempt_id,
            "kind": kind,
            "created_at": now(),
            "detail_digest": detail_digest,
            "previous_digest": previous,
        }
        event = StudyEvent(**data, event_digest=digest(data))
        connection.execute(
            "INSERT INTO events VALUES (?,?,?)", (sequence, study_id, canonical_json(event))
        )

    @staticmethod
    def _save_study(connection: sqlite3.Connection, record: StudyRecord) -> None:
        previous = StudyReader._study(connection, record.id)
        if previous.spec_digest != record.spec_digest or previous.spec != record.spec:
            raise NrlyError("immutable_study", "a study design cannot be rewritten")
        if previous.holdout != record.holdout:
            raise NrlyError("immutable_study", "the holdout commitment cannot be rewritten")
        if previous.nominee_digest is not None and (
            record.nominee_digest != previous.nominee_digest
            or record.nominee_attempt_id != previous.nominee_attempt_id
        ):
            raise NrlyError("immutable_nominee", "the final nominee is frozen")
        if previous.status in {"completed", "interrupted"} and previous != record:
            raise NrlyError("immutable_study", "a terminal study cannot be rewritten")
        connection.execute(
            "UPDATE studies SET payload=? WHERE id=?", (canonical_json(record), record.id)
        )

    @staticmethod
    def _save_attempt(connection: sqlite3.Connection, record: AttemptRecord) -> None:
        previous = StudyReader._attempt(connection, record.study_id, record.id)
        if (
            previous.candidate_digest != record.candidate_digest
            or digest(record.candidate) != record.candidate_digest
            or previous.created_at != record.created_at
            or previous.sequence != record.sequence
        ):
            raise NrlyError("immutable_attempt", "registered candidate identity cannot change")
        if previous.status in TERMINAL and previous != record:
            raise NrlyError("immutable_attempt", "terminal attempts cannot be rewritten")
        connection.execute(
            "UPDATE attempts SET payload=? WHERE id=?", (canonical_json(record), record.id)
        )

    @staticmethod
    def _charge(spec: StudySpec, usage: StudyUsage, count: int, *, final: bool) -> StudyUsage:
        runs = (3 + len(spec.controls) + len(spec.comparators)) * len(study_seeds(spec))
        charged = StudyUsage(
            attempts=usage.attempts + (0 if final else 1),
            runs=usage.runs + runs,
            case_executions=usage.case_executions + runs * count,
            output_tokens=usage.output_tokens + runs * spec.run_budget.max_output_tokens,
            capture_bytes=usage.capture_bytes + runs * spec.run_budget.max_capture_bytes,
            wall_seconds=usage.wall_seconds + runs * spec.run_budget.wall_seconds,
            final_runs_reserved=runs if final else usage.final_runs_reserved,
        )
        budget = spec.budget
        if (
            charged.attempts > budget.max_attempts
            or charged.runs > budget.max_runs
            or charged.case_executions > budget.max_case_executions
            or charged.output_tokens > budget.max_output_tokens
            or charged.capture_bytes > budget.max_capture_bytes
            or charged.wall_seconds > budget.max_wall_seconds
        ):
            raise NrlyError("study_budget_exceeded", "cumulative study admission budget exhausted")
        return charged

    def _open(self) -> None:
        if self.closed:
            raise NrlyError("study_engine_closed", "study engine is shutting down")

    def create(
        self, spec: StudySpec, holdout: list[CaseSpec], request_id: str | None = None
    ) -> StudyRecord:
        """Operator-only provisioning. Never expose this method through the agent MCP API."""
        _request(request_id)
        spec = StudySpec.model_validate_json(spec.model_dump_json())
        sealed = [CaseSpec.model_validate_json(case.model_dump_json()) for case in holdout]
        if (
            not sealed
            or len(sealed) > spec.run_budget.max_cases
            or any(not has_scoring_contract(case) or case.split != "test" for case in sealed)
            or len({case.id for case in sealed}) != len(sealed)
            or len({case.group for case in sealed if case.group is not None}) > 32
            or not any(case.role == "target" for case in sealed)
            or not any(case.role == "regression" for case in sealed)
        ):
            raise NrlyError(
                "invalid_holdout",
                "holdout requires bounded, unique, scored test targets/regressions",
            )
        public_ids = {case.id for case in spec.development_cases}

        def prompt_key(case: CaseSpec) -> str:
            # Plain input is rendered as a single user turn by chat-capable backends.
            return canonical_json(
                {
                    "messages": [m.model_dump(mode="json") for m in case.messages]
                    if case.messages
                    else [{"role": "user", "content": case.input}]
                }
            )

        public_inputs = {prompt_key(case) for case in spec.development_cases}
        sealed_inputs = {prompt_key(case) for case in sealed}
        for prompt_comparator in spec.comparators:
            public_inputs.update(
                prompt_key(case)
                for case in prompt_comparator.transform_cases(spec.development_cases)
            )
            sealed_inputs.update(
                prompt_key(case) for case in prompt_comparator.transform_cases(sealed)
            )
        if any(case.id in public_ids for case in sealed) or public_inputs & sealed_inputs:
            raise NrlyError("holdout_overlap", "holdout ids and exact inputs must be disjoint")
        for comparator in [None, *spec.comparators]:
            ExperimentSpec(
                schema_version="0.3" if spec.schema_version == "0.2" else "0.2",
                name="sealed study admission validation",
                model=comparator.model if comparator and comparator.model else spec.model,
                generation=spec.generation,
                cases=comparator.transform_cases(sealed) if comparator else sealed,
                budget=spec.run_budget,
            )
        serialized = json.dumps([case.model_dump(mode="json") for case in sealed], sort_keys=True)
        if len(serialized.encode()) + len(canonical_json(spec).encode()) > 4 * 1024 * 1024:
            raise NrlyError("budget_exceeded", "study specification exceeds 4 MiB")
        commitment = HoldoutCommitment(
            digest=hashlib.sha256(serialized.encode()).hexdigest(),
            count=len(sealed),
            targets=sum(case.role == "target" for case in sealed),
            regressions=sum(case.role == "regression" for case in sealed),
        )
        spec_digest = digest(
            {"spec": spec.model_dump(mode="json"), "holdout": commitment.model_dump()}
        )
        usage = self._charge(spec, StudyUsage(), len(sealed), final=True)
        # A design must admit at least one discovery attempt as well as its final.
        self._charge(spec, usage, len(spec.development_cases), final=False)
        with self.guard, self._write() as connection:
            self._open()
            if request_id is not None:
                row = connection.execute(
                    "SELECT payload,spec_digest FROM studies WHERE request_id=?", (request_id,)
                ).fetchone()
                if row:
                    if row[1] != spec_digest:
                        raise NrlyError("request_conflict", "request id names another study design")
                    return StudyRecord.model_validate_json(row[0])
            record = StudyRecord(
                id=uuid.uuid4().hex,
                spec=spec,
                spec_digest=spec_digest,
                holdout=commitment,
                created_at=now(),
                usage=usage,
            )
            connection.execute(
                "INSERT INTO studies VALUES (?,?,?,?,?)",
                (record.id, request_id, spec_digest, record.created_at, canonical_json(record)),
            )
            connection.execute("INSERT INTO holdouts VALUES (?,?)", (record.id, serialized))
            self._event(connection, record.id, "created_final_budget_reserved", spec_digest)
            return record

    def submit_candidate(
        self, study_id: str, candidate: CandidateSpec, request_id: str | None = None
    ) -> AttemptRecord:
        _request(request_id)
        candidate = CandidateSpec.model_validate_json(candidate.model_dump_json())
        candidate_digest = digest(candidate)
        if len(canonical_json(candidate).encode()) > 4 * 1024 * 1024:
            raise NrlyError("budget_exceeded", "candidate specification exceeds 4 MiB")
        with self.guard:
            self._open()
            with self._write() as connection:
                study = self._study(connection, study_id)
                if request_id is not None:
                    row = connection.execute(
                        "SELECT payload,candidate_digest FROM attempts WHERE study_id=? "
                        "AND request_id=?",
                        (study_id, request_id),
                    ).fetchone()
                    if row:
                        if row[1] != candidate_digest:
                            raise NrlyError(
                                "request_conflict", "request id names another candidate"
                            )
                        return AttemptRecord.model_validate_json(row[0])
                if study.status != "open":
                    raise NrlyError("study_frozen", "the study no longer accepts candidates")
                try:
                    validate_targets(candidate, study.spec.mutable_targets)
                except ValueError as error:
                    raise NrlyError("invalid_study_target", str(error)) from error
                usage = self._charge(
                    study.spec, study.usage, len(study.spec.development_cases), final=False
                )
                attempt = AttemptRecord(
                    id=uuid.uuid4().hex,
                    study_id=study_id,
                    sequence=usage.attempts,
                    candidate=candidate,
                    candidate_digest=candidate_digest,
                    status="queued",
                    created_at=now(),
                )
                connection.execute(
                    "INSERT INTO attempts VALUES (?,?,?,?,?,?)",
                    (
                        attempt.id,
                        study_id,
                        attempt.sequence,
                        request_id,
                        candidate_digest,
                        canonical_json(attempt),
                    ),
                )
                self._save_study(connection, study.model_copy(update={"usage": usage}))
                self._event(
                    connection, study_id, "candidate_admitted", candidate_digest, attempt.id
                )
            self.cancellations[attempt.id] = threading.Event()
            self.futures[attempt.id] = self.pool.submit(self._execute_attempt, study_id, attempt.id)
            return attempt

    def finalize(self, study_id: str, attempt_id: str) -> StudyRecord:
        with self.guard:
            self._open()
            with self._write() as connection:
                study = self._study(connection, study_id)
                attempt = self._attempt(connection, study_id, attempt_id)
                if study.nominee_attempt_id is not None:
                    if study.nominee_attempt_id != attempt_id:
                        raise NrlyError(
                            "study_frozen", "a different final nominee is already frozen"
                        )
                    return study
                if study.status != "open" or attempt.status != "succeeded":
                    raise NrlyError("nominee_incomplete", "nominate a completed discovery attempt")
                active = connection.execute(
                    "SELECT COUNT(*) FROM attempts WHERE study_id=? AND "
                    "json_extract(payload,'$.status') IN ('queued','running')",
                    (study_id,),
                ).fetchone()[0]
                if active:
                    raise NrlyError("study_busy", "finish or cancel admitted candidates first")
                study = study.model_copy(
                    update={
                        "status": "finalizing",
                        "nominee_attempt_id": attempt.id,
                        "nominee_digest": attempt.candidate_digest,
                    }
                )
                self._save_study(connection, study)
                self._event(
                    connection, study_id, "nominee_frozen", attempt.candidate_digest, attempt.id
                )
            key = "final:" + study_id
            self.cancellations[key] = threading.Event()
            self.futures[key] = self.pool.submit(self._execute_final, study_id)
            return study

    def cancel(self, study_id: str, attempt_id: str) -> AttemptRecord:
        with self.guard:
            attempt = self.get_attempt(study_id, attempt_id)
            if attempt.status in TERMINAL:
                return attempt
            event = self.cancellations.get(attempt_id)
            if event is None:
                raise NrlyError("attempt_not_owned", "this engine does not own the attempt")
            event.set()
            return attempt

    def wait(self, study_id: str, attempt_id: str, timeout: float | None = None) -> AttemptRecord:
        self.get_attempt(study_id, attempt_id)
        future = self.futures.get(attempt_id)
        if future is not None:
            future.result(timeout)
        return self.get_attempt(study_id, attempt_id)

    def wait_final(self, study_id: str, timeout: float | None = None) -> StudyRecord:
        self.get(study_id)
        future = self.futures.get("final:" + study_id)
        if future is not None:
            future.result(timeout)
        return self.get(study_id)

    def cancel_final(self, study_id: str) -> StudyRecord:
        with self.guard:
            study = self.get(study_id)
            if study.status in {"completed", "interrupted"}:
                return study
            if study.status != "finalizing":
                raise NrlyError("final_not_started", "this study has no final evaluation to cancel")
            event = self.cancellations.get("final:" + study_id)
            if event is None:
                raise NrlyError("final_not_owned", "this engine does not own the final evaluation")
            event.set()
            return study

    def close(self) -> None:
        with self.guard:
            if self.closed:
                return
            self.closed = True
            for event in self.cancellations.values():
                event.set()
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.private_runner.close()

    def _recover(self) -> None:
        with self._write() as connection:
            rows = connection.execute(
                "SELECT payload FROM attempts WHERE json_extract(payload,'$.status') "
                "IN ('queued','running')"
            ).fetchall()
            for row in rows:
                attempt = AttemptRecord.model_validate_json(row[0])
                study = self._study(connection, attempt.study_id)
                interrupted = attempt.model_copy(
                    update={
                        "status": "interrupted",
                        "finished_at": now(),
                        "error_code": "study_restarted",
                        "assessment": _inconclusive(study.spec, "attempt interrupted by restart"),
                    }
                )
                self._save_attempt(connection, interrupted)
                self._event(
                    connection, study.id, "attempt_interrupted", digest(interrupted), attempt.id
                )
            rows = connection.execute(
                "SELECT payload FROM studies WHERE json_extract(payload,'$.status')='finalizing'"
            ).fetchall()
            for row in rows:
                study = StudyRecord.model_validate_json(row[0])
                interrupted_study = study.model_copy(
                    update={
                        "status": "interrupted",
                        "finished_at": now(),
                        "final_error_code": "study_restarted",
                        "final_result": _inconclusive(
                            study.spec, "final evaluation interrupted by restart"
                        ),
                    }
                )
                self._save_study(connection, interrupted_study)
                self._event(connection, study.id, "final_interrupted", digest(interrupted_study))

    def _pin_identity(
        self, study_id: str, identity: ModelIdentity | None, comparator: str | None = None
    ) -> None:
        if identity is None:
            return
        with self._write() as connection:
            study = self._study(connection, study_id)
            registered_models = {
                item.name: item.model for item in study.spec.comparators if item.kind == "model"
            }
            model = registered_models.get(comparator) if comparator else study.spec.model
            if (
                model is None
                or (identity.backend, identity.model_id, identity.revision)
                != (model.backend, model.model_id, model.revision)
                or not identity.fingerprint
            ):
                raise NrlyError(
                    "study_identity_mismatch", "executed model does not match registration"
                )
            related = [
                pin
                for name, pin in study.comparator_identities.items()
                if registered_models.get(name) == model
            ]
            if model == study.spec.model and study.model_identity is not None:
                related.append(study.model_identity)
            if any(pin != identity for pin in related):
                raise NrlyError("study_identity_mismatch", "study runtime identity changed")
            previous = (
                study.comparator_identities.get(comparator) if comparator else study.model_identity
            )
            if previous is None:
                updates: dict[str, Any] = {"model_identity": identity}
                if comparator:
                    updates = {
                        "comparator_identities": {
                            **study.comparator_identities,
                            comparator: identity,
                        }
                    }
                self._save_study(connection, study.model_copy(update=updates))
                self._event(connection, study.id, "runtime_pinned", digest(identity))
            elif previous != identity:
                raise NrlyError("study_identity_mismatch", "study runtime identity changed")

    def _arms(
        self,
        study: StudyRecord,
        candidate: CandidateSpec,
        cases: list[CaseSpec],
        key: str,
        event: threading.Event,
    ) -> list[tuple[str, RunRecord]]:
        programs: list[tuple[str, ProgramSpec, ModelSpec, list[CaseSpec], str | None]] = [
            ("baseline", ProgramSpec(), study.spec.model, cases, None),
            ("no_op", ProgramSpec(), study.spec.model, cases, None),
            *(
                ("control:" + control.name, control.program, study.spec.model, cases, None)
                for control in study.spec.controls
            ),
            *(
                (
                    "comparator:" + comparator.name,
                    ProgramSpec(),
                    comparator.model or study.spec.model,
                    comparator.transform_cases(cases),
                    comparator.name if comparator.kind == "model" else None,
                )
                for comparator in study.spec.comparators
            ),
            ("candidate", candidate.program, study.spec.model, cases, None),
        ]
        results: list[tuple[str, RunRecord]] = []
        for seed in study_seeds(study.spec):
            generation = study.spec.generation.model_copy(update={"seed": seed})
            for name, program, model, arm_cases, comparator_key in programs:
                if event.is_set():
                    return results
                arm = study_arm_name(study.spec, name, seed)
                current = self.get(study.id)
                pinned = (
                    current.comparator_identities.get(comparator_key)
                    if comparator_key
                    else current.model_identity
                )
                if pinned is not None and program.expected_base is None:
                    program = program.model_copy(update={"expected_base": pinned.fingerprint})
                spec = ExperimentSpec(
                    schema_version="0.3" if study.spec.schema_version == "0.2" else "0.2",
                    name="controlled study arm",
                    model=model,
                    generation=generation,
                    program=program,
                    cases=arm_cases,
                    budget=study.spec.run_budget,
                )
                run = self.private_runner.submit(spec, request_id=f"{key}:{len(results)}")
                if not key.startswith("final:"):
                    with self._write() as connection:
                        connection.execute(
                            "INSERT INTO development_runs VALUES (?,?,?,?)",
                            (study.id, key, arm, run.id),
                        )
                while True:
                    if event.is_set():
                        self.private_runner.cancel(run.id)
                    try:
                        run = self.private_runner.wait(run.id, timeout=0.1)
                        break
                    except TimeoutError:
                        continue
                results.append((arm, run))
                self._pin_identity(study.id, run.model_identity, comparator_key)
                if run.status != "succeeded":
                    return results
        return results

    def _execute_attempt(self, study_id: str, attempt_id: str) -> None:
        event = self.cancellations[attempt_id]
        study = self.get(study_id)
        attempt = self.get_attempt(study_id, attempt_id)
        assessment = _inconclusive(study.spec, "attempt did not complete")
        status = "failed"
        error_code: str | None = None
        try:
            with self._write() as connection:
                self._save_attempt(connection, attempt.model_copy(update={"status": "running"}))
                self._event(
                    connection, study_id, "attempt_started", attempt.candidate_digest, attempt_id
                )
            arms = self._arms(
                study, attempt.candidate, study.spec.development_cases, attempt_id, event
            )
            assessment = assess_study_arms(study.spec, arms)
            status = "succeeded" if _has_comparable_evidence(study.spec, assessment) else "failed"
            if status == "failed":
                error_code = "study_inconclusive"
        except Exception:
            error_code = "study_execution_failed"
        if event.is_set():
            status, error_code = "cancelled", "study_cancelled"
            assessment = _inconclusive(study.spec, "attempt cancelled")
        final = AttemptRecord.model_validate(
            {
                **attempt.model_dump(mode="json"),
                "status": status,
                "finished_at": now(),
                "assessment": assessment.model_dump(mode="json"),
                "error_code": error_code,
            }
        )
        with self._write() as connection:
            self._save_attempt(connection, final)
            self._event(connection, study_id, "attempt_" + status, digest(final), attempt_id)

    def _execute_final(self, study_id: str) -> None:
        study = self.get(study_id)
        event = self.cancellations["final:" + study_id]
        assessment = _inconclusive(study.spec, "final evaluation did not complete")
        error_code: str | None = None
        try:
            if study.nominee_attempt_id is None:
                raise NrlyError("nominee_incomplete", "final evaluation requires a frozen nominee")
            attempt = self.get_attempt(study_id, study.nominee_attempt_id)
            if digest(attempt.candidate) != study.nominee_digest:
                raise NrlyError("immutable_nominee", "the frozen candidate digest does not match")
            with self._read() as connection:
                row = connection.execute(
                    "SELECT payload FROM holdouts WHERE study_id=?", (study_id,)
                ).fetchone()
            if row is None or hashlib.sha256(row[0].encode()).hexdigest() != study.holdout.digest:
                raise NrlyError("holdout_mismatch", "the sealed case commitment does not match")
            cases = [CaseSpec.model_validate(item) for item in json.loads(row[0])]
            arms = self._arms(study, attempt.candidate, cases, "final:" + study_id, event)
            assessment = assess_study_arms(study.spec, arms)
            if not _has_comparable_evidence(study.spec, assessment):
                error_code = "study_inconclusive"
        except Exception:
            error_code = "study_execution_failed"
        if event.is_set():
            assessment = _inconclusive(study.spec, "final evaluation cancelled")
            error_code = "study_cancelled"
        with self._write() as connection:
            current = self._study(connection, study_id)
            final = current.model_copy(
                update={
                    "status": "completed",
                    "finished_at": now(),
                    "final_result": assessment,
                    "final_error_code": error_code,
                }
            )
            self._save_study(connection, final)
            self._event(connection, study_id, "final_completed", digest(final))
