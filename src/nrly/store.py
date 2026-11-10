from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from nrly.contracts import ExperimentSpec, NrlyError, RunRecord, canonical_json, digest

TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "runs.sqlite3"
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs "
                "(id TEXT PRIMARY KEY, request_id TEXT UNIQUE, spec_digest TEXT NOT NULL, "
                "created_at TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise NrlyError("invalid_run_id", "run id must be a 32-character hexadecimal handle")
        path = self.root / "runs" / run_id
        if not path.resolve().is_relative_to(self.root):
            raise NrlyError("invalid_path", "run path escapes the experiment store")
        return path

    def create(
        self, spec: ExperimentSpec, request_id: str | None = None, parent_run: str | None = None
    ) -> tuple[RunRecord, bool]:
        serialized = canonical_json(spec)
        if len(serialized.encode()) > 4 * 1024 * 1024:
            raise NrlyError("budget_exceeded", "experiment specification exceeds 4 MiB")
        if request_id is not None and not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,160}", request_id):
            raise NrlyError("invalid_request", "request id has invalid characters or length")
        spec_digest = digest(spec)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if request_id is not None:
                row = connection.execute(
                    "SELECT payload,spec_digest FROM runs WHERE request_id=?", (request_id,)
                ).fetchone()
                if row:
                    if row[1] != spec_digest:
                        raise NrlyError("request_conflict", "request id already names another spec")
                    return RunRecord.model_validate_json(row[0]), False
            record = RunRecord(
                id=uuid.uuid4().hex,
                status="queued",
                spec=spec,
                spec_digest=spec_digest,
                created_at=now(),
                parent_run=parent_run,
            )
            atomic_text(self.run_dir(record.id) / "spec.json", serialized)
            connection.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?)",
                (record.id, request_id, spec_digest, record.created_at, canonical_json(record)),
            )
        return record, True

    def get(self, run_id: str) -> RunRecord:
        self.run_dir(run_id)
        with self.connect() as connection:
            row = connection.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise NrlyError("run_not_found", f"no experiment run has handle {run_id}")
        return RunRecord.model_validate_json(row[0])

    def list(self, limit: int = 20) -> list[RunRecord]:
        if not 1 <= limit <= 1000:
            raise NrlyError("invalid_request", "limit must be between 1 and 1000")
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [RunRecord.model_validate_json(row[0]) for row in rows]

    def unfinished(self) -> builtins.list[RunRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM runs WHERE json_extract(payload, '$.status') "
                "NOT IN ('succeeded','failed','cancelled','interrupted') ORDER BY created_at"
            ).fetchall()
        return [RunRecord.model_validate_json(row[0]) for row in rows]

    def save(self, record: RunRecord) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT payload FROM runs WHERE id=?", (record.id,)).fetchone()
            if row is None:
                raise NrlyError("run_not_found", "cannot update an unknown run")
            previous = RunRecord.model_validate_json(row[0])
            if previous.status in TERMINAL and canonical_json(previous) != canonical_json(record):
                raise NrlyError("immutable_run", "completed runs cannot be rewritten")
            if (
                previous.spec_digest != record.spec_digest
                or digest(record.spec) != record.spec_digest
            ):
                raise NrlyError("spec_mismatch", "a run's experiment specification is immutable")
            if (previous.created_at, previous.parent_run) != (record.created_at, record.parent_run):
                raise NrlyError("identity_mismatch", "run creation and lineage are immutable")
            if record.status in TERMINAL:
                data = canonical_json(record)
                atomic_text(self.run_dir(record.id) / "run.json", data)
                atomic_text(
                    self.run_dir(record.id) / "manifest.json",
                    json.dumps(
                        {
                            "schema_version": "0.1",
                            "run_id": record.id,
                            "run_sha256": hashlib.sha256(data.encode()).hexdigest(),
                            "spec_sha256": record.spec_digest,
                        },
                        indent=2,
                    ),
                )
            connection.execute(
                "UPDATE runs SET payload=? WHERE id=?", (canonical_json(record), record.id)
            )

    def export(self, run_id: str, destination: Path) -> Path:
        record = self.get(run_id)
        if record.status not in TERMINAL:
            raise NrlyError("run_incomplete", "only terminal runs can be exported")
        source = self.run_dir(run_id)
        verified = read_bundle(source)
        if canonical_json(verified) != canonical_json(record):
            raise NrlyError("artifact_mismatch", "run artifact differs from the recorded result")
        destination = destination.resolve()
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise NrlyError("destination_exists", "export destination must be absent or empty")
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("run.json", "manifest.json", "spec.json"):
            atomic_text(destination / name, (source / name).read_text(encoding="utf-8"))
        return destination


def read_bundle(directory: Path) -> RunRecord:
    directory = directory.resolve()
    files = [directory / name for name in ("manifest.json", "run.json", "spec.json")]
    for path in files:
        if path.resolve().parent != directory or not path.is_file():
            raise NrlyError("invalid_bundle", "bundle files must be local regular files")
        if path.stat().st_size > 32 * 1024 * 1024:
            raise NrlyError("invalid_bundle", "bundle member exceeds 32 MiB")
    try:
        manifest = json.loads(files[0].read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise NrlyError("invalid_bundle", "manifest must be a JSON object")
        content = files[1].read_bytes()
        record = RunRecord.model_validate_json(content)
        spec = ExperimentSpec.model_validate_json(files[2].read_bytes())
        if (
            manifest.get("schema_version") != "0.1"
            or manifest.get("run_id") != record.id
            or manifest.get("run_sha256") != hashlib.sha256(content).hexdigest()
            or manifest.get("spec_sha256") != digest(spec)
            or digest(spec) != record.spec_digest
            or digest(record.spec) != record.spec_digest
        ):
            raise NrlyError("artifact_mismatch", "bundle checksums or identities do not match")
        return record
    except (ValueError, TypeError, KeyError) as error:
        raise NrlyError("invalid_bundle", f"invalid experiment bundle: {error}") from error
