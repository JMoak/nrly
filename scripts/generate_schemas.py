"""Regenerate or verify public schemas from the authoritative runtime contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from nrly.artifacts import TensorArtifact
from nrly.contracts import (
    CaptureSpec,
    DiagnosticSpec,
    ExperimentSpec,
    ModelSpec,
    ProgramSpec,
    RunRecord,
    TaskAdapterSpec,
    WeightOperationSpec,
    WeightReadSpec,
)
from nrly.deployment import PatchManifest
from nrly.lineage import EditRevision, SequentialReport
from nrly.study_contracts import (
    AttemptRecord,
    CandidateSpec,
    ComparatorSpec,
    StudyAssessment,
    StudyBudget,
    StudyRecord,
    StudySpec,
)

SCHEMAS: dict[str, type[BaseModel]] = {
    "diagnostics": DiagnosticSpec,
    "capture": CaptureSpec,
    "task-adapter": TaskAdapterSpec,
    "comparator": ComparatorSpec,
    "tensor-artifact": TensorArtifact,
    "edit-revision": EditRevision,
    "patch": PatchManifest,
    "sequential-report": SequentialReport,
    "experiment": ExperimentSpec,
    "model": ModelSpec,
    "program": ProgramSpec,
    "run": RunRecord,
    "weight-read": WeightReadSpec,
    "weight-operation": WeightOperationSpec,
    "candidate": CandidateSpec,
    "study": StudySpec,
    "study-budget": StudyBudget,
    "study-record": StudyRecord,
    "study-attempt": AttemptRecord,
    "study-assessment": StudyAssessment,
}
DESTINATION = Path(__file__).resolve().parents[1] / "schemas"


def schema_text(model: type[BaseModel]) -> str:
    schema = model.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on drift without writing files")
    args = parser.parse_args()
    stale: list[str] = []
    for name, model in SCHEMAS.items():
        path = DESTINATION / f"{name}.schema.json"
        expected = schema_text(model)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                stale.append(path.name)
        else:
            DESTINATION.mkdir(exist_ok=True)
            path.write_text(expected, encoding="utf-8")
    if stale:
        raise SystemExit(
            "Public schema drift: " + ", ".join(stale) + ". Run scripts/generate_schemas.py."
        )
    print(f"{'Verified' if args.check else 'Generated'} {len(SCHEMAS)} public schemas")


if __name__ == "__main__":
    main()
