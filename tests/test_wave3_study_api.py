from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nrly.api import Application, bounded
from nrly.contracts import CaseSpec, ModelSpec, NrlyError
from nrly.study_contracts import StudySpec


def test_full_development_case_and_comparator_are_recoverable_but_holdout_is_not(
    tmp_path: Path,
) -> None:
    app = Application(tmp_path / "runs")
    try:
        spec = StudySpec.model_validate(
            {
                "schema_version": "0.2",
                "name": "Large public evidence pagination",
                "model": ModelSpec(
                    backend="huggingface", model_id="operator-fixture", revision="a" * 40
                ),
                "seeds": [11, 29],
                "comparators": [{"name": "prompt", "kind": "prompt", "prefix": "x" * 5000}],
                "development_cases": [
                    {
                        "id": "chat",
                        "messages": [{"role": "user", "content": "hello 🧠" * 1000}],
                        "adapter": {
                            "kind": "instruction_constraints",
                            "required": [str(i) + "a" * 999 for i in range(10)],
                            "forbidden": [str(i) + "b" * 999 for i in range(10)],
                        },
                    },
                    {"id": "local", "input": "retain", "expected": "yes", "role": "regression"},
                ],
            }
        )
        study = app.studies.create(
            spec,
            [
                CaseSpec(
                    id="sealed", input="private question", expected="private answer", split="test"
                ),
                CaseSpec(
                    id="sealed-local",
                    input="private regression",
                    expected="no",
                    split="test",
                    role="regression",
                ),
            ],
        )
        summary = bounded(app.get_study_cases(study.id, limit=1))
        assert summary["cases"][0]["adapter_kind"] == "instruction_constraints"
        assert "adapter" not in summary["cases"][0]
        assert "messages" not in summary["cases"][0]
        content = ""
        offset = 0
        while True:
            page = bounded(app.get_study_case_spec(study.id, "chat", offset=offset))
            content += page["content"]
            if page["next_offset"] is None:
                break
            offset = page["next_offset"]
        assert hashlib.sha256(content.encode()).hexdigest() == page["sha256"]
        assert json.loads(content) == spec.development_cases[0].model_dump(mode="json")
        assert "private question" not in content
        message = bounded(
            app.get_study_case_messages(study.id, "chat", content_offset=6, content_limit=2)
        )
        assert message["messages"][0]["content"] == "🧠h"
        for getter in (app.get_study_case_spec, app.get_study_case_messages):
            with pytest.raises(NrlyError, match="development case"):
                getter(study.id, "sealed")
        first = bounded(app.get_study_comparator(study.id, "prompt"))
        second = bounded(app.get_study_comparator(study.id, "prompt", offset=first["next_offset"]))
        parsed = json.loads(first["content"] + second["content"])
        assert parsed["registration"]["prefix"] == "x" * 5000
        assert parsed["model_identity"] is None
        assert (
            bounded(app.get_study(study.id))["comparators"][0]["registration_digest"]
            == parsed["registration_digest"]
        )
    finally:
        app.close()
