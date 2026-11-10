"""Operator-side fixture provisioning; no sealed cases are sent over MCP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nrly.contracts import BudgetSpec, CaseSpec, ProgramSpec
from nrly.runner import Runner
from nrly.studies import StudyEngine
from nrly.study_contracts import CandidateSpec, StudyBudget, StudySpec


def main() -> None:
    runner = Runner(Path(sys.argv[1]))
    engine = StudyEngine(runner)
    try:
        spec = StudySpec(
            name="MCP controlled study",
            development_cases=[
                CaseSpec(id="repair", input="[22,44]", expected="66", scorer="numeric"),
                CaseSpec(
                    id="preserve", input="[0,0]", expected="0", scorer="numeric", role="regression"
                ),
            ],
            controls=[
                CandidateSpec(
                    name="negative",
                    program=ProgramSpec(kind="weight_scale", target="output.weight", factor=0),
                )
            ],
            mutable_targets=["output.weight"],
            run_budget=BudgetSpec(wall_seconds=30, max_cases=2, max_output_tokens=64),
            budget=StudyBudget(max_attempts=3),
        )
        holdout = [
            CaseSpec(
                id="sealed-target-canary",
                input="[900001,900003]",
                expected="1800004",
                scorer="numeric",
                split="test",
            ),
            CaseSpec(
                id="sealed-regression-canary",
                input="[0.0,0.0]",
                expected="0",
                scorer="numeric",
                role="regression",
                split="test",
            ),
        ]
        record = engine.create(spec, holdout)
        print(json.dumps({"study_id": record.id}))
    finally:
        engine.close()
        runner.close()


if __name__ == "__main__":
    main()
