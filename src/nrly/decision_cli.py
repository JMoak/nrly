"""CLI adapters for the same validated Python qualification operations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from nrly.artifacts import MAX_MANIFEST_BYTES, read_local_file, write_immutable
from nrly.contracts import Contract, ModelSpec, NrlyError, ProgramSpec, canonical_json
from nrly.decision_contracts import DecisionFreeze, DecisionObservation, DecisionProtocol
from nrly.decision_handoff import prepare_shadow_handoff
from nrly.decision_qualification import (
    assess_decisions,
    read_decision_run,
    run_decision_study,
    save_decision_run,
)
from nrly.decision_report import render_decision_report


class DecisionExperiment(Contract):
    model: ModelSpec
    program: ProgramSpec
    protocol: DecisionProtocol


def register_decision_commands(commands: Any) -> None:
    command = commands.add_parser("decision", help="compare binary decision weight edits")
    actions = command.add_subparsers(dest="decision_action", required=True)
    run = actions.add_parser("run", help="run a pinned local causal-model comparison")
    run.add_argument("spec", type=Path)
    run.add_argument("destination", type=Path)
    report = actions.add_parser("report", help="export aggregate-only offline HTML review")
    report.add_argument("bundle", type=Path)
    report.add_argument("destination", type=Path)


def execute_decision_command(args: argparse.Namespace) -> dict[str, object]:
    if args.decision_action == "report":
        run = read_decision_run(args.bundle)
    else:
        if args.model_cache:
            os.environ["HF_HOME"] = str(args.model_cache.resolve())
        from nrly.backends.decision import CausalDecisionBackend
        from nrly.backends.huggingface import HuggingFaceBackend

        spec = DecisionExperiment.model_validate_json(
            read_local_file(args.spec.parent, args.spec.name, MAX_MANIFEST_BYTES),
        )
        if spec.model.backend != "huggingface":
            raise NrlyError("unsupported_operation", "decision CLI requires a causal HF model")
        if args.destination.exists():
            raise NrlyError("destination_exists", "use a new directory to retain every attempt")
        count = 0

        def journal(observation: DecisionObservation) -> None:
            nonlocal count
            write_immutable(
                args.destination / "private-journal",
                f"{count:04d}.json",
                canonical_json(observation).encode(),
            )
            count += 1

        write_immutable(args.destination, "private-spec.json", canonical_json(spec).encode())

        def freeze_record(value: DecisionFreeze) -> None:
            write_immutable(args.destination, "freeze.json", canonical_json(value).encode())

        backend = HuggingFaceBackend(spec.model)
        with backend:
            run = run_decision_study(
                CausalDecisionBackend(backend),
                spec.protocol,
                spec.program,
                record=journal,
                on_freeze=freeze_record,
            )
        save_decision_run(args.destination, run)
    assessment = assess_decisions(run)
    write_immutable(args.destination, "review.html", render_decision_report(assessment).encode())
    write_immutable(args.destination, "assessment.json", canonical_json(assessment).encode())
    if assessment.status == "complete":
        write_immutable(
            args.destination,
            "shadow.json",
            canonical_json(prepare_shadow_handoff(run)).encode(),
        )
    return {"assessment": assessment.model_dump(mode="json"), "destination": str(args.destination)}
