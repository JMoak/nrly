from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from nrly.api import capabilities, run_summary
from nrly.contracts import (
    CaseSpec,
    ExperimentSpec,
    ModelSpec,
    NrlyError,
    ProgramSpec,
    WeightReadSpec,
)
from nrly.evaluation import compare_runs, evaluate_run
from nrly.runner import Runner
from nrly.store import Store, read_bundle
from nrly.study_contracts import AttemptRecord, CandidateSpec, StudyRecord, StudySpec


def contains_recorded_evidence(recorded: object, replayed: object) -> bool:
    """Compare all recorded fields while allowing newly added inspection metadata."""
    if isinstance(recorded, dict) and isinstance(replayed, dict):
        return all(
            key in replayed and contains_recorded_evidence(value, replayed[key])
            for key, value in recorded.items()
        )
    if isinstance(recorded, list) and isinstance(replayed, list):
        return len(recorded) == len(replayed) and all(
            contains_recorded_evidence(left, right)
            for left, right in zip(recorded, replayed, strict=True)
        )
    return recorded == replayed


def read_weight_selections(path: Path) -> list[WeightReadSpec]:
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, list):
        raise NrlyError("invalid_request", "read specification must contain an array of selections")
    return [WeightReadSpec.model_validate(value) for value in raw]


def emit(value: Any) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    from nrly.artifact_cli import register_artifact_commands

    result = argparse.ArgumentParser(description="nrly — weight-space programming workbench")
    result.add_argument("--root", type=Path, default=Path(".nrly"), help="local run store")
    result.add_argument("--model-cache", type=Path, help="Hugging Face cache root (HF_HOME)")
    commands = result.add_subparsers(dest="command", required=True)
    register_artifact_commands(commands)
    run = commands.add_parser("run", help="execute a JSON experiment specification")
    run.add_argument("spec", type=Path)
    run.add_argument("--request-id")
    inspect = commands.add_parser(
        "inspect", help="catalog a model and read weights, without inference"
    )
    inspect.add_argument(
        "--model-spec", type=Path, help="ModelSpec JSON; defaults to the tiny fixture"
    )
    inspect.add_argument(
        "--target", action="append", default=[], help="exact parameter name to read"
    )
    inspect.add_argument("--max-elements", type=int, default=16)
    inspect.add_argument("--offset", type=int, default=0, help="flat offset for each target")
    inspect.add_argument("--indices", type=int, nargs="+", help="exact flattened indices")
    inspect.add_argument(
        "--read-spec", type=Path, help="JSON array of rich tensor read specifications"
    )
    for name in ("show", "evaluate"):
        commands.add_parser(name).add_argument("run_id")
    listing = commands.add_parser("list")
    listing.add_argument("--limit", type=int, default=20)
    comparison = commands.add_parser("compare")
    comparison.add_argument("baseline_id")
    comparison.add_argument("candidate_id")
    export = commands.add_parser("export", help="export a verified, portable run bundle")
    export.add_argument("run_id")
    export.add_argument("destination", type=Path)
    replay = commands.add_parser("replay", help="verify and replay a bundle against its exact base")
    replay.add_argument("bundle", type=Path)
    demo = commands.add_parser(
        "demo", help="run a real Torch engineering fixture repair and restoration"
    )
    demo.add_argument("--export", type=Path, help="export the repaired run bundle")
    commands.add_parser(
        "doctor", help="check installed packages and local accelerator availability"
    )
    schema = commands.add_parser("schema", help="print a versioned public JSON Schema")
    schema.add_argument(
        "kind",
        choices=[
            "experiment",
            "model",
            "program",
            "study",
            "candidate",
            "attempt",
            "study-record",
            "weight-read",
            "diagnostics",
            "task-adapter",
            "comparator",
            "edit-revision",
            "patch",
        ],
        default="experiment",
        nargs="?",
    )
    commands.add_parser("mcp", help="start the local stdio MCP server")
    diagnostics = commands.add_parser(
        "diagnostics", help="page recorded continuation probabilities"
    )
    diagnostics.add_argument("run_id")
    diagnostics.add_argument("case_id")
    diagnostics.add_argument("--offset", type=int, default=0)
    diagnostics.add_argument("--limit", type=int, default=8)
    diagnostics.add_argument("--top-k-offset", type=int, default=0)
    diagnostics.add_argument("--top-k-limit", type=int, default=3)
    capture = commands.add_parser("capture", help="page recorded activation values")
    capture.add_argument("run_id")
    capture.add_argument("case_id")
    capture.add_argument("target")
    capture.add_argument("--offset", type=int, default=0)
    capture.add_argument("--limit", type=int, default=64)
    report = commands.add_parser("report", help="render an offline experiment review")
    report.add_argument("run_id")
    report.add_argument("--baseline")
    report.add_argument("--output", type=Path, required=True)
    study = commands.add_parser("study", help="controlled candidate discovery and one-shot holdout")
    actions = study.add_subparsers(dest="study_command", required=True)
    create = actions.add_parser("create", help="operator-only provisioning of sealed test cases")
    create.add_argument("spec", type=Path)
    create.add_argument("--holdout", type=Path, required=True)
    create.add_argument("--request-id")
    actions.add_parser("list")
    for name in ("show", "attempts", "events"):
        action = actions.add_parser(name)
        action.add_argument("study_id")
        if name != "show":
            action.add_argument("--offset", type=int, default=0)
            action.add_argument("--limit", type=int, default=20)
    submit = actions.add_parser(
        "submit", help="register a candidate and await development assessment"
    )
    submit.add_argument("study_id")
    submit.add_argument("candidate", type=Path)
    submit.add_argument("--request-id")
    final = actions.add_parser(
        "finalize", help="freeze a nominee and await the final holdout result"
    )
    final.add_argument("study_id")
    final.add_argument("attempt_id")
    context = actions.add_parser(
        "export-context", help="export a public study handoff in CorePack's .ai layout"
    )
    context.add_argument("study_id")
    context.add_argument("destination", type=Path)
    return result


def execute(args: argparse.Namespace) -> int:
    if args.command in {"patch", "edit"}:
        from nrly.artifact_cli import execute_artifact_command

        value, code = execute_artifact_command(args)
        emit(value)
        return code
    if args.command in {"diagnostics", "capture"}:
        from nrly.evidence import capture_page, diagnostics_page

        record = Store(args.root).get(args.run_id)
        if args.command == "diagnostics":
            emit(
                diagnostics_page(
                    record,
                    args.case_id,
                    args.offset,
                    args.limit,
                    args.top_k_offset,
                    args.top_k_limit,
                )
            )
        else:
            emit(capture_page(record, args.case_id, args.target, args.offset, args.limit))
        return 0
    if args.command == "study":
        return execute_study(args)
    if args.command == "schema":
        from nrly.contracts import DiagnosticSpec, TaskAdapterSpec
        from nrly.deployment import PatchManifest
        from nrly.lineage import EditRevision
        from nrly.study_contracts import ComparatorSpec

        models: dict[str, type[BaseModel]] = {
            "diagnostics": DiagnosticSpec,
            "task-adapter": TaskAdapterSpec,
            "comparator": ComparatorSpec,
            "edit-revision": EditRevision,
            "patch": PatchManifest,
            "experiment": ExperimentSpec,
            "model": ModelSpec,
            "program": ProgramSpec,
            "study": StudySpec,
            "candidate": CandidateSpec,
            "attempt": AttemptRecord,
            "study-record": StudyRecord,
            "weight-read": WeightReadSpec,
        }
        emit(models[args.kind].model_json_schema())
        return 0
    if args.command == "doctor":
        data = capabilities()
        try:
            import torch

            data["accelerator"] = {
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "devices": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ],
            }
        except ImportError:
            data["accelerator"] = {
                "cuda_available": False,
                "reason": "install nrly[torch] or nrly[hf]",
            }
        data["python"] = sys.version
        data["installed_nrly"] = importlib.metadata.version("nrly")
        emit(data)
        return 0
    if args.command == "mcp":
        from nrly.api import Application
        from nrly.mcp import create_server

        app = Application(args.root, args.model_cache)
        try:
            create_server(app).run(transport="stdio")
        finally:
            app.close()
        return 0
    if args.command in {"show", "list", "evaluate", "compare", "export", "report"}:
        store = Store(args.root)
        if args.command == "show":
            emit(store.get(args.run_id))
        elif args.command == "list":
            emit([run_summary(record) for record in store.list(args.limit)])
        elif args.command == "evaluate":
            emit(evaluate_run(store.get(args.run_id)))
        elif args.command == "compare":
            emit(compare_runs(store.get(args.baseline_id), store.get(args.candidate_id)))
        elif args.command == "export":
            emit({"bundle": str(store.export(args.run_id, args.destination))})
        elif args.command == "report":
            from nrly.review import write_run_report

            emit(
                {
                    "report": str(
                        write_run_report(
                            store.get(args.run_id),
                            args.output,
                            store.get(args.baseline) if args.baseline else None,
                        )
                    )
                }
            )
        return 0
    with Runner(args.root, args.model_cache) as runner:
        if args.command == "demo":
            cases = [
                CaseSpec(
                    id=f"sum-{index}",
                    input=json.dumps(pair),
                    expected=str(sum(pair)),
                    scorer="numeric",
                    split="test",
                    role="regression" if index == 0 else "target",
                )
                for index, pair in enumerate(([0, 0], [2, 4], [1, 3], [5, 7], [10, 2], [3, 9]))
            ]
            baseline_spec = ExperimentSpec(name="Tiny sum baseline", cases=cases)
            baseline = runner.wait(runner.submit(baseline_spec).id)
            candidate_spec = baseline_spec.model_copy(
                update={
                    "name": "Tiny sum weight repair",
                    "program": ProgramSpec(kind="weight_scale", target="output.weight", factor=2),
                }
            )
            candidate = runner.wait(runner.submit(candidate_spec).id)
            restored = runner.wait(runner.submit(baseline_spec).id)
            comparison = compare_runs(baseline, candidate)
            outputs_restored = (
                baseline.status == restored.status == "succeeded"
                and [sample.output for sample in baseline.samples]
                == [sample.output for sample in restored.samples]
                and baseline.model_identity == restored.model_identity
            )
            data = {
                "scope": "engineering_fixture",
                "baseline": baseline.id,
                "candidate": candidate.id,
                "restoration_run": restored.id,
                "base_restored": outputs_restored,
                "comparison": comparison.model_dump(mode="json"),
            }
            if args.export:
                data["bundle"] = str(runner.store.export(candidate.id, args.export))
            emit(data)
            return (
                0
                if (
                    comparison.comparable
                    and outputs_restored
                    and comparison.verdict == "observed_improvement"
                    and comparison.candidate.summary.accuracy == 1
                )
                else 1
            )
        if args.command == "inspect":
            model = (
                ModelSpec.model_validate_json(args.model_spec.read_bytes())
                if args.model_spec
                else ModelSpec()
            )
            if args.read_spec and (args.target or args.offset or args.indices):
                raise NrlyError("invalid_request", "use either --read-spec or target selectors")
            if not args.target and (args.offset or args.indices):
                raise NrlyError("invalid_request", "offset and indices require --target")
            reads = (
                read_weight_selections(args.read_spec)
                if args.read_spec
                else [
                    WeightReadSpec(
                        target=target,
                        max_elements=args.max_elements,
                        offset=args.offset,
                        indices=args.indices or [],
                    )
                    for target in args.target
                ]
            )
            spec = ExperimentSpec(
                name="Model inspection",
                purpose="inspection",
                model=model,
                weight_reads=reads,
            )
            record = runner.wait(runner.submit(spec).id)
            emit(record)
            return 0 if record.status == "succeeded" else 1
        if args.command == "replay":
            original = read_bundle(args.bundle)
            if original.model_identity is None:
                raise NrlyError(
                    "missing_identity", "cannot replay a bundle without a verified model identity"
                )
            spec = original.spec.model_copy(
                update={
                    "program": original.spec.program.model_copy(
                        update={"expected_base": original.model_identity.fingerprint}
                    )
                }
            )
            record = runner.wait(runner.submit(spec, parent_run=original.id).id)
            identity_matches = record.model_identity == original.model_identity
            succeeded = record.status == original.status == "succeeded"
            outputs_match = (
                (
                    succeeded
                    and identity_matches
                    and [sample.output for sample in record.samples]
                    == [sample.output for sample in original.samples]
                )
                if spec.purpose == "experiment"
                else None
            )
            inspection_matches = (
                (
                    succeeded
                    and identity_matches
                    and record.inspection is not None
                    and original.inspection is not None
                    and contains_recorded_evidence(
                        original.inspection.model_dump(mode="json"),
                        record.inspection.model_dump(mode="json"),
                    )
                )
                if spec.purpose == "inspection" or original.inspection is not None
                else None
            )
            original_samples = [
                sample.model_dump(mode="json", exclude={"elapsed_ms"})
                for sample in original.samples
            ]
            replayed_samples = [
                sample.model_dump(mode="json", exclude={"elapsed_ms"}) for sample in record.samples
            ]
            sample_evidence_matches = (
                succeeded
                and identity_matches
                and contains_recorded_evidence(original_samples, replayed_samples)
            )
            weight_evidence_matches = (
                succeeded
                and identity_matches
                and contains_recorded_evidence(
                    [sample["weight_changes"] for sample in original_samples],
                    [sample["weight_changes"] for sample in replayed_samples],
                )
            )
            activation_evidence_matches = (
                succeeded
                and identity_matches
                and contains_recorded_evidence(
                    [sample["captures"] for sample in original_samples],
                    [sample["captures"] for sample in replayed_samples],
                )
            )
            diagnostic_evidence_matches = (
                succeeded
                and identity_matches
                and contains_recorded_evidence(
                    [sample.get("diagnostics") for sample in original_samples],
                    [sample.get("diagnostics") for sample in replayed_samples],
                )
            )
            replay_matches = (
                sample_evidence_matches
                and outputs_match is not False
                and inspection_matches is not False
            )
            emit(
                {
                    "run": record.model_dump(mode="json"),
                    "evaluation": evaluate_run(record).model_dump(mode="json"),
                    "replay": {
                        "original_run": original.id,
                        "outputs_match": outputs_match,
                        "inspection_matches": inspection_matches,
                        "identity_matches": identity_matches,
                        "sample_evidence_matches": sample_evidence_matches,
                        "weight_evidence_matches": weight_evidence_matches,
                        "activation_evidence_matches": activation_evidence_matches,
                        "diagnostic_evidence_matches": diagnostic_evidence_matches,
                        "replay_matches": replay_matches,
                    },
                }
            )
            return 0 if replay_matches else 1
        spec = ExperimentSpec.model_validate_json(args.spec.read_bytes())
        record = runner.wait(runner.submit(spec, args.request_id).id)
        emit(
            {
                "run": record.model_dump(mode="json"),
                "evaluation": evaluate_run(record).model_dump(mode="json"),
            }
        )
        return 0 if record.status == "succeeded" else 1


def main() -> None:
    # Machine-readable pipes must use the same encoding on Windows and POSIX,
    # regardless of the console code page or PYTHONIOENCODING inherited by a caller.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        code = execute(parser().parse_args())
    except (NrlyError, ValidationError, OSError, json.JSONDecodeError) as error:
        code = 2
        detail = (
            error.as_info().model_dump()
            if isinstance(error, NrlyError)
            else {
                "code": "invalid_request",
                "message": str(error),
            }
        )
        print(json.dumps({"error": detail}, ensure_ascii=False), file=sys.stderr)
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


def execute_study(args: argparse.Namespace) -> int:
    from nrly.studies import StudyEngine, StudyReader
    from nrly.study_contracts import CandidateSpec, StudySpec

    if args.study_command in {"list", "show", "attempts", "events", "export-context"}:
        reader = StudyReader(args.root)
        if args.study_command == "export-context":
            from nrly.context_pack import export_context_pack

            emit(
                {"context_pack": str(export_context_pack(reader, args.study_id, args.destination))}
            )
        elif args.study_command == "list":
            from nrly.api import study_summary

            emit([study_summary(record) for record in reader.list()])
        elif args.study_command == "show":
            emit(reader.get(args.study_id))
        elif args.study_command == "attempts":
            emit(
                [
                    record.model_dump(mode="json")
                    for record in reader.attempts(
                        args.study_id,
                        args.offset,
                        args.limit,
                    )
                ]
            )
        else:
            emit(
                [
                    record.model_dump(mode="json")
                    for record in reader.events(
                        args.study_id,
                        args.offset,
                        args.limit,
                    )
                ]
            )
        return 0
    with Runner(args.root, args.model_cache) as runner:
        engine = StudyEngine(runner)
        try:
            if args.study_command == "create":
                spec = StudySpec.model_validate_json(args.spec.read_bytes())
                raw = json.loads(args.holdout.read_bytes())
                if not isinstance(raw, list):
                    raise NrlyError(
                        "invalid_holdout", "holdout file must contain an array of cases"
                    )
                record = engine.create(
                    spec,
                    [CaseSpec.model_validate(case) for case in raw],
                    args.request_id,
                )
                emit(record)
                return 0
            if args.study_command == "submit":
                candidate = CandidateSpec.model_validate_json(args.candidate.read_bytes())
                attempt = engine.submit_candidate(args.study_id, candidate, args.request_id)
                attempt = engine.wait(args.study_id, attempt.id)
                emit(attempt)
                return 0 if attempt.status == "succeeded" else 1
            engine.finalize(args.study_id, args.attempt_id)
            final = engine.wait_final(args.study_id)
            emit(final)
            return 0 if final.status == "completed" and final.final_error_code is None else 1
        finally:
            engine.close()


if __name__ == "__main__":
    main()
