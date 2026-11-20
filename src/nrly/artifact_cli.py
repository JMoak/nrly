"""CLI adapters for reusable weight programs; numerical work stays in the kernel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nrly.contracts import ExperimentSpec, NrlyError, PortableBase, ProgramSpec
from nrly.deployment import (
    PortablePatchManifest,
    bind_operations,
    export_patch,
    export_portable_patch,
    read_any_patch,
)
from nrly.evaluation import evaluate_run
from nrly.lineage import EditStore, compare_sequential, conflict_footprint, weight_operations
from nrly.runner import Runner
from nrly.store import Store


def register_artifact_commands(commands: Any) -> None:
    patch = commands.add_parser("patch", help="export, inspect or execute a portable weight patch")
    actions = patch.add_subparsers(dest="patch_command", required=True)
    export = actions.add_parser(
        "export", help="package the declarative program from a completed run"
    )
    export.add_argument("run_id")
    export.add_argument("destination", type=Path)
    export.add_argument(
        "--portable", action="store_true",
        help="pin the base weights instead of this machine's runtime fingerprint",
    )
    inspect = actions.add_parser("inspect", help="verify a patch without loading a model")
    inspect.add_argument("package", type=Path)
    run = actions.add_parser("run", help="run an experiment using a verified patch program")
    run.add_argument("package", type=Path)
    run.add_argument("spec", type=Path)
    run.add_argument("--request-id")
    edit = commands.add_parser("edit", help="record and compare immutable cumulative edits")
    actions = edit.add_subparsers(dest="edit_command", required=True)
    create = actions.add_parser("create", help="append a pinned delta to an optional parent")
    create.add_argument("name")
    create.add_argument("program", type=Path)
    create.add_argument("--parent")
    for name in ("show", "history", "select"):
        action = actions.add_parser(name)
        action.add_argument("revision")
        if name == "select":
            action.add_argument("--ancestor", help="omit to select the unchanged original")
    compare = actions.add_parser("compare", help="compare original, A and A+B with edit ancestry")
    for name in ("original_run", "parent_run", "candidate_run", "parent_revision", "revision"):
        compare.add_argument(name)
    export = actions.add_parser("export", help="package an edit with identity from a measured run")
    export.add_argument("revision")
    export.add_argument("run_id")
    export.add_argument("destination", type=Path)


def execute_artifact_command(args: argparse.Namespace) -> tuple[Any, int]:
    if args.command == "edit":
        edits = EditStore(args.root / "edits")
        if args.edit_command == "create":
            program = ProgramSpec.model_validate_json(args.program.read_bytes())
            return edits.create(args.name, program, args.parent), 0
        if args.edit_command == "history":
            return [item.model_dump(mode="json") for item in edits.ancestry(args.revision)], 0
        if args.edit_command == "select":
            return edits.select_ancestor(args.revision, args.ancestor), 0
        revision = edits.get(args.revision)
        if args.edit_command == "show":
            return {
                "revision": revision.model_dump(mode="json"),
                "footprint": conflict_footprint(revision).model_dump(mode="json"),
            }, 0
        store = Store(args.root)
        if args.edit_command == "compare":
            report = compare_sequential(
                store.get(args.original_run),
                store.get(args.parent_run),
                store.get(args.candidate_run),
                edits.get(args.parent_revision),
                revision,
            )
            return report, 0 if report.comparable else 1
        run = store.get(args.run_id)
        if run.status != "succeeded" or run.model_identity is None:
            raise NrlyError("incomplete_run", "patch export requires a completed measured run")
        if run.spec.program != revision.program:
            raise NrlyError("lineage_mismatch", "measured run must execute the exported revision")
        return export_patch(args.destination, revision.program, run.model_identity, revision), 0
    if args.patch_command == "inspect":
        return read_any_patch(args.package), 0
    if args.patch_command == "export":
        run = Store(args.root).get(args.run_id)
        if run.status != "succeeded" or run.model_identity is None or not run.samples:
            raise NrlyError("incomplete_run", "patch export requires a completed inference run")
        if args.portable:
            operations = weight_operations(run.spec.program)
            if not operations:
                raise NrlyError(
                    "invalid_program", "a portable patch requires executed weight operations"
                )
            base = PortableBase.from_identity(run.model_identity)
            return export_portable_patch(args.destination, operations, base), 0
        program = ProgramSpec.model_validate(
            {
                **run.spec.program.model_dump(),
                "expected_base": run.model_identity.fingerprint,
            }
        )
        return export_patch(args.destination, program, run.model_identity), 0
    manifest = read_any_patch(args.package)
    spec = ExperimentSpec.model_validate_json(args.spec.read_bytes())
    if spec.purpose != "experiment":
        raise NrlyError("invalid_request", "patch execution requires an inference experiment")
    if isinstance(manifest, PortablePatchManifest):
        return _run_portable(args, manifest, spec)
    if (
        spec.model.backend != manifest.identity.backend
        or spec.model.model_id != manifest.identity.model_id
        or spec.model.revision != manifest.identity.revision
    ):
        raise NrlyError("base_mismatch", "experiment model differs from the patch model")
    spec = ExperimentSpec.model_validate(
        {
            **spec.model_dump(),
            "program": manifest.program.model_dump(),
        }
    )
    with Runner(args.root, args.model_cache) as runner:
        run = runner.wait(runner.submit(spec, args.request_id).id)
    return {
        "patch_id": manifest.id,
        "run": run.model_dump(mode="json"),
        "evaluation": evaluate_run(run).model_dump(mode="json"),
    }, 0 if run.status == "succeeded" else 1


def _run_portable(
    args: argparse.Namespace, manifest: PortablePatchManifest, spec: ExperimentSpec,
) -> tuple[Any, int]:
    """Bind a portable patch to this runtime, then run it as an ordinary pinned program.

    The package names weights, not a machine, so the local fingerprint is learned from
    an inspection of the experiment's own model. The experiment is submitted only after
    that recorded identity matches the package base.
    """
    base = manifest.base
    if (spec.model.backend, spec.model.model_id, spec.model.revision) != (
        base.backend, base.model_id, base.revision,
    ):
        raise NrlyError("base_mismatch", "experiment model differs from the patch model")
    with Runner(args.root, args.model_cache) as runner:
        # The experiment's budget also bounds this load of the same model.
        inspection = runner.wait(
            runner.submit(
                ExperimentSpec(
                    name=f"Local identity for patch {manifest.id[:12]}",
                    purpose="inspection", model=spec.model, budget=spec.budget,
                )
            ).id
        )
        identity = inspection.model_identity
        if inspection.status != "succeeded" or identity is None:
            failure = inspection.error
            raise NrlyError(
                failure.code if failure else "incomplete_run",
                f"inspection run {inspection.id} did not establish the local model identity"
                + (f": {failure.message}" if failure else ""),
            )
        program = bind_operations(manifest.operations, base, identity)
        bound = ExperimentSpec.model_validate(
            {**spec.model_dump(), "program": program.model_dump()}
        )
        run = runner.wait(runner.submit(bound, args.request_id).id)
    return {
        "patch_id": manifest.id,
        "bound_fingerprint": identity.fingerprint,
        "inspection_run": inspection.id,
        "run": run.model_dump(mode="json"),
        "evaluation": evaluate_run(run).model_dump(mode="json"),
    }, 0 if run.status == "succeeded" else 1
