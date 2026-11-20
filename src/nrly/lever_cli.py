"""CLI adapters for the lever map; every action calls a validated application operation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nrly.artifacts import MAX_MANIFEST_BYTES, read_local_file
from nrly.contracts import CaseSpec, NrlyError


def register_lever_commands(commands: Any) -> None:
    lever = commands.add_parser(
        "lever", help="derive, measure, referee and map tested sites in a model's weights"
    )
    actions = lever.add_subparsers(dest="lever_command", required=True)
    derive = actions.add_parser("derive", help="run a recipe and pencil the result on the map")
    derive.add_argument("request", type=Path)
    dial = actions.add_parser("dial", help="measure strengths on development cases")
    dial.add_argument("entry")
    dial.add_argument("request", type=Path)
    nominate = actions.add_parser(
        "nominate", help="operator override: choose a tested strength the dial did not"
    )
    nominate.add_argument("entry")
    nominate.add_argument("strength", type=float)
    referee = actions.add_parser(
        "referee", help="operator-only: held-out study that inks or strikes the lever"
    )
    referee.add_argument("entry")
    referee.add_argument("request", type=Path)
    referee.add_argument("holdout", type=Path)
    referee.add_argument(
        "--replication", action="store_true", help="repeat an existing verdict on this runtime"
    )
    legend = actions.add_parser("legend", help="name the direction from an inherited dictionary")
    legend.add_argument("entry")
    legend.add_argument("request", type=Path)
    legend.add_argument("--cache", type=Path, help="decoder cache directory")
    legend.add_argument("--local-file", type=Path, help="read a local dictionary file")
    for name in ("show", "history"):
        actions.add_parser(name).add_argument("entry")
    listing = actions.add_parser("list", help="latest revision of every lever")
    listing.add_argument("--weights", help="only levers for this base weights digest")
    listing.add_argument("--behavior")
    listing.add_argument("--status", choices=["pencil", "inked", "struck"])
    chart = actions.add_parser("map", help="write the offline map page")
    chart.add_argument("destination", type=Path)
    chart.add_argument("--weights", help="only levers for this base weights digest")
    chart.add_argument("--layers", type=int, help="model depth, to show unsurveyed layers")
    chart.add_argument("--title", default="nrly lever map")
    page = actions.add_parser("page", help="write one lever's offline datasheet")
    page.add_argument("entry")
    page.add_argument("destination", type=Path)
    export = actions.add_parser("export", help="write a portable lever bundle")
    export.add_argument("entry")
    export.add_argument("destination", type=Path)
    actions.add_parser("import", help="verify and add a lever bundle").add_argument(
        "bundle", type=Path
    )


def _request(path: Path) -> bytes:
    return read_local_file(path.parent, path.name, MAX_MANIFEST_BYTES)


def _write_page(destination: Path, content: str) -> dict[str, object]:
    if destination.exists():
        raise NrlyError("destination_exists", "choose a new file; pages are never overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    return {"page": str(destination), "characters": len(content)}


def _summary(entry: Any) -> dict[str, object]:
    return {
        "id": entry.id,
        "name": entry.name,
        "status": entry.status,
        "behavior": entry.behavior,
        "weights_sha256": entry.base.weights_sha256,
        "model_id": entry.base.model_id,
        "strength": entry.strength,
        "selectivity": next(
            (point.selectivity for point in entry.dial.points if point.strength == entry.strength),
            None,
        )
        if entry.dial is not None
        else None,
        "verdict": entry.verdict.assessment.verdict if entry.verdict else None,
        "replications": len(entry.replications),
        "contraindications": len(entry.contraindications),
        "parent_id": entry.parent_id,
    }


def execute_lever_command(args: argparse.Namespace) -> tuple[Any, int]:
    from nrly.lever_app import (
        DeriveRequest,
        DialRequest,
        LegendRequest,
        RefereeRequest,
        derive_lever,
        dial_lever,
        legend_lever,
        lever_map,
        nominate_lever,
        referee_lever,
    )

    action = args.lever_command
    if action == "nominate":
        return nominate_lever(args.root, args.entry, args.strength), 0
    if action == "derive":
        request = DeriveRequest.model_validate_json(_request(args.request))
        return derive_lever(args.root, request, args.model_cache), 0
    if action == "dial":
        value = dial_lever(
            args.root,
            args.entry,
            DialRequest.model_validate_json(_request(args.request)),
            args.model_cache,
        )
        return value, 0
    if action == "referee":
        raw = json.loads(_request(args.holdout))
        if not isinstance(raw, list):
            raise NrlyError("invalid_holdout", "holdout file must contain an array of cases")
        value = referee_lever(
            args.root,
            args.entry,
            RefereeRequest.model_validate_json(_request(args.request)),
            [CaseSpec.model_validate(case) for case in raw],
            args.model_cache,
            replication=args.replication,
        )
        return value, 0 if value["refereed"] and value["final_error_code"] is None else 1
    if action == "legend":
        value = legend_lever(
            args.root,
            args.entry,
            LegendRequest.model_validate_json(_request(args.request)),
            cache=args.cache,
            local_file=args.local_file,
        )
        return value, 0
    levers = lever_map(args.root)
    if action == "show":
        return levers.get(args.entry), 0
    if action == "history":
        return [_summary(entry) for entry in levers.history(args.entry)], 0
    if action == "list":
        entries = levers.heads(args.weights)
        if args.behavior:
            entries = [entry for entry in entries if entry.behavior == args.behavior]
        if args.status:
            entries = [entry for entry in entries if entry.status == args.status]
        return [_summary(entry) for entry in entries], 0
    if action in {"map", "page"}:
        from nrly.lever_report import render_lever, render_map

        if action == "page":
            return _write_page(args.destination, render_lever(levers.get(args.entry))), 0
        content = render_map(levers.heads(args.weights), title=args.title, layers=args.layers)
        return _write_page(args.destination, content), 0
    if action == "export":
        return {"bundle": str(levers.export(args.entry, args.destination))}, 0
    return _summary(levers.import_bundle(args.bundle)), 0
