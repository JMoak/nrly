"""The lever map: immutable, content-addressed revisions of tested sites in one checkpoint.

Nothing here writes a status. ``seal`` derives it from the attached verdict, a dial only
selects a strength, and every change is a new revision that names its parent, so nulls,
refutations, and superseded dials stay readable.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from nrly.artifacts import (
    MAX_MANIFEST_BYTES,
    TensorArtifact,
    TensorArtifactStore,
    checked_handle,
    read_local_file,
    write_immutable,
)
from nrly.contracts import (
    Contract,
    GenerationSpec,
    NrlyError,
    PortableBase,
    WeightOperationSpec,
    canonical_json,
    digest,
)
from nrly.evaluation.reports import EvaluationReport
from nrly.evaluation.statistics import paired_inference
from nrly.lever_contracts import (
    Contraindication,
    Derivation,
    Dial,
    DosePoint,
    Legend,
    LeverEntry,
    LeverStatus,
    RefereeVerdict,
    StrengthWindow,
    lever_name,
    operations_digest,
    seal,
)
from nrly.lineage import weight_operations
from nrly.study_contracts import CandidateSpec, InferenceRule, StudyRecord, study_seeds

MAX_ENTRY_BYTES = MAX_MANIFEST_BYTES
MAX_MAP_ENTRIES = 10_000
BUNDLE_ENTRY = "lever.json"
BUNDLE_TENSORS = "tensors"
_ENTRY_FILE = re.compile(r"([0-9a-f]{64})\.json")
_IDENTITY = ("base", "behavior", "sites", "name", "derivation")
_Role = Literal["target", "regression"]


def _reason(error: ValueError) -> str:
    """First validation message only: full pydantic errors echo unbounded input."""
    if isinstance(error, ValidationError):
        first = error.errors(include_url=False, include_context=False, include_input=False)[0]
        place = ".".join(str(part) for part in first["loc"])
        return (f"{place}: {first['msg']}" if place else first["msg"])[:300]
    return str(error)[:300]


def _require_coherent(entry: LeverEntry) -> None:
    """What an entry's edit and dial must agree on beyond the sealed contract.

    The contract already ties every verdict to the entry's operations and base.
    """
    if entry.dial is None:
        return
    if entry.dial.nominated is None:
        if entry.operations:
            raise NrlyError(
                "invalid_lever", "a dial that nominates no strength records no operations"
            )
        return
    tested = {point.strength: point.operations_digest for point in entry.dial.points}
    if tested.get(entry.dial.nominated) != operations_digest(entry.operations):
        raise NrlyError(
            "invalid_lever", "operations must be the edit tested at the nominated strength"
        )


def _seal(content: Mapping[str, object]) -> LeverEntry:
    try:
        entry = seal(dict(content))
    except ValueError as error:
        raise NrlyError("invalid_lever", f"invalid lever entry: {_reason(error)}") from error
    _require_coherent(entry)
    return entry


def _plain(records: Sequence[Contract]) -> list[dict[str, object]]:
    return [record.model_dump(mode="json") for record in records]


def _verify_link(parent: LeverEntry, child: LeverEntry) -> None:
    """A revision adds to the record; it never rewrites identity or refereed evidence."""
    if child.parent_id != parent.id or any(
        getattr(child, field) != getattr(parent, field) for field in _IDENTITY
    ):
        raise NrlyError(
            "lineage_mismatch",
            "a revision cannot change a lever's base, behavior, sites, name, or derivation",
        )
    if parent.verdict is not None and (
        child.verdict != parent.verdict
        or child.operations != parent.operations
        or child.strength != parent.strength
        or child.dial != parent.dial
    ):
        raise NrlyError(
            "lineage_mismatch", "a revision cannot change a refereed edit or its verdict"
        )
    if (
        child.replications[: len(parent.replications)] != parent.replications
        or child.contraindications[: len(parent.contraindications)] != parent.contraindications
        or child.attribution[: len(parent.attribution)] != parent.attribution
        or (parent.dial is not None and child.dial is None)
        or (parent.legend is not None and child.legend is None)
    ):
        raise NrlyError(
            "lineage_mismatch", "a revision cannot remove what an earlier revision recorded"
        )


def _require_refereed(entry: LeverEntry, verdict: RefereeVerdict) -> None:
    """Refuse a verdict for another edit or other weights before a revision is sealed."""
    if not entry.operations:
        raise NrlyError(
            "invalid_lever", "a verdict referees operations; nominate a strength first"
        )
    if verdict.operations_digest != operations_digest(entry.operations):
        raise NrlyError(
            "verdict_mismatch", "a verdict must referee exactly this lever's operations"
        )
    if verdict.base != entry.base:
        raise NrlyError(
            "verdict_mismatch", "a verdict must be refereed on the weights this lever pins"
        )


def _copy_tensor(
    source: TensorArtifactStore, target: TensorArtifactStore, reference: TensorArtifact
) -> None:
    copied = target.put(source.get(reference), dtype=reference.dtype)
    if copied != reference:
        raise NrlyError("artifact_mismatch", "direction artifact changed while being copied")


class LeverMap:
    """Content-addressed lever entries and the direction tensors they reference."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.entries = self.root / "entries"
        self.tensors = TensorArtifactStore(self.root / "tensors")

    def _holds(self, entry_id: str) -> bool:
        # lexists, so a link planted under a handle is refused on read instead of skipped.
        return os.path.lexists(self.entries / f"{entry_id}.json")

    def _store(self, entry: LeverEntry) -> LeverEntry:
        raw = canonical_json(entry).encode("utf-8")
        if len(raw) > MAX_ENTRY_BYTES:
            raise NrlyError("budget_exceeded", "lever entry exceeds 4 MiB")
        write_immutable(self.entries, f"{entry.id}.json", raw)
        return entry

    def _revise(self, parent: LeverEntry, **changes: object) -> LeverEntry:
        content = parent.model_dump(mode="json", exclude={"id"})
        content.update(changes)
        content["parent_id"] = parent.id
        entry = _seal(content)
        _verify_link(parent, entry)
        return self._store(entry)

    def add(self, content: Mapping[str, object]) -> LeverEntry:
        """Seal and store an entry; a named parent must be held and is never rewritten."""
        entry = _seal(content)
        if entry.parent_id is not None:
            if not self._holds(entry.parent_id):
                raise NrlyError("unknown_lever", "the named parent entry is not on this map")
            _verify_link(self.get(entry.parent_id), entry)
        if entry.derivation is not None:
            try:
                self.tensors.get(entry.derivation.direction)
            except NrlyError as error:
                raise NrlyError(
                    error.code,
                    "the derivation's direction artifact is not intact in this map's tensor store",
                ) from error
        return self._store(entry)

    def _read(self, entry_id: str) -> LeverEntry:
        checked_handle(entry_id)
        if not self._holds(entry_id):
            raise NrlyError("unknown_lever", "this map holds no lever entry with that id")
        raw = read_local_file(self.entries, f"{entry_id}.json", MAX_ENTRY_BYTES)
        try:
            entry = LeverEntry.model_validate_json(raw)
        except ValueError as error:
            raise NrlyError("invalid_artifact", "invalid or altered lever entry") from error
        if entry.id != entry_id:
            raise NrlyError("artifact_mismatch", "lever entry handle differs from its content")
        return entry

    def get(self, entry_id: str) -> LeverEntry:
        """Read one verified entry; it must also be a faithful revision of a held parent."""
        entry = self._read(entry_id)
        if entry.parent_id is not None and self._holds(entry.parent_id):
            _verify_link(self._read(entry.parent_id), entry)
        return entry

    def attach_dial(
        self, entry_id: str, dial: Dial, operations: Sequence[WeightOperationSpec]
    ) -> LeverEntry:
        """Record a dose-response and, when it nominates a strength, the edit it compiles to."""
        parent = self.get(entry_id)
        if parent.verdict is not None:
            raise NrlyError(
                "lever_refereed", "the refereed edit cannot change once a verdict exists"
            )
        return self._revise(
            parent,
            dial=dial.model_dump(mode="json"),
            strength=dial.nominated,
            operations=_plain(operations),
        )

    def attach_legend(self, entry_id: str, legend: Legend, attribution: str) -> LeverEntry:
        """Add inherited atlas names (always pencil) with the credit their licence requires."""
        parent = self.get(entry_id)
        if parent.derivation is None:
            raise NrlyError("invalid_lever", "a legend describes a derived direction")
        line = attribution.strip()
        folded = line.casefold()
        if (
            legend.source.repo_id.casefold() not in folded
            or legend.source.license.casefold() not in folded
        ):
            raise NrlyError(
                "invalid_lever", "the attribution line must name the atlas source and its licence"
            )
        lines = parent.attribution if line in parent.attribution else [*parent.attribution, line]
        return self._revise(parent, legend=legend.model_dump(mode="json"), attribution=lines)

    def attach_verdict(self, entry_id: str, verdict: RefereeVerdict) -> LeverEntry:
        """Attach the first registered verdict; the entry's status follows from it."""
        parent = self.get(entry_id)
        if parent.verdict is not None:
            raise NrlyError(
                "verdict_exists", "a lever keeps its first verdict; add later ones as replications"
            )
        _require_refereed(parent, verdict)
        return self._revise(parent, verdict=verdict.model_dump(mode="json"))

    def add_replication(self, entry_id: str, verdict: RefereeVerdict) -> LeverEntry:
        """Record the same edit refereed on another runtime; the first verdict keeps the status."""
        parent = self.get(entry_id)
        if parent.verdict is None:
            raise NrlyError("verdict_missing", "a replication repeats an existing verdict")
        recorded = {record.runtime_fingerprint for record in (parent.verdict, *parent.replications)}
        if verdict.runtime_fingerprint in recorded:
            raise NrlyError(
                "replication_exists", "this runtime already has a verdict recorded for the lever"
            )
        _require_refereed(parent, verdict)
        return self._revise(parent, replications=_plain([*parent.replications, verdict]))

    def add_contraindication(self, entry_id: str, item: Contraindication) -> LeverEntry:
        parent = self.get(entry_id)
        return self._revise(parent, contraindications=_plain([*parent.contraindications, item]))

    def history(self, entry_id: str) -> list[LeverEntry]:
        """Held revisions, root first. An imported chain may begin after an absent parent."""
        chain = [self._read(entry_id)]
        while (parent_id := chain[-1].parent_id) is not None and self._holds(parent_id):
            if len(chain) >= MAX_MAP_ENTRIES:
                raise NrlyError("lineage_mismatch", "lever history exceeds the revision bound")
            parent = self._read(parent_id)
            _verify_link(parent, chain[-1])
            chain.append(parent)
        return list(reversed(chain))

    def heads(self, weights_sha256: str | None = None) -> list[LeverEntry]:
        """Latest revision of every chain: entries that no held entry names as its parent."""
        if weights_sha256 is not None:
            checked_handle(weights_sha256)
        identifiers: list[str] = []
        if self.entries.is_dir():
            with os.scandir(self.entries) as members:
                for count, member in enumerate(members, start=1):
                    if count > MAX_MAP_ENTRIES:
                        raise NrlyError(
                            "budget_exceeded",
                            f"a lever map scan reads at most {MAX_MAP_ENTRIES} entries",
                        )
                    match = _ENTRY_FILE.fullmatch(member.name)
                    if match is not None:
                        identifiers.append(match.group(1))
        entries = {identifier: self._read(identifier) for identifier in sorted(identifiers)}
        for entry in entries.values():
            parent = entries.get(entry.parent_id) if entry.parent_id is not None else None
            if parent is not None:
                _verify_link(parent, entry)
        revised = {entry.parent_id for entry in entries.values()}
        return sorted(
            (
                entry
                for entry in entries.values()
                if entry.id not in revised
                and (weights_sha256 is None or entry.base.weights_sha256 == weights_sha256)
            ),
            key=lambda entry: (entry.name, entry.id),
        )

    def lookup(
        self,
        weights_sha256: str,
        *,
        behavior: str | None = None,
        status: LeverStatus | None = None,
    ) -> list[LeverEntry]:
        """Current levers for one exact checkpoint, optionally by behavior test and status."""
        return [
            entry
            for entry in self.heads(weights_sha256)
            if (behavior is None or entry.behavior == behavior)
            and (status is None or entry.status == status)
        ]

    def export(self, entry_id: str, destination: Path) -> Path:
        """Write one entry and its direction artifact to a new or empty directory."""
        entry = self.get(entry_id)
        if destination.is_symlink() or destination.is_junction():
            raise NrlyError("invalid_artifact", "bundle destination cannot be a link")
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise NrlyError("destination_exists", "bundle destination must be absent or empty")
        if entry.derivation is not None:
            _copy_tensor(
                self.tensors,
                TensorArtifactStore(destination / BUNDLE_TENSORS),
                entry.derivation.direction,
            )
        write_immutable(destination, BUNDLE_ENTRY, canonical_json(entry).encode("utf-8"))
        return destination.resolve()

    def import_bundle(self, directory: Path) -> LeverEntry:
        """Verify and add an exported entry; its parent may be absent from this map."""
        raw = read_local_file(directory, BUNDLE_ENTRY, MAX_ENTRY_BYTES)
        try:
            entry = LeverEntry.model_validate_json(raw)
        except ValueError as error:
            raise NrlyError("invalid_artifact", "invalid or altered lever bundle") from error
        _require_coherent(entry)
        if entry.parent_id is not None and self._holds(entry.parent_id):
            _verify_link(self.get(entry.parent_id), entry)
        if entry.derivation is not None:
            _copy_tensor(
                TensorArtifactStore(directory / BUNDLE_TENSORS),
                self.tensors,
                entry.derivation.direction,
            )
        return self._store(entry)


def pencil(derivation: Derivation) -> dict[str, object]:
    """Content of a fresh pencil entry for a derivation: a site and direction, no strength."""
    entry = _seal(
        {
            "name": lever_name(derivation.recipe.behavior, [derivation.chosen]),
            "behavior": derivation.recipe.behavior,
            "base": derivation.base.model_dump(mode="json"),
            "sites": [derivation.chosen.model_dump(mode="json")],
            "derivation": derivation.model_dump(mode="json"),
        }
    )
    return entry.model_dump(mode="json", exclude={"id"})


def _passes(report: EvaluationReport, role: _Role) -> list[int]:
    return [int(case.outcome == "pass") for case in report.cases if case.role == role]


def _blocks(
    reports: Sequence[EvaluationReport], baseline: Sequence[EvaluationReport], role: _Role
) -> list[list[int]]:
    """Per seed, candidate pass minus the same seed's baseline pass on one role's cases."""
    return [
        [
            after - before
            for after, before in zip(_passes(report, role), _passes(original, role), strict=True)
        ]
        for report, original in zip(reports, baseline, strict=True)
    ]


def _rate(reports: Sequence[EvaluationReport], role: _Role) -> float:
    rates = [sum(passes) / len(passes) for passes in (_passes(report, role) for report in reports)]
    return math.fsum(rates) / len(rates)


def _finish(reports: Sequence[EvaluationReport]) -> tuple[float, float | None]:
    """Finish rate averaged over seeds, and accuracy among finished outputs pooled over seeds."""
    rates: list[float] = []
    finished_total = passed_total = 0
    for report in reports:
        finished, passed = report.summary.finished, report.summary.finished_passed
        if (
            finished is None
            or passed is None
            or report.summary.total != len(report.cases)
            or not passed <= finished <= report.summary.total
        ):
            raise NrlyError(
                "dial_incomplete", "truncation scoring requires consistent finish counts"
            )
        rates.append(finished / report.summary.total)
        finished_total += finished
        passed_total += passed
    return math.fsum(rates) / len(rates), passed_total / finished_total if finished_total else None


def _window(points: Sequence[DosePoint]) -> tuple[StrengthWindow | None, float | None]:
    def rank(point: DosePoint) -> float:
        return -math.inf if point.selectivity is None else point.selectivity

    runs: list[list[DosePoint]] = []
    adjacent = False
    for point in points:
        helps = point.inference.target_superior and point.inference.regression_non_inferior
        if helps and adjacent:
            runs[-1].append(point)
        elif helps:
            runs.append([point])
        adjacent = helps
    if not runs:
        return None, None
    # max() keeps the first of equal keys and runs are in increasing strength, so a full
    # tie resolves to the lower strengths.
    best = max(runs, key=lambda run: (len(run), max(rank(point) for point in run)))
    nominated = min(best, key=lambda point: (-rank(point), abs(point.strength), point.strength))
    window = StrengthWindow(low=best[0].strength, high=best[-1].strength, points=len(best))
    return window, nominated.strength


def build_dial(
    *,
    rule: InferenceRule,
    truncation: Literal["unscored", "fail"],
    generation: GenerationSpec,
    seeds: Sequence[int],
    runtime_fingerprint: str,
    baseline: Sequence[EvaluationReport],
    points: Mapping[float, tuple[str, Sequence[EvaluationReport]]],
) -> Dial:
    """Dose-response of an edit against the unedited model on development cases.

    Exploratory evidence: it nominates a strength for a registered study to referee and
    never yields a status.
    """
    if not seeds or len(set(seeds)) != len(seeds) or not points:
        raise NrlyError(
            "dial_incomplete", "a dial needs distinct seeds and at least one tested strength"
        )
    arms = [baseline, *(reports for _edit, reports in points.values())]
    if any(len(reports) != len(seeds) for reports in arms):
        raise NrlyError("dial_incomplete", "every dial arm needs exactly one report per seed")
    layout = [(case.case_id, case.role) for case in baseline[0].cases]
    if {role for _case, role in layout} != {"target", "regression"} or len(
        {case for case, _role in layout}
    ) != len(layout):
        raise NrlyError("dial_incomplete", "a dial needs distinct target and regression cases")
    for report in (report for reports in arms for report in reports):
        # An unscored case would otherwise enter a paired block as a failure.
        if not report.complete or any(case.outcome == "unscored" for case in report.cases):
            raise NrlyError("dial_incomplete", "every dial report must be complete and scored")
        if [(case.case_id, case.role) for case in report.cases] != layout:
            raise NrlyError("dial_incomplete", "dial reports must score the same ordered cases")
        if report.scope != baseline[0].scope or report.truncation != truncation:
            raise NrlyError(
                "dial_incomplete", "dial reports must share a scope and the dial's truncation"
            )
    measure_finish = truncation == "fail"
    baseline_finish = _finish(baseline)[0] if measure_finish else None
    dose: list[DosePoint] = []
    try:
        for strength in sorted(points):
            edit, reports = points[strength]
            inference = paired_inference(
                "baseline",
                _blocks(reports, baseline, "target"),
                _blocks(reports, baseline, "regression"),
                rule,
            )
            cost = max(rule.regression_margin, -inference.regression.mean_difference)
            finish_rate, accuracy = _finish(reports) if measure_finish else (None, None)
            dose.append(
                DosePoint(
                    strength=strength,
                    operations_digest=edit,
                    inference=inference,
                    target_rate=_rate(reports, "target"),
                    regression_rate=_rate(reports, "regression"),
                    finish_rate=finish_rate,
                    accuracy_when_finished=accuracy,
                    selectivity=inference.target.mean_difference / cost if cost > 0 else None,
                )
            )
        window, nominated = _window(dose)
        return Dial(
            rule=rule,
            truncation=truncation,
            generation=generation,
            seeds=list(seeds),
            cases=len(layout),
            runtime_fingerprint=runtime_fingerprint,
            baseline_target_rate=_rate(baseline, "target"),
            baseline_regression_rate=_rate(baseline, "regression"),
            baseline_finish_rate=baseline_finish,
            points=dose,
            window=window,
            nominated=nominated,
        )
    except ValueError as error:
        raise NrlyError(
            "invalid_dial", f"dose-response cannot form a dial: {_reason(error)}"
        ) from error


def verdict_from_study(
    study: StudyRecord, candidate: CandidateSpec, operations: Sequence[WeightOperationSpec]
) -> RefereeVerdict:
    """The verdict a completed registered study gives for its frozen nominee's exact edit."""
    if study.status != "completed" or study.final_result is None or study.finished_at is None:
        raise NrlyError(
            "verdict_unavailable", "a verdict requires a completed study with a final result"
        )
    if study.nominee_digest is None or study.nominee_digest != digest(candidate):
        raise NrlyError("verdict_unavailable", "the candidate is not the study's frozen nominee")
    edit = list(operations)
    try:
        compiled = weight_operations(candidate.program)
    except (NrlyError, ValueError) as error:
        raise NrlyError(
            "verdict_unavailable", "the nominee is not a declarative weight edit"
        ) from error
    if operations_digest(compiled) != operations_digest(edit):
        raise NrlyError(
            "verdict_unavailable", "the nominee does not compile to exactly these operations"
        )
    if study.model_identity is None:
        raise NrlyError("verdict_unavailable", "the study pinned no model identity")
    try:
        base = PortableBase.from_identity(study.model_identity)
    except (NrlyError, ValueError) as error:
        raise NrlyError(
            "verdict_unavailable",
            "the study's model identity does not pin its weights digest and dtype",
        ) from error
    try:
        return RefereeVerdict(
            base=base,
            study_id=study.id,
            study_spec_digest=study.spec_digest,
            holdout=study.holdout,
            operations_digest=operations_digest(edit),
            nominee_digest=study.nominee_digest,
            runtime_fingerprint=study.model_identity.fingerprint,
            inference_rule=study.spec.inference,
            truncation=study.spec.truncation,
            seeds=study_seeds(study.spec),
            controls=[control.name for control in study.spec.controls],
            comparators=[f"{item.kind}:{item.name}" for item in study.spec.comparators],
            development_cases=len(study.spec.development_cases),
            assessment=study.final_result,
            final_error_code=study.final_error_code,
            finished_at=study.finished_at,
        )
    except ValueError as error:
        raise NrlyError(
            "verdict_unavailable", f"the study record cannot form a verdict: {_reason(error)}"
        ) from error
