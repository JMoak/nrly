"""Immutable cumulative edits and observed sequential-interference evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from nrly.artifacts import (
    MAX_MANIFEST_BYTES,
    checked_handle,
    read_local_file,
    write_immutable,
)
from nrly.contracts import (
    Contract,
    NrlyError,
    ProgramSpec,
    RunRecord,
    WeightOperationSpec,
    canonical_json,
    digest,
)
from nrly.evaluation.reports import ComparisonReport, compare_runs


def weight_operations(program: ProgramSpec) -> list[WeightOperationSpec]:
    program = ProgramSpec.model_validate(program.model_dump())
    if program.kind == "identity":
        return []
    if program.kind == "pipeline":
        return program.operations
    if not program.kind.startswith("weight_"):
        raise NrlyError("invalid_program", "reusable edits require declarative weight operations")
    return [
        WeightOperationSpec.model_validate(
            program.model_dump(exclude={"expected_base", "operations"}),
        )
    ]


class _RevisionContent(Contract):
    schema_version: Literal["nrly-edit/1"] = "nrly-edit/1"
    name: str = Field(min_length=1, max_length=160)
    parent_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parent_program_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    appended_at: StrictInt = Field(default=0, ge=0, le=31)
    program: ProgramSpec

    @model_validator(mode="after")
    def pinned_cumulative_program(self) -> _RevisionContent:
        if self.program.kind != "pipeline" or not self.program.expected_base:
            raise ValueError("edit revisions require a pinned cumulative weight pipeline")
        if len(self.program.operations) <= self.appended_at:
            raise ValueError("each edit revision must append at least one operation")
        if self.parent_id is None:
            if self.parent_program_sha256 is not None or self.appended_at != 0:
                raise ValueError("root edits cannot contain parent metadata")
        elif self.parent_program_sha256 is None or self.appended_at == 0:
            raise ValueError("child edits require the parent program digest and prefix length")
        return self


class EditRevision(_RevisionContent):
    id: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def immutable_identity(self) -> EditRevision:
        content = _RevisionContent.model_validate(self.model_dump(exclude={"id"}))
        if digest(content) != self.id:
            raise ValueError("edit revision content differs from its immutable identity")
        return self


class ConflictFootprint(Contract):
    parent_targets: list[str]
    appended_targets: list[str]
    overlapping_targets: list[str]
    disjoint_sparse_targets: list[str]
    semantics: Literal["declared_coordinates_only_no_behavioral_independence"] = (
        "declared_coordinates_only_no_behavioral_independence"
    )


def conflict_footprint(revision: EditRevision) -> ConflictFootprint:
    revision = EditRevision.model_validate(revision.model_dump())
    before = revision.program.operations[: revision.appended_at]
    after = revision.program.operations[revision.appended_at :]
    left = {str(op.target) for op in before}
    right = {str(op.target) for op in after}
    disjoint = []
    for target in sorted(left & right):
        parent_ops = [op for op in before if op.target == target]
        child_ops = [op for op in after if op.target == target]
        all_ops = parent_ops + child_ops
        if (
            all(op.kind in {"weight_patch", "weight_sparse_add"} for op in all_ops)
            and all(op.shape == all_ops[0].shape for op in all_ops)
            and not {i for op in parent_ops for i in op.indices}
            & {i for op in child_ops for i in op.indices}
        ):
            disjoint.append(target)
    return ConflictFootprint(
        parent_targets=sorted(left), appended_targets=sorted(right),
        overlapping_targets=sorted((left & right) - set(disjoint)),
        disjoint_sparse_targets=disjoint,
    )


def _verify_link(parent: EditRevision, child: EditRevision) -> None:
    if (
        child.parent_id != parent.id
        or child.parent_program_sha256 != digest(parent.program)
        or child.program.expected_base != parent.program.expected_base
        or child.appended_at != len(parent.program.operations)
        or child.program.operations[: child.appended_at] != parent.program.operations
    ):
        raise NrlyError(
            "lineage_mismatch", "child program does not preserve the exact parent prefix",
        )


class EditStore:
    """Content-addressed local revisions; rollback selects an ancestor without mutation."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def create(
        self, name: str, program: ProgramSpec, parent_id: str | None = None,
    ) -> EditRevision:
        """Append the supplied delta to the parent's cumulative program, if present."""
        appended = weight_operations(program)
        if not program.expected_base or not appended:
            raise NrlyError("invalid_program", "an edit requires pinned nonempty weight operations")
        parent = self.get(parent_id) if parent_id else None
        if parent and parent.program.expected_base != program.expected_base:
            raise NrlyError("base_mismatch", "parent and appended edit must share the exact base")
        previous = parent.program.operations if parent else []
        content = _RevisionContent(
            name=name, parent_id=parent.id if parent else None,
            parent_program_sha256=digest(parent.program) if parent else None,
            appended_at=len(previous),
            program=ProgramSpec(
                kind="pipeline", expected_base=program.expected_base,
                operations=[*previous, *appended],
            ),
        )
        revision = EditRevision(id=digest(content), **content.model_dump())
        raw = canonical_json(revision).encode("utf-8")
        if len(raw) > MAX_MANIFEST_BYTES:
            raise NrlyError("budget_exceeded", "edit revision exceeds 4 MiB")
        write_immutable(self.root, f"{revision.id}.json", raw)
        return revision

    def _read(self, identifier: str) -> EditRevision:
        checked_handle(identifier)
        raw = read_local_file(self.root, f"{identifier}.json", MAX_MANIFEST_BYTES)
        try:
            revision = EditRevision.model_validate_json(raw)
        except ValueError as error:
            raise NrlyError("invalid_artifact", "invalid edit revision") from error
        if revision.id != identifier:
            raise NrlyError("artifact_mismatch", "revision handle differs from its content")
        return revision

    def ancestry(self, identifier: str) -> list[EditRevision]:
        result = [self._read(identifier)]
        while result[-1].parent_id is not None:
            if len(result) >= 32:
                raise NrlyError("lineage_mismatch", "edit ancestry exceeds the operation bound")
            parent = self._read(str(result[-1].parent_id))
            _verify_link(parent, result[-1])
            result.append(parent)
        return list(reversed(result))

    def get(self, identifier: str) -> EditRevision:
        return self.ancestry(identifier)[-1]

    def select_ancestor(self, identifier: str, ancestor_id: str | None = None) -> ProgramSpec:
        ancestry = self.ancestry(identifier)
        if ancestor_id is None:
            return ProgramSpec(expected_base=ancestry[-1].program.expected_base)
        for revision in ancestry:
            if revision.id == ancestor_id:
                return revision.program
        raise NrlyError("lineage_mismatch", "selected revision is not an ancestor of this edit")


class SequentialReport(Contract):
    schema_version: Literal["nrly-sequential/1"] = "nrly-sequential/1"
    parent_revision: str
    candidate_revision: str
    comparable: bool
    verdict: Literal["no_observed_regression", "observed_regression", "inconclusive"]
    issues: list[str]
    lost_prior_improvements: list[str]
    footprint: ConflictFootprint
    original_to_parent: ComparisonReport
    original_to_candidate: ComparisonReport
    parent_to_candidate: ComparisonReport
    limitation: str = "Observed cases only; coordinate separation does not imply independence."


def compare_sequential(
    original: RunRecord, parent_run: RunRecord, candidate_run: RunRecord,
    parent_revision: EditRevision, candidate_revision: EditRevision,
) -> SequentialReport:
    """Compare original, A, and ordered A+B on the same fully scored cases."""
    parent_revision = EditRevision.model_validate(parent_revision.model_dump())
    candidate_revision = EditRevision.model_validate(candidate_revision.model_dump())
    issues: list[str] = []
    try:
        _verify_link(parent_revision, candidate_revision)
    except NrlyError as error:
        issues.append(str(error))
    if original.spec.program.kind != "identity":
        issues.append("original arm must execute the unedited identity program")
    if digest(parent_run.spec.program) != digest(parent_revision.program):
        issues.append("parent run does not execute the recorded parent revision")
    if digest(candidate_run.spec.program) != digest(candidate_revision.program):
        issues.append("candidate run does not execute the recorded cumulative revision")
    for run in (original, parent_run, candidate_run):
        if (
            run.model_identity is None
            or run.model_identity.fingerprint != candidate_revision.program.expected_base
        ):
            issues.append("run model identity differs from the pinned edit base")
    first = compare_runs(original, parent_run)
    cumulative = compare_runs(original, candidate_run)
    sequential = compare_runs(parent_run, candidate_run)
    comparisons = (("original/A", first), ("original/A+B", cumulative), ("A/A+B", sequential))
    for label, comparison in comparisons:
        issues.extend(f"{label}: {issue}" for issue in comparison.issues)
    comparable = not issues
    lost = sorted(set(first.improved_cases) & set(sequential.regressed_cases)) if comparable else []
    verdict: Literal["no_observed_regression", "observed_regression", "inconclusive"] = (
        "inconclusive" if issues else "observed_regression"
        if cumulative.regressed_cases or sequential.regressed_cases else "no_observed_regression"
    )
    return SequentialReport(
        parent_revision=parent_revision.id, candidate_revision=candidate_revision.id,
        comparable=comparable, verdict=verdict, issues=issues, lost_prior_improvements=lost,
        footprint=conflict_footprint(candidate_revision), original_to_parent=first,
        original_to_candidate=cumulative, parent_to_candidate=sequential,
    )
