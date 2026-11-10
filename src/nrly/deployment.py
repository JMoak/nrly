"""Portable checksummed declarative patches with scoped Python application.

Packages contain no code and no model weights. They identify one exact model
execution identity and its base tensor digest. Consumers load their own trusted
model, then apply the bounded program only inside the context. This API does not
export merged checkpoints or overwrite source model files.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from nrly.artifacts import MAX_MANIFEST_BYTES, read_local_file, write_immutable
from nrly.contracts import (
    Contract,
    ModelIdentity,
    NrlyError,
    ProgramSpec,
    WeightChange,
    canonical_json,
    digest,
)
from nrly.lineage import EditRevision, weight_operations

if TYPE_CHECKING:
    from torch import nn


class _PatchContent(Contract):
    schema_version: Literal["nrly-patch/1"] = "nrly-patch/1"
    identity: ModelIdentity
    program: ProgramSpec
    revision: EditRevision | None = None

    @model_validator(mode="after")
    def exact_identity(self) -> _PatchContent:
        if not self.identity.fingerprint or self.program.expected_base != self.identity.fingerprint:
            raise ValueError("patch program must pin the exact supplied model fingerprint")
        base_digest = self.identity.details.get("weights_sha256")
        if not isinstance(base_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", base_digest):
            raise ValueError("patch identity requires the exact base tensor digest")
        weight_operations(self.program)
        if self.revision is not None and digest(self.revision.program) != digest(self.program):
            raise ValueError("patch program must equal its referenced cumulative revision")
        return self


class PatchManifest(_PatchContent):
    id: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def checksum(self) -> PatchManifest:
        content = _PatchContent.model_validate(self.model_dump(exclude={"id"}))
        if digest(content) != self.id:
            raise ValueError("patch checksum does not match the manifest content")
        return self


def export_patch(
    destination: Path, program: ProgramSpec, identity: ModelIdentity,
    revision: EditRevision | None = None,
) -> PatchManifest:
    """Write a new package; existing content and source checkpoints are never replaced."""
    content = _PatchContent.model_validate(
        {"program": program.model_dump(), "identity": identity.model_dump(),
         "revision": revision.model_dump() if revision else None},
    )
    manifest = PatchManifest(id=digest(content), **content.model_dump())
    raw = canonical_json(manifest).encode("utf-8")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise NrlyError("budget_exceeded", "patch manifest exceeds 4 MiB")
    if destination.is_symlink() or destination.is_junction():
        raise NrlyError("invalid_artifact", "patch destination cannot be a link")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise NrlyError("destination_exists", "patch destination must be absent or empty")
    write_immutable(destination, "manifest.json", raw)
    return manifest


def read_patch(directory: Path) -> PatchManifest:
    raw = read_local_file(directory, "manifest.json", MAX_MANIFEST_BYTES)
    try:
        return PatchManifest.model_validate_json(raw)
    except (ValueError, NrlyError) as error:
        raise NrlyError("invalid_artifact", "invalid or mismatched patch manifest") from error


@contextmanager
def apply_patch_package(
    model: nn.Module, identity: ModelIdentity, directory: Path,
) -> Iterator[list[WeightChange]]:
    """Apply for local inference, then restore touched weights, including on exceptions.

    Use an exclusively owned model, with no concurrent inference or edits. The
    caller supplies the recorded identity of that already loaded model. Tensor
    bytes are independently verified before and after the context.
    """
    from nrly.kernel import intervene, tensor_digest

    manifest = read_patch(directory)
    if identity != manifest.identity:
        raise NrlyError("base_mismatch", "loaded model identity differs from the patch identity")
    base_digest = manifest.identity.details["weights_sha256"]
    if tensor_digest(model) != base_digest:
        raise NrlyError("base_mismatch", "loaded model weights differ from the patch base")
    try:
        with intervene(model, manifest.program, identity.fingerprint) as changes:
            yield changes
    finally:
        if tensor_digest(model) != base_digest:
            raise NrlyError("restoration_failed", "model state differs after patch restoration")
