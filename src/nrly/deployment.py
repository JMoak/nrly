"""Checksummed declarative patch packages with scoped Python application.

Packages contain no code and no model weights. Consumers load their own trusted
model, then apply the bounded program only inside the context. This API does not
export merged checkpoints or overwrite source model files.

There are two formats because there are two things a package can pin:

``nrly-patch/1`` is evidence. It pins one runtime fingerprint, which also hashes the
device, driver and library versions, so it is accepted only by that exact runtime.

``nrly-patch/2`` is meant to travel. It pins a ``PortableBase``: the exact weights at
a load dtype, and no machine.

Binding converts a portable edit into a runnable ``ProgramSpec`` for one local runtime
whose weights digest has been verified. The kernel and worker need no portable mode:
the local fingerprint itself covers ``weights_sha256``, so a bound program still
executes only on the weights it was written for.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from nrly.artifacts import MAX_MANIFEST_BYTES, read_local_file, write_immutable
from nrly.contracts import (
    Contract,
    ModelIdentity,
    NrlyError,
    PortableBase,
    ProgramSpec,
    WeightChange,
    WeightOperationSpec,
    canonical_json,
    digest,
)
from nrly.lineage import EditRevision, weight_operations

if TYPE_CHECKING:
    from torch import nn

_MANIFEST_MEMBERS: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


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


class _PortablePatchContent(Contract):
    schema_version: Literal["nrly-patch/2"] = "nrly-patch/2"
    base: PortableBase
    operations: list[WeightOperationSpec] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def bounded_payload(self) -> _PortablePatchContent:
        # The bound a pipeline enforces, so every valid package binds to a runnable program.
        if sum(len(op.values) + len(op.left) + len(op.right) for op in self.operations) > 65536:
            raise ValueError("portable patch payload exceeds 65536 values")
        return self


class PortablePatchManifest(_PortablePatchContent):
    """A weight edit that pins exact weights at a load dtype and no runtime fingerprint."""

    id: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def checksum(self) -> PortablePatchManifest:
        content = _PortablePatchContent.model_validate(self.model_dump(exclude={"id"}))
        if digest(content) != self.id:
            raise ValueError("patch checksum does not match the manifest content")
        return self


def _write_manifest(destination: Path, manifest: PatchManifest | PortablePatchManifest) -> None:
    raw = canonical_json(manifest).encode("utf-8")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise NrlyError("budget_exceeded", "patch manifest exceeds 4 MiB")
    if destination.is_symlink() or destination.is_junction():
        raise NrlyError("invalid_artifact", "patch destination cannot be a link")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise NrlyError("destination_exists", "patch destination must be absent or empty")
    write_immutable(destination, "manifest.json", raw)


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
    _write_manifest(destination, manifest)
    return manifest


def export_portable_patch(
    destination: Path, operations: Sequence[WeightOperationSpec], base: PortableBase,
) -> PortablePatchManifest:
    """Write a new package that pins weights, not a machine; nothing is ever replaced."""
    content = _PortablePatchContent.model_validate(
        {"base": base.model_dump(),
         "operations": [operation.model_dump() for operation in operations]},
    )
    manifest = PortablePatchManifest(id=digest(content), **content.model_dump())
    _write_manifest(destination, manifest)
    return manifest


def read_patch(directory: Path) -> PatchManifest:
    raw = read_local_file(directory, "manifest.json", MAX_MANIFEST_BYTES)
    try:
        return PatchManifest.model_validate_json(raw)
    except (ValueError, NrlyError) as error:
        raise NrlyError("invalid_artifact", "invalid or mismatched patch manifest") from error


def read_portable_patch(directory: Path) -> PortablePatchManifest:
    raw = read_local_file(directory, "manifest.json", MAX_MANIFEST_BYTES)
    try:
        return PortablePatchManifest.model_validate_json(raw)
    except ValueError as error:
        raise NrlyError("invalid_artifact", "invalid or mismatched patch manifest") from error


def read_any_patch(directory: Path) -> PatchManifest | PortablePatchManifest:
    """Read either format, chosen by the manifest's declared ``schema_version``.

    Only a manifest declaring ``nrly-patch/2`` is read as portable. Everything else
    takes the ``nrly-patch/1`` path unchanged, which rejects any other version.
    """
    raw = read_local_file(directory, "manifest.json", MAX_MANIFEST_BYTES)
    try:
        if _MANIFEST_MEMBERS.validate_json(raw).get("schema_version") == "nrly-patch/2":
            return PortablePatchManifest.model_validate_json(raw)
        return PatchManifest.model_validate_json(raw)
    except (ValueError, NrlyError) as error:
        raise NrlyError("invalid_artifact", "invalid or mismatched patch manifest") from error


def _base_differences(base: PortableBase, identity: ModelIdentity) -> str:
    local = {
        **identity.details,
        "backend": identity.backend, "model_id": identity.model_id, "revision": identity.revision,
    }
    return ", ".join(
        name for name in PortableBase.model_fields if local.get(name) != getattr(base, name)
    )


def bind_operations(
    operations: Sequence[WeightOperationSpec], base: PortableBase, identity: ModelIdentity,
) -> ProgramSpec:
    """Bind a portable edit to one local runtime that holds exactly the base weights.

    ``identity`` must be the recorded identity of the model loaded here, taken from its
    backend or from an inspection run. The returned pipeline pins that runtime's
    fingerprint, which covers ``weights_sha256``, so the kernel and worker refuse it
    wherever the loaded weights or runtime differ.
    """
    if not base.matches(identity):
        raise NrlyError(
            "base_mismatch",
            "loaded model differs from the portable patch base in "
            + _base_differences(base, identity),
        )
    if not identity.fingerprint:
        raise NrlyError("invalid_identity", "model identity lacks its runtime fingerprint")
    return ProgramSpec.model_validate(
        {"kind": "pipeline", "expected_base": identity.fingerprint,
         "operations": [operation.model_dump() for operation in operations]},
    )


@contextmanager
def apply_patch_package(
    model: nn.Module, identity: ModelIdentity, directory: Path,
) -> Iterator[list[WeightChange]]:
    """Apply for local inference, then restore touched weights, including on exceptions.

    Use an exclusively owned model, with no concurrent inference or edits. The
    caller supplies the recorded identity of that already loaded model. Tensor
    bytes are independently verified before and after the context.

    An ``nrly-patch/1`` package requires that exact identity. A portable package
    requires only its ``PortableBase``, and is bound to the supplied local identity
    once that identity records the same model, revision, dtype and weights.
    """
    from nrly.kernel import intervene, tensor_digest

    manifest = read_any_patch(directory)
    if isinstance(manifest, PortablePatchManifest):
        program = bind_operations(manifest.operations, manifest.base, identity)
        base_digest = manifest.base.weights_sha256
    else:
        if identity != manifest.identity:
            raise NrlyError(
                "base_mismatch", "loaded model identity differs from the patch identity"
            )
        program = manifest.program
        base_digest = str(manifest.identity.details["weights_sha256"])
    if tensor_digest(model) != base_digest:
        raise NrlyError("base_mismatch", "loaded model weights differ from the patch base")
    try:
        with intervene(model, program, identity.fingerprint) as changes:
            yield changes
    finally:
        if tensor_digest(model) != base_digest:
            raise NrlyError("restoration_failed", "model state differs after patch restoration")
