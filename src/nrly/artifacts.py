"""Bounded content-addressed tensor data for trusted local Python authoring.

The format is raw little-endian numeric bytes plus strict JSON, never pickle,
Python, object arrays, or a model loader. Artifact size does not enlarge the
executable ProgramSpec payload limit.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field, StrictInt, model_validator

from nrly.contracts import Contract, NrlyError, canonical_json, digest

MAX_TENSOR_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_DIGEST = r"^[0-9a-f]{64}$"


class _TensorDescription(Contract):
    schema_version: Literal["nrly-tensor/1"] = "nrly-tensor/1"
    sha256: str = Field(pattern=_DIGEST)
    dtype: Literal["float32", "float64"]
    shape: tuple[StrictInt, ...] = Field(min_length=1, max_length=8)
    byte_length: StrictInt = Field(ge=1, le=MAX_TENSOR_BYTES)

    @model_validator(mode="after")
    def valid_size(self) -> _TensorDescription:
        width = 4 if self.dtype == "float32" else 8
        if any(x <= 0 or x > 2**31 - 1 for x in self.shape):
            raise ValueError("tensor dimensions must be positive bounded integers")
        if math.prod(self.shape) * width != self.byte_length:
            raise ValueError("tensor shape, dtype, and byte length disagree")
        return self


class TensorArtifact(_TensorDescription):
    id: str = Field(pattern=_DIGEST)

    @model_validator(mode="after")
    def valid_identity(self) -> TensorArtifact:
        description = _TensorDescription.model_validate(self.model_dump(exclude={"id"}))
        if digest(description) != self.id:
            raise ValueError("tensor artifact identity does not match its description")
        return self


def checked_handle(identifier: str) -> str:
    if not re.fullmatch(_DIGEST, identifier):
        raise NrlyError("invalid_artifact", "artifact id must be a SHA256 hexadecimal digest")
    return identifier


def read_local_file(directory: Path, filename: str, limit: int) -> bytes:
    """Read a bounded regular member; callers supply fixed names or digest handles."""
    directory = directory.resolve()
    path = directory / filename
    try:
        if (
            path.is_symlink() or path.is_junction() or path.resolve().parent != directory
            or not path.is_file() or path.stat().st_size > limit
        ):
            raise NrlyError("invalid_artifact", "artifact member must be a bounded local file")
        with path.open("rb") as stream:
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise NrlyError("invalid_artifact", "artifact member exceeds its size limit")
        return content
    except OSError as error:
        raise NrlyError(
            "invalid_artifact", f"artifact member is unavailable: {filename}",
        ) from error


def write_immutable(directory: Path, filename: str, content: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    path = directory / filename
    if path.resolve().parent != directory or path.is_symlink() or path.is_junction():
        raise NrlyError("invalid_artifact", "artifact member escapes its store")
    try:
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        if read_local_file(directory, filename, len(content)) != content:
            raise NrlyError(
                "artifact_mismatch", "immutable artifact already has different bytes",
            ) from error


class TensorArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def put(
        self, values: ArrayLike, *, dtype: Literal["float32", "float64"] = "float32",
    ) -> TensorArtifact:
        if dtype not in {"float32", "float64"}:
            raise NrlyError("invalid_artifact", "tensor dtype must be float32 or float64")
        value = np.asarray(values)
        width = 4 if dtype == "float32" else 8
        if (
            value.dtype.kind not in "fiu" or not 1 <= value.ndim <= 8
            or value.size == 0 or value.size * width > MAX_TENSOR_BYTES
        ):
            raise NrlyError("invalid_artifact", "tensor data must be real, nonempty, and <=16 MiB")
        with np.errstate(over="ignore", invalid="ignore"):
            converted = np.asarray(value, dtype="<f4" if width == 4 else "<f8", order="C")
        if not np.isfinite(converted).all():
            raise NrlyError("invalid_artifact", "tensor data must remain finite in stored dtype")
        content = converted.tobytes(order="C")
        description = _TensorDescription(
            sha256=hashlib.sha256(content).hexdigest(), dtype=dtype,
            shape=tuple(converted.shape), byte_length=len(content),
        )
        reference = TensorArtifact(id=digest(description), **description.model_dump())
        write_immutable(self.root, f"{reference.id}.bin", content)
        write_immutable(
            self.root, f"{reference.id}.json", canonical_json(reference).encode("utf-8"),
        )
        return reference

    def inspect(self, identifier: str) -> TensorArtifact:
        checked_handle(identifier)
        raw = read_local_file(self.root, f"{identifier}.json", 4096)
        try:
            reference = TensorArtifact.model_validate_json(raw)
        except ValueError as error:
            raise NrlyError("invalid_artifact", "invalid tensor artifact manifest") from error
        if reference.id != identifier:
            raise NrlyError("artifact_mismatch", "tensor artifact handle differs from manifest")
        self.get(reference)
        return reference

    def get(self, reference: TensorArtifact) -> NDArray[np.float32] | NDArray[np.float64]:
        reference = TensorArtifact.model_validate(reference.model_dump())
        raw = read_local_file(self.root, f"{reference.id}.bin", reference.byte_length)
        if len(raw) != reference.byte_length or hashlib.sha256(raw).hexdigest() != reference.sha256:
            raise NrlyError("artifact_mismatch", "tensor bytes do not match the artifact manifest")
        value = np.frombuffer(raw, dtype="<f4" if reference.dtype == "float32" else "<f8")
        if not np.isfinite(value).all():
            raise NrlyError("invalid_artifact", "tensor artifact contains nonfinite values")
        return value.reshape(reference.shape).copy()
