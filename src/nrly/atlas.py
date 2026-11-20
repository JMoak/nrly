"""Inherited feature dictionaries, read as a legend for a lever's direction.

Other groups publish sparse autoencoders and transcoders with labelled features for
common checkpoints. nrly never retrains them: it reads one decoder matrix out of a
pinned safetensors file and reports which of its rows lie nearest a lever's direction.
The result is always ``pencil``. A nearby feature is a name to test, never evidence
that editing along the direction has that feature's effect.

The safetensors container is parsed here as bounded numeric bytes and strict JSON:
never pickle, code, Python objects, or a model loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field, StrictInt, ValidationError, model_validator

from nrly.artifacts import checked_handle, read_local_file, write_immutable
from nrly.contracts import Contract, NrlyError, canonical_json
from nrly.lever_contracts import AtlasSource, Legend, LegendEntry

MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_ATLAS_TENSOR_BYTES = 2 * 1024 * 1024 * 1024
MAX_LABEL_RESPONSE_BYTES = 1024 * 1024
MAX_LABEL_CHARS = 500
MAX_LEGEND_ENTRIES = 32
_READ_BYTES = 64 * 1024 * 1024
_BLOCK_ELEMENTS = 1 << 22
_DIGEST = r"^[0-9a-f]{64}$"

# Byte width of every dtype the container defines, so each entry's size can be checked
# even though only the floating-point ones are ever decoded.
_WIDTHS = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}
_STORED = {"F32": "<f4", "F64": "<f8", "F16": "<f2", "BF16": "<u2"}

_REPO_ID = r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
_HUB_FILENAME = r"[A-Za-z0-9._+=-]+(/[A-Za-z0-9._+=-]+)*"
# The Hub path syntax reads these leading segments as a repository type, not a namespace.
_HUB_TYPE_PREFIXES = frozenset({"models", "datasets", "spaces", "kernels", "buckets"})

_NEURONPEDIA = "https://www.neuronpedia.org"
_NEURONPEDIA_PART = r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}"


class ByteSource(Protocol):
    """Random access to one immutable file; ``read`` returns exactly ``length`` bytes."""

    def size(self) -> int: ...

    def read(self, offset: int, length: int) -> bytes: ...


class RangedFilesystem(Protocol):
    """The two fsspec calls ``HubFile`` makes; ``HfFileSystem`` provides both."""

    def info(self, path: str) -> dict[str, Any]: ...

    def cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes: ...


def _check_range(offset: int, length: int, size: int) -> None:
    if (
        type(offset) is not int or type(length) is not int
        or offset < 0 or length < 0 or offset + length > size
    ):
        raise NrlyError("invalid_artifact", "read lies outside the atlas file")


class LocalFile:
    """A regular local file; symbolic links and junctions are refused."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        try:
            if self.path.is_symlink() or self.path.is_junction() or not self.path.is_file():
                raise NrlyError("invalid_artifact", "atlas file must be a regular local file")
            self._size = self.path.stat().st_size
        except OSError as error:
            raise NrlyError(
                "invalid_artifact", f"atlas file is unavailable: {self.path.name}",
            ) from error

    def size(self) -> int:
        return self._size

    def read(self, offset: int, length: int) -> bytes:
        _check_range(offset, length, self._size)
        try:
            with self.path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(length)
        except OSError as error:
            raise NrlyError(
                "invalid_artifact", f"atlas file is unavailable: {self.path.name}",
            ) from error
        if len(data) != length:
            raise NrlyError("invalid_artifact", "atlas file changed while it was being read")
        return data


def _hub_filesystem() -> RangedFilesystem:
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as error:
        raise NrlyError(
            "backend_unavailable",
            "reading an atlas from the Hub needs huggingface_hub; install the hf extra",
        ) from error
    # Anonymous: published dictionaries are public, so no stored credential is sent.
    filesystem: RangedFilesystem = HfFileSystem(token=False)
    return filesystem


class HubFile:
    """One file of a Hugging Face model repository at an exact commit, read by range.

    Only the byte ranges asked for are requested, so one tensor of a multi-hundred
    megabyte file costs that tensor. ``filesystem`` replaces the default anonymous
    ``HfFileSystem``, for a gated repository or a test.
    """

    def __init__(
        self,
        repo_id: str,
        revision: str,
        filename: str,
        *,
        filesystem: RangedFilesystem | None = None,
    ) -> None:
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise NrlyError(
                "invalid_request",
                "an atlas is pinned by a 40-hex commit, never a branch or tag",
            )
        if (
            not isinstance(repo_id, str) or not re.fullmatch(_REPO_ID, repo_id)
            or repo_id.split("/")[0] in _HUB_TYPE_PREFIXES
        ):
            raise NrlyError("invalid_request", "repo_id must be a model repository namespace/name")
        if (
            not isinstance(filename, str) or len(filename) > 300
            or not re.fullmatch(_HUB_FILENAME, filename)
            or any(part in {".", ".."} for part in filename.split("/"))
        ):
            raise NrlyError("invalid_request", "filename must be a plain relative repository path")
        self.repo_id = repo_id
        self.revision = revision
        self.filename = filename
        self._path = f"{repo_id}@{revision}/{filename}"
        self._filesystem = filesystem if filesystem is not None else _hub_filesystem()
        self._size: int | None = None

    def _unavailable(self, error: Exception) -> NrlyError:
        return NrlyError(
            "atlas_unavailable",
            f"cannot read {self.filename} from {self.repo_id}@{self.revision[:12]}: "
            f"{type(error).__name__}",
        )

    def size(self) -> int:
        if self._size is None:
            try:
                info = self._filesystem.info(self._path)
            except Exception as error:
                raise self._unavailable(error) from error
            size = info.get("size") if isinstance(info, dict) else None
            if type(size) is not int or size < 0 or info.get("type") != "file":
                raise NrlyError(
                    "atlas_unavailable", f"{self.filename} is not a file in {self.repo_id}",
                )
            self._size = size
        return self._size

    def read(self, offset: int, length: int) -> bytes:
        _check_range(offset, length, self.size())
        if length == 0:
            return b""
        try:
            # fsspec's default read-ahead cache would fetch an extra block past every range.
            data = self._filesystem.cat_file(
                self._path, start=offset, end=offset + length, cache_type="none",
            )
        except Exception as error:
            raise self._unavailable(error) from error
        if not isinstance(data, bytes | bytearray) or len(data) != length:
            raise NrlyError(
                "atlas_unavailable", "the Hub returned a different range than was requested",
            )
        return bytes(data)


@dataclass(frozen=True, slots=True)
class TensorEntry:
    """One tensor of a safetensors header; ``offset`` counts from the start of the file."""

    dtype: str
    shape: tuple[int, ...]
    offset: int
    byte_length: int


def _invalid(message: str) -> NrlyError:
    return NrlyError("invalid_artifact", message)


def _read_exact(source: ByteSource, offset: int, length: int) -> bytes:
    data = source.read(offset, length)
    if not isinstance(data, bytes) or len(data) != length:
        raise _invalid("atlas source returned fewer bytes than were requested")
    return data


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document = dict(pairs)
    if len(document) != len(pairs):
        raise ValueError("duplicate key")
    return document


def _tensor_entry(name: str, value: object, base: int, data_length: int) -> TensorEntry:
    label = repr(name[:80])
    if not isinstance(value, dict):
        raise _invalid(f"safetensors entry {label} must be an object")
    dtype, shape, offsets = value.get("dtype"), value.get("shape"), value.get("data_offsets")
    if not isinstance(dtype, str) or dtype not in _WIDTHS:
        raise _invalid(f"safetensors entry {label} has an unknown dtype")
    if (
        not isinstance(shape, list) or len(shape) > 8
        or any(type(x) is not int or x < 0 for x in shape)
    ):
        raise _invalid(f"safetensors entry {label} needs at most 8 non-negative dimensions")
    if (
        not isinstance(offsets, list) or len(offsets) != 2
        or any(type(x) is not int for x in offsets)
    ):
        raise _invalid(f"safetensors entry {label} needs two integer data offsets")
    begin, end = offsets
    if not 0 <= begin <= end <= data_length:
        raise _invalid(f"safetensors entry {label} lies outside the file")
    if math.prod(shape) * _WIDTHS[dtype] != end - begin:
        raise _invalid(f"safetensors entry {label} disagrees on shape, dtype, and byte range")
    return TensorEntry(dtype, tuple(shape), base + begin, end - begin)


def read_header(source: ByteSource) -> dict[str, TensorEntry]:
    """Parse and fully validate a safetensors header without reading any tensor data."""
    size = source.size()
    if type(size) is not int or size < 8:
        raise _invalid("safetensors file is shorter than its length prefix")
    length = int.from_bytes(_read_exact(source, 0, 8), "little")
    if length > MAX_HEADER_BYTES:
        raise _invalid("safetensors header exceeds 16 MiB")
    if length > size - 8:
        raise _invalid("safetensors header length runs past the end of the file")
    try:
        document = json.loads(
            _read_exact(source, 8, length).decode("utf-8"), object_pairs_hook=_unique_keys,
        )
    except (ValueError, RecursionError) as error:
        raise _invalid("safetensors header is not strict UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise _invalid("safetensors header must be a JSON object")
    entries: dict[str, TensorEntry] = {}
    for name, value in document.items():
        if name == "__metadata__":
            if not isinstance(value, dict):
                raise _invalid("safetensors metadata must be an object")
            continue
        entries[name] = _tensor_entry(name, value, 8 + length, size - 8 - length)
    return entries


def _named(header: dict[str, TensorEntry], name: str) -> TensorEntry:
    entry = header.get(name)
    if entry is None:
        raise _invalid(f"safetensors file has no tensor named {name[:80]!r}")
    return entry


def _all_finite(values: NDArray[Any]) -> bool:
    flat = values.reshape(-1)
    return all(
        bool(np.isfinite(flat[start : start + _BLOCK_ELEMENTS]).all())
        for start in range(0, flat.size, _BLOCK_ELEMENTS)
    )


def _read_values(source: ByteSource, entry: TensorEntry) -> tuple[NDArray[np.float32], str]:
    """Decode one tensor as float32 and hash its bytes exactly as they are stored."""
    stored = _STORED.get(entry.dtype)
    if stored is None:
        raise _invalid(f"tensor dtype {entry.dtype} is not a supported floating-point type")
    if max(entry.byte_length, math.prod(entry.shape) * 4) > MAX_ATLAS_TENSOR_BYTES:
        raise _invalid("tensor exceeds 2 GiB as stored or as float32")
    raw = np.empty(entry.byte_length, dtype=np.uint8)
    hasher = hashlib.sha256()
    for start in range(0, entry.byte_length, _READ_BYTES):
        chunk = _read_exact(
            source, entry.offset + start, min(_READ_BYTES, entry.byte_length - start),
        )
        raw[start : start + len(chunk)] = np.frombuffer(chunk, dtype=np.uint8)
        hasher.update(chunk)
    values: NDArray[np.float32]
    if entry.dtype == "BF16":
        # NumPy has no bfloat16. A bfloat16 is the high half of the float32 of equal value.
        bits = raw.view("<u2").astype(np.uint32)
        bits <<= 16
        values = bits.view(np.float32)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            values = raw.view(stored).astype(np.float32, copy=False)
    if not _all_finite(values):
        raise _invalid("tensor holds values that are not finite in float32")
    return values.reshape(entry.shape), hasher.hexdigest()


def read_tensor(source: ByteSource, name: str) -> NDArray[np.float32]:
    """Read one F32, F64, F16, or BF16 tensor as finite float32."""
    return _read_values(source, _named(read_header(source), name))[0]


def _atlas_source(**fields: object) -> AtlasSource:
    try:
        return AtlasSource.model_validate(fields)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        raise NrlyError(
            "invalid_request", f"invalid atlas source field {where}: {first['msg']}",
        ) from error


def load_decoder(
    source: ByteSource,
    *,
    kind: Literal["transcoder", "sae"],
    repo_id: str,
    revision: str,
    filename: str,
    license: str,
    layer: int,
    writes: str,
    expected_width: int,
    tensor: str = "W_dec",
    orientation: Literal["latents_by_width", "width_by_latents"] = "latents_by_width",
) -> tuple[AtlasSource, NDArray[np.float32]]:
    """Read a decoder as ``[latents, width]`` and pin it to its stored bytes.

    The orientation is stated by the caller and checked against ``expected_width``.
    It is never inferred, because a square or mislabelled matrix would silently turn
    feature directions into something else.
    """
    if orientation not in ("latents_by_width", "width_by_latents"):
        raise NrlyError(
            "invalid_request", "orientation must be latents_by_width or width_by_latents",
        )
    if type(expected_width) is not int or expected_width < 1:
        raise NrlyError("invalid_request", "expected_width must be a positive integer")
    if isinstance(source, HubFile) and (
        (source.repo_id, source.revision, source.filename) != (repo_id, revision, filename)
    ):
        raise NrlyError("atlas_mismatch", "the Hub file being read is not the one being recorded")
    entry = _named(read_header(source), tensor)
    found = list(entry.shape)
    if len(found) != 2:
        raise NrlyError("atlas_mismatch", f"decoder {tensor!r} must be a matrix, found {found}")
    width_axis = 1 if orientation == "latents_by_width" else 0
    if found[width_axis] != expected_width or found[1 - width_axis] < 1:
        raise NrlyError(
            "atlas_mismatch",
            f"decoder {tensor!r} has shape {found}, which is not {orientation} "
            f"for width {expected_width}",
        )
    fields: dict[str, object] = {
        "kind": kind, "repo_id": repo_id, "revision": revision, "filename": filename,
        "tensor": tensor, "license": license, "layer": layer, "writes": writes,
        "latents": found[1 - width_axis], "width": expected_width,
    }
    # Reject bad provenance before fetching what may be hundreds of megabytes.
    _atlas_source(**fields, tensor_sha256="0" * 64)
    values, tensor_sha256 = _read_values(source, entry)
    decoder = values if width_axis == 1 else np.ascontiguousarray(values.T)
    return _atlas_source(**fields, tensor_sha256=tensor_sha256), decoder


def _checked_decoder(decoder: ArrayLike, source: AtlasSource) -> NDArray[Any]:
    matrix = np.asarray(decoder)
    if matrix.dtype.kind != "f" or matrix.shape != (source.latents, source.width):
        raise NrlyError(
            "atlas_mismatch",
            f"decoder must be a real [{source.latents}, {source.width}] matrix "
            "to match its atlas source",
        )
    return matrix


class _CacheManifest(Contract):
    schema_version: Literal["nrly-decoder-cache/1"] = "nrly-decoder-cache/1"
    tensor_sha256: str = Field(pattern=_DIGEST)
    sha256: str = Field(pattern=_DIGEST)
    latents: StrictInt = Field(ge=1)
    width: StrictInt = Field(ge=1)
    byte_length: StrictInt = Field(ge=4, le=MAX_ATLAS_TENSOR_BYTES)

    @model_validator(mode="after")
    def valid_size(self) -> _CacheManifest:
        if self.latents * self.width * 4 != self.byte_length:
            raise ValueError("cached decoder shape and byte length disagree")
        return self


class DecoderCache:
    """Decoded decoders on disk, keyed by the SHA-256 of the tensor as published.

    The key identifies the published bytes, while the cached bytes are their float32
    ``[latents, width]`` decoding, so the manifest carries a second checksum for what is
    actually stored. Provenance stays in ``AtlasSource``: the same tensor may be reached
    through more than one revision or file.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def put(self, source: AtlasSource, decoder: ArrayLike) -> None:
        matrix = _checked_decoder(decoder, source)
        if matrix.size * 4 > MAX_ATLAS_TENSOR_BYTES:
            raise _invalid("decoder exceeds 2 GiB as float32")
        with np.errstate(over="ignore", invalid="ignore"):
            stored = np.ascontiguousarray(matrix, dtype="<f4")
        if not _all_finite(stored):
            raise _invalid("decoder must be finite in float32")
        content = stored.tobytes(order="C")
        manifest = _CacheManifest(
            tensor_sha256=source.tensor_sha256, sha256=hashlib.sha256(content).hexdigest(),
            latents=source.latents, width=source.width, byte_length=len(content),
        )
        write_immutable(self.root, f"{source.tensor_sha256}.bin", content)
        write_immutable(
            self.root, f"{source.tensor_sha256}.json", canonical_json(manifest).encode("utf-8"),
        )

    def get(self, tensor_sha256: str) -> NDArray[np.float32] | None:
        """Return the verified decoder, read-only, or None when it was never cached."""
        handle = checked_handle(tensor_sha256)
        if not (self.root / f"{handle}.json").is_file():
            return None
        try:
            manifest = _CacheManifest.model_validate_json(
                read_local_file(self.root, f"{handle}.json", 4096)
            )
        except ValueError as error:
            raise _invalid("invalid decoder cache manifest") from error
        if manifest.tensor_sha256 != handle:
            raise NrlyError("artifact_mismatch", "decoder cache manifest names another tensor")
        raw = read_local_file(self.root, f"{handle}.bin", manifest.byte_length)
        if len(raw) != manifest.byte_length or hashlib.sha256(raw).hexdigest() != manifest.sha256:
            raise NrlyError("artifact_mismatch", "cached decoder bytes do not match their manifest")
        values = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
        if not _all_finite(values):
            raise _invalid("cached decoder contains nonfinite values")
        return values.reshape(manifest.latents, manifest.width)


def _unit_direction(direction: ArrayLike, width: int) -> NDArray[np.float64]:
    given = np.asarray(direction)
    if given.dtype.kind not in "fiu" or given.ndim != 1:
        raise NrlyError("invalid_request", "direction must be a real vector")
    if given.size != width:
        raise NrlyError(
            "atlas_mismatch",
            f"direction has {given.size} dimensions but the atlas writes {width}",
        )
    vector = given.astype(np.float64)
    if not np.isfinite(vector).all():
        raise NrlyError("invalid_request", "direction must be finite")
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        raise NrlyError("invalid_request", "direction must be nonzero")
    # Scaling by the largest component first keeps the norm from overflowing or vanishing.
    scaled = vector / scale
    unit: NDArray[np.float64] = scaled / np.linalg.norm(scaled)
    return unit


def _cosines(matrix: NDArray[Any], unit: NDArray[np.float64]) -> NDArray[np.float64]:
    rows = max(1, _BLOCK_ELEMENTS // matrix.shape[1])
    cosines = np.zeros(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], rows):
        block = matrix[start : start + rows].astype(np.float64)
        norms = np.linalg.norm(block, axis=1)
        if not np.isfinite(norms).all():
            raise _invalid("decoder rows must have finite norms")
        # A zero row writes nothing, so it keeps cosine 0 instead of dividing by zero.
        np.divide(block @ unit, norms, out=cosines[start : start + rows], where=norms > 0)
    return np.clip(cosines, -1.0, 1.0)


def _explained(rows: NDArray[Any], unit: NDArray[np.float64]) -> float:
    basis = rows.astype(np.float64)
    norms = np.linalg.norm(basis, axis=1)
    # Unit rows span the same space and keep the solve's rank cutoff scale-free.
    basis = basis[norms > 0] / norms[norms > 0, None]
    if basis.shape[0] == 0:
        return 0.0
    coefficients = np.linalg.lstsq(basis.T, unit, rcond=None)[0]
    fitted = basis.T @ coefficients
    return float(np.clip(fitted @ fitted, 0.0, 1.0))


def _legend_entry(
    index: int,
    cosine: float,
    url_for: Callable[[int], str | None] | None,
    label_for: Callable[[int], tuple[str, str] | None] | None,
) -> LegendEntry:
    url = url_for(index) if url_for is not None else None
    if url is not None and (
        not isinstance(url, str) or not re.fullmatch(r"https?://[!-~]+", url)
    ):
        raise NrlyError("invalid_request", "url_for must return an http(s) URL or None")
    named = label_for(index) if label_for is not None else None
    label: str | None = None
    label_source: str | None = None
    if named is not None:
        if (
            not isinstance(named, tuple) or len(named) != 2
            or not all(isinstance(text, str) and text.strip() for text in named)
        ):
            raise NrlyError(
                "invalid_request",
                "label_for must return a nonempty (label, label_source) pair or None",
            )
        label, label_source = named
    try:
        return LegendEntry(
            index=index, cosine=cosine, url=url, label=label, label_source=label_source,
        )
    except ValidationError as error:
        raise NrlyError(
            "invalid_request", "url_for or label_for returned text beyond the legend's bounds",
        ) from error


def legend(
    direction: ArrayLike,
    decoder: ArrayLike,
    source: AtlasSource,
    *,
    top_k: int = 8,
    url_for: Callable[[int], str | None] | None = None,
    label_for: Callable[[int], tuple[str, str] | None] | None = None,
) -> Legend:
    """Name the ``top_k`` decoder rows nearest a direction, by absolute cosine.

    ``explained`` is the share of the direction's squared norm inside the span of
    those rows. Ties go to the lower index and zero rows have cosine 0. A label is
    recorded only with its source. The result is pencil: it suggests what to test.
    """
    if type(top_k) is not int or not 1 <= top_k <= MAX_LEGEND_ENTRIES:
        raise NrlyError("invalid_request", "top_k must be an integer from 1 to 32")
    matrix = _checked_decoder(decoder, source)
    unit = _unit_direction(direction, source.width)
    cosines = _cosines(matrix, unit)
    # A stable sort keeps equal magnitudes in index order.
    order = np.argsort(-np.abs(cosines), kind="stable")[:top_k]
    entries = [
        _legend_entry(int(index), float(cosines[index]), url_for, label_for) for index in order
    ]
    return Legend(source=source, entries=entries, explained=_explained(matrix[order], unit))


def _neuronpedia_path(model: str, source_id: str) -> str:
    for part in (model, source_id):
        if not isinstance(part, str) or not re.fullmatch(_NEURONPEDIA_PART, part):
            raise NrlyError(
                "invalid_request",
                "Neuronpedia model and source ids are 1..80 letters, digits, '.', '_' or '-', "
                "starting with a letter or digit",
            )
    return f"{model}/{source_id}"


def _feature_index(index: int) -> int:
    if type(index) is not int or not 0 <= index < 2**31:
        raise NrlyError("invalid_request", "feature index must be a non-negative integer")
    return index


def neuronpedia_url(model: str, source_id: str, index: int) -> str:
    """The public page of one feature; every path component is validated."""
    return f"{_NEURONPEDIA}/{_neuronpedia_path(model, source_id)}/{_feature_index(index)}"


def _first_description(document: object) -> str | None:
    # Assumed, not verified against the live API: the feature JSON is an object whose
    # "explanations" is a list of objects, and an explanation's text is its string field
    # "description". Any other shape yields no label.
    if not isinstance(document, dict):
        return None
    explanations = document.get("explanations")
    if not isinstance(explanations, list) or not explanations:
        return None
    first = explanations[0]
    description = first.get("description") if isinstance(first, dict) else None
    if not isinstance(description, str):
        return None
    # Third-party text: drop control and formatting characters, then collapse whitespace.
    text = " ".join(
        "".join(ch for ch in description if ch.isprintable() or ch.isspace()).split()
    )
    if len(text) > MAX_LABEL_CHARS:
        text = text[: MAX_LABEL_CHARS - 3] + "..."
    return text or None


def neuronpedia_labels(
    model: str,
    source_id: str,
    *,
    opener: Callable[..., Any] | None = None,
    timeout: float = 10.0,
) -> Callable[[int], tuple[str, str] | None]:
    """Opt-in label lookup for ``legend(label_for=...)``; one GET per feature.

    Each request carries only a generic ``User-Agent``. The returned callable never
    raises: any network, size, or parse failure is simply no label. ``opener`` is
    called like ``urllib.request.urlopen(request, timeout=...)``.
    """
    path = _neuronpedia_path(model, source_id)
    if isinstance(timeout, bool) or not isinstance(timeout, int | float) or not 0 < timeout <= 60:
        raise NrlyError("invalid_request", "timeout must be within (0, 60] seconds")
    open_url = opener if opener is not None else urllib.request.urlopen
    label_source = f"neuronpedia:{path}"

    def label_for(index: int) -> tuple[str, str] | None:
        try:
            request = urllib.request.Request(
                f"{_NEURONPEDIA}/api/feature/{path}/{_feature_index(index)}",
                headers={"User-Agent": "nrly"},
            )
            with open_url(request, timeout=timeout) as response:
                body = response.read(MAX_LABEL_RESPONSE_BYTES + 1)
            if not isinstance(body, bytes) or len(body) > MAX_LABEL_RESPONSE_BYTES:
                return None
            label = _first_description(json.loads(body))
        except Exception:
            return None
        return (label, label_source) if label else None

    return label_for
