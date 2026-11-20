from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import urllib.error
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nrly.atlas as atlas
from nrly.atlas import (
    MAX_LABEL_RESPONSE_BYTES,
    DecoderCache,
    HubFile,
    LocalFile,
    TensorEntry,
    legend,
    load_decoder,
    neuronpedia_labels,
    neuronpedia_url,
    read_header,
    read_tensor,
)
from nrly.contracts import NrlyError
from nrly.lever_contracts import AtlasSource

REPO = "mwhanna/qwen3-0.6b-transcoders-lowl0"
REVISION = "f28e5268fa1f3f685e028a8ba64a553e862dc7c3"
FILENAME = "layer_3.safetensors"
PROVENANCE: dict[str, Any] = {
    "kind": "transcoder", "repo_id": REPO, "revision": REVISION, "filename": FILENAME,
    "license": "mit", "layer": 3, "writes": "mlp.hook_out",
}
Stored = dict[str, tuple[str, list[int], bytes]]


def framed(header: bytes, data: bytes = b"") -> bytes:
    return struct.pack("<Q", len(header)) + header + data


def container(tensors: Stored, metadata: dict[str, str] | None = None) -> bytes:
    header: dict[str, object] = {} if metadata is None else {"__metadata__": metadata}
    data = b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    return framed(json.dumps(header).encode("utf-8"), data)


def local(tmp_path: Path, content: bytes, name: str = FILENAME) -> LocalFile:
    path = tmp_path / name
    path.write_bytes(content)
    return LocalFile(path)


def one_entry(tmp_path: Path, entry: object, data: bytes = bytes(8)) -> LocalFile:
    return local(tmp_path, framed(json.dumps({"w": entry}).encode("utf-8"), data))


def code_of(action: Callable[[], object]) -> str:
    with pytest.raises(NrlyError) as caught:
        action()
    return caught.value.code


def atlas_source(latents: int, width: int, tensor_sha256: str = "ab" * 32) -> AtlasSource:
    return AtlasSource(
        **PROVENANCE, tensor_sha256=tensor_sha256, latents=latents, width=width,
    )


def orthonormal_rows(latents: int, width: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    basis, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(width, width)))
    rows: np.ndarray[Any, np.dtype[np.float64]] = basis.T[:latents]
    return rows


class Claimed:
    """Claims a file size without holding the bytes, so size bounds are reached cheaply."""

    def __init__(self, head: bytes, size: int) -> None:
        self.head = head
        self.claimed = size
        self.reads: list[tuple[int, int]] = []

    def size(self) -> int:
        return self.claimed

    def read(self, offset: int, length: int) -> bytes:
        self.reads.append((offset, length))
        assert offset + length <= len(self.head), "the reader went past the header"
        return self.head[offset : offset + length]


class Recording:
    def __init__(self, inner: LocalFile) -> None:
        self.inner = inner
        self.reads: list[tuple[int, int]] = []

    def size(self) -> int:
        return self.inner.size()

    def read(self, offset: int, length: int) -> bytes:
        self.reads.append((offset, length))
        return self.inner.read(offset, length)


class FakeHub:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.ranges: list[tuple[str, int, int]] = []
        self.options: list[dict[str, Any]] = []
        self.stats = 0
        self.failure: Exception | None = None

    def info(self, path: str) -> dict[str, Any]:
        self.stats += 1
        if self.failure is not None:
            raise self.failure
        return {"name": path, "size": len(self.files[path]), "type": "file"}

    def cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes:
        assert start is not None and end is not None, "an unranged read fetches the whole file"
        self.ranges.append((path, start, end))
        self.options.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return self.files[path][start:end]


def test_reads_f32_f64_and_f16_as_float32_and_reports_absolute_offsets(tmp_path: Path) -> None:
    values = np.array([[1.5, -2.25, 0.0], [3.0e-3, 1024.0, -7.75]])
    content = container(
        {
            "single": ("F32", [2, 3], values.astype("<f4").tobytes()),
            "double": ("F64", [2, 3], values.astype("<f8").tobytes()),
            "half": ("F16", [3, 2], values.astype("<f2").tobytes()),
            "scalar": ("F32", [], np.float32(0.5).tobytes()),
            "empty": ("F32", [0, 4], b""),
            "steps": ("I64", [1], struct.pack("<q", 9)),
        },
        metadata={"format": "pt"},
    )
    source = local(tmp_path, content)
    start = 8 + int.from_bytes(content[:8], "little")
    header = read_header(source)
    assert list(header) == ["single", "double", "half", "scalar", "empty", "steps"]
    assert header["single"] == TensorEntry("F32", (2, 3), start, 24)
    assert header["double"] == TensorEntry("F64", (2, 3), start + 24, 48)
    assert header["half"] == TensorEntry("F16", (3, 2), start + 72, 12)
    assert header["empty"] == TensorEntry("F32", (0, 4), start + 88, 0)

    for name in ("single", "double"):
        result = read_tensor(source, name)
        assert result.dtype == np.float32 and result.shape == (2, 3)
        np.testing.assert_array_equal(result, values.astype(np.float32))
    half = read_tensor(source, "half")
    assert half.dtype == np.float32
    np.testing.assert_array_equal(half, values.astype("<f2").astype(np.float32).reshape(3, 2))
    assert half[1, 1] == np.float32(np.float16(3.0e-3)) != np.float32(3.0e-3)
    assert read_tensor(source, "scalar").shape == () and read_tensor(source, "scalar") == 0.5
    assert read_tensor(source, "empty").shape == (0, 4)
    # A known integer tensor does not spoil the file, but it is never decoded as a direction.
    assert code_of(lambda: read_tensor(source, "steps")) == "invalid_artifact"
    assert code_of(lambda: read_tensor(source, "W_dec")) == "invalid_artifact"


def test_bf16_matches_torch_bit_for_bit(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    values = torch.randn(5, 16, generator=torch.Generator().manual_seed(7)) * 37.0
    values[0, :5] = torch.tensor([0.0, -0.0, 1e-30, -3.0e38, 1e-40])
    stored = values.to(torch.bfloat16)
    raw = stored.view(torch.int16).numpy().astype("<i2").tobytes()
    result = read_tensor(local(tmp_path, container({"W": ("BF16", [5, 16], raw)})), "W")
    assert result.dtype == np.float32 and result.shape == (5, 16)
    np.testing.assert_array_equal(result, stored.to(torch.float32).numpy())
    assert np.signbit(result[0, 1]) and result[0, 1] == 0.0
    assert result[0, 3] < -2.9e38 and 0.0 < result[0, 4] < 1e-39


@pytest.mark.parametrize(
    "content",
    [
        b"\x10\x00\x00",
        struct.pack("<Q", 64) + b'{"w": {"dtype"',
        framed(b'{"w": {"dtype": "F32", "shape": [2], "data_offs'),
        framed(b"\xff\xfe{}"),
        framed(b"[1, 2]"),
        framed(b'"W_dec"'),
        framed(b"7"),
        framed(b'{"w": [0, 8]}', bytes(8)),
        framed(b'{"__metadata__": "pt"}'),
        framed(
            b'{"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},'
            b' "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}',
            bytes(8),
        ),
        framed(b"[" * 200_000 + b"]" * 200_000),
    ],
    ids=[
        "truncated length prefix", "header length past the end", "truncated header json",
        "not utf-8", "json array", "json string", "json number", "entry is not an object",
        "metadata is not an object", "duplicate tensor name", "nesting bomb",
    ],
)
def test_malformed_container_is_refused(tmp_path: Path, content: bytes) -> None:
    source = local(tmp_path, content)
    assert code_of(lambda: read_header(source)) == "invalid_artifact"
    assert code_of(lambda: read_tensor(source, "w")) == "invalid_artifact"


@pytest.mark.parametrize(
    "entry",
    [
        {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
        {"dtype": "F32", "shape": [0], "data_offsets": [9, 9]},
        {"dtype": "F32", "shape": [2], "data_offsets": [8, 0]},
        {"dtype": "F32", "shape": [2], "data_offsets": [-8, 0]},
        {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]},
        {"dtype": "F64", "shape": [2], "data_offsets": [0, 8]},
        {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 4]},
        {"dtype": "F128", "shape": [2], "data_offsets": [0, 8]},
        {"dtype": "f32", "shape": [2], "data_offsets": [0, 8]},
        {"dtype": 32, "shape": [2], "data_offsets": [0, 8]},
        {"shape": [2], "data_offsets": [0, 8]},
        {"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]},
        {"dtype": "F32", "shape": [2.0], "data_offsets": [0, 8]},
        {"dtype": "F32", "shape": [True, 2], "data_offsets": [0, 8]},
        {"dtype": "F32", "shape": [1] * 8 + [2], "data_offsets": [0, 8]},
        {"dtype": "F32", "shape": 2, "data_offsets": [0, 8]},
        {"dtype": "F32", "shape": [2], "data_offsets": [0, 8, 8]},
        {"dtype": "F32", "shape": [2], "data_offsets": [0.0, 8.0]},
        {"dtype": "F32", "shape": [2]},
    ],
    ids=[
        "end outside the file", "range outside the file", "empty range outside the file",
        "begin after end", "negative begin", "shape larger than range", "width larger than range",
        "shape smaller than range", "unknown dtype", "lowercase dtype", "dtype is not a string",
        "missing dtype", "negative dimension", "float dimension", "boolean dimension",
        "nine dimensions", "shape is not a list", "three offsets", "float offsets",
        "missing offsets",
    ],
)
def test_inconsistent_tensor_entry_is_refused(tmp_path: Path, entry: dict[str, Any]) -> None:
    source = one_entry(tmp_path, entry)
    assert code_of(lambda: read_header(source)) == "invalid_artifact"


@pytest.mark.parametrize(
    ("dtype", "raw"),
    [
        ("F32", struct.pack("<2f", 1.0, float("nan"))),
        ("F32", struct.pack("<2f", float("-inf"), 1.0)),
        ("F64", struct.pack("<2d", 1.0, float("inf"))),
        # Finite as stored, but beyond float32: it would become an infinite direction.
        ("F64", struct.pack("<2d", 1.0, 1e300)),
        ("F16", struct.pack("<2H", 0x3C00, 0x7C00)),
        ("BF16", struct.pack("<2H", 0x3F80, 0x7F80)),
        ("BF16", struct.pack("<2H", 0x3F80, 0xFFC1)),
    ],
    ids=["f32 nan", "f32 -inf", "f64 inf", "f64 overflow", "f16 inf", "bf16 inf", "bf16 nan"],
)
def test_nonfinite_values_are_refused(tmp_path: Path, dtype: str, raw: bytes) -> None:
    source = local(tmp_path, container({"w": (dtype, [2], raw)}))
    assert read_header(source)["w"].shape == (2,)
    assert code_of(lambda: read_tensor(source, "w")) == "invalid_artifact"


def test_size_bounds_are_enforced_before_any_large_read() -> None:
    oversized = Claimed(struct.pack("<Q", 16 * 1024 * 1024 + 1), 2**40)
    assert code_of(lambda: read_header(oversized)) == "invalid_artifact"
    assert oversized.reads == [(0, 8)]

    def claimed(dtype: str, width: int, count: int) -> Claimed:
        entry = {"dtype": dtype, "shape": [count], "data_offsets": [0, count * width]}
        head = framed(json.dumps({"w": entry}).encode("utf-8"))
        return Claimed(head, len(head) + count * width)

    # 2 GiB + 4 bytes as stored, then a half-precision tensor that only exceeds it decoded.
    for source in (claimed("F32", 4, 2**29 + 1), claimed("F16", 2, 2**29 + 1)):
        header_reads = [(0, 8), (8, len(source.head) - 8)]
        assert read_header(source)["w"].shape == (2**29 + 1,)
        source.reads.clear()
        assert code_of(lambda source=source: read_tensor(source, "w")) == "invalid_artifact"
        assert source.reads == header_reads


def test_tensor_bound_applies_to_stored_and_decoded_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atlas, "MAX_ATLAS_TENSOR_BYTES", 64)
    source = local(
        tmp_path,
        container(
            {
                "f32 at the bound": ("F32", [16], bytes(64)),
                "f32 over": ("F32", [17], bytes(68)),
                "f16 over when decoded": ("F16", [17], bytes(34)),
                "f64 at the bound": ("F64", [8], bytes(64)),
                "f64 over as stored": ("F64", [9], bytes(72)),
            }
        ),
    )
    assert read_tensor(source, "f32 at the bound").shape == (16,)
    assert read_tensor(source, "f64 at the bound").shape == (8,)
    for name in ("f32 over", "f16 over when decoded", "f64 over as stored"):
        assert code_of(lambda n=name: read_tensor(source, n)) == "invalid_artifact"


def test_short_read_from_a_source_is_refused(tmp_path: Path) -> None:
    inner = local(tmp_path, container({"w": ("F32", [2], bytes(8))}))

    class Short(Recording):
        def read(self, offset: int, length: int) -> bytes:
            return super().read(offset, length)[: max(0, length - 1)]

    assert code_of(lambda: read_header(Short(inner))) == "invalid_artifact"


def test_local_file_is_bounds_checked_and_must_be_a_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "plain.bin"
    path.write_bytes(bytes(range(10)))
    source = LocalFile(path)
    assert source.size() == 10
    assert source.read(2, 3) == bytes([2, 3, 4])
    assert source.read(10, 0) == b""
    for offset, length in [(9, 2), (-1, 1), (0, -1), (11, 0), (0, 11)]:
        assert code_of(lambda o=offset, n=length: source.read(o, n)) == "invalid_artifact"
    assert code_of(lambda: LocalFile(tmp_path)) == "invalid_artifact"
    assert code_of(lambda: LocalFile(tmp_path / "absent.bin")) == "invalid_artifact"
    path.write_bytes(bytes(4))
    assert code_of(lambda: source.read(0, 10)) == "invalid_artifact"


def test_local_file_refuses_a_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "target.safetensors"
    target.write_bytes(container({"w": ("F32", [2], bytes(8))}))
    link = tmp_path / "link.safetensors"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("this account cannot create symbolic links")
    assert code_of(lambda: LocalFile(link)) == "invalid_artifact"
    assert LocalFile(target).size() == target.stat().st_size


def hub_fixture(rows: int = 6, width: int = 4) -> tuple[FakeHub, bytes, bytes, str]:
    decoder = np.random.default_rng(3).normal(size=(rows, width)).astype("<f4").tobytes()
    content = container(
        {
            "W_enc": ("F32", [1024, 4], np.ones((1024, 4), "<f4").tobytes()),
            "W_dec": ("F32", [rows, width], decoder),
            "b_dec": ("F32", [width], bytes(4 * width)),
        }
    )
    path = f"{REPO}@{REVISION}/{FILENAME}"
    return FakeHub({path: content}), content, decoder, path


def test_hub_file_fetches_only_the_header_and_the_requested_tensor() -> None:
    hub, content, decoder, path = hub_fixture()
    source = HubFile(REPO, REVISION, FILENAME, filesystem=hub)
    pinned, matrix = load_decoder(source, **PROVENANCE, expected_width=4)

    header_end = 8 + int.from_bytes(content[:8], "little")
    begin = content.index(decoder)
    assert hub.ranges == [
        (path, 0, 8), (path, 8, header_end), (path, begin, begin + len(decoder)),
    ]
    assert sum(end - start for _, start, end in hub.ranges) < len(content) // 8
    # Without this option fsspec reads a whole extra block ahead of every range.
    assert hub.options == [{"cache_type": "none"}] * 3
    assert hub.stats == 1
    np.testing.assert_array_equal(matrix, np.frombuffer(decoder, "<f4").reshape(6, 4))
    assert pinned.tensor_sha256 == hashlib.sha256(decoder).hexdigest()
    assert (pinned.repo_id, pinned.revision, pinned.filename) == (REPO, REVISION, FILENAME)


def test_large_tensor_is_fetched_in_bounded_ranges_that_tile_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub, content, decoder, path = hub_fixture()
    monkeypatch.setattr(atlas, "_READ_BYTES", 40)
    pinned, matrix = load_decoder(
        HubFile(REPO, REVISION, FILENAME, filesystem=hub), **PROVENANCE, expected_width=4,
    )
    begin = content.index(decoder)
    assert hub.ranges[2:] == [
        (path, begin, begin + 40), (path, begin + 40, begin + 80), (path, begin + 80, begin + 96),
    ]
    assert pinned.tensor_sha256 == hashlib.sha256(decoder).hexdigest()
    np.testing.assert_array_equal(matrix, np.frombuffer(decoder, "<f4").reshape(6, 4))


@pytest.mark.parametrize(
    ("repo_id", "revision", "filename"),
    [
        (REPO, "main", FILENAME),
        (REPO, "v1.0", FILENAME),
        (REPO, REVISION[:7], FILENAME),
        (REPO, REVISION.upper(), FILENAME),
        (REPO, REVISION + "\n", FILENAME),
        ("qwen3-transcoders", REVISION, FILENAME),
        ("datasets/qwen3-transcoders", REVISION, FILENAME),
        ("mwhanna/qwen3@main", REVISION, FILENAME),
        (REPO, REVISION, "../layer_3.safetensors"),
        (REPO, REVISION, "/layer_3.safetensors"),
        (REPO, REVISION, "resid//params.safetensors"),
        (REPO, REVISION, "resid\\params.safetensors"),
        (REPO, REVISION, "layer_*.safetensors"),
        (REPO, REVISION, ""),
    ],
)
def test_hub_file_requires_an_exact_commit_and_plain_names(
    repo_id: str, revision: str, filename: str,
) -> None:
    hub = FakeHub({})
    code = code_of(lambda: HubFile(repo_id, revision, filename, filesystem=hub))
    assert code == "invalid_request"
    assert hub.stats == 0 and hub.ranges == []


def test_hub_file_reports_remote_failures_without_leaking_exception_types() -> None:
    hub, content, _, path = hub_fixture()
    source = HubFile(REPO, REVISION, FILENAME, filesystem=hub)
    assert code_of(lambda: source.read(len(content) - 1, 2)) == "invalid_artifact"
    assert hub.ranges == []
    assert source.read(len(content), 0) == b"" and hub.ranges == []

    hub.failure = ConnectionError("offline")
    assert code_of(lambda: source.read(0, 8)) == "atlas_unavailable"
    unreachable = HubFile(REPO, REVISION, "layer_4.safetensors", filesystem=hub)
    assert code_of(unreachable.size) == "atlas_unavailable"

    hub.failure = None
    hub.files[path] = content[:-1]
    assert code_of(lambda: source.read(len(content) - 8, 8)) == "atlas_unavailable"

    class Folder(FakeHub):
        def info(self, path: str) -> dict[str, Any]:
            return {"name": path, "size": 0, "type": "directory"}

    folder = HubFile(REPO, REVISION, "resid", filesystem=Folder({}))
    assert code_of(folder.size) == "atlas_unavailable"


def test_hub_file_without_huggingface_hub_is_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert code_of(lambda: HubFile(REPO, REVISION, FILENAME)) == "backend_unavailable"


def test_atlas_imports_without_optional_model_dependencies() -> None:
    blocked = ("torch", "huggingface_hub", "safetensors", "sae_lens", "circuit_tracer")
    script = (
        f"import sys\nfor name in {blocked!r}:\n    sys.modules[name] = None\n"
        "import nrly.atlas\nprint(nrly.atlas.MAX_HEADER_BYTES)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(16 * 1024 * 1024)


def test_load_decoder_pins_the_stored_tensor_bytes(tmp_path: Path) -> None:
    values = np.random.default_rng(5).normal(size=(6, 4)).astype("<f4")
    raw = values.tobytes()
    first = local(tmp_path, container({"W_dec": ("F32", [6, 4], raw)}), "a.safetensors")
    source, decoder = load_decoder(first, **PROVENANCE, expected_width=4)
    assert decoder.dtype == np.float32 and decoder.shape == (6, 4)
    np.testing.assert_array_equal(decoder, values)
    assert source == atlas_source(6, 4, hashlib.sha256(raw).hexdigest())
    assert source.tensor == "W_dec"

    # The same tensor at other offsets, beside other tensors, keeps the same identity.
    moved = container(
        {"W_enc": ("F32", [3], bytes(12)), "W_dec": ("F32", [6, 4], raw)}, {"format": "pt"},
    )
    again, _ = load_decoder(local(tmp_path, moved, "b.safetensors"), **PROVENANCE, expected_width=4)
    assert again.tensor_sha256 == source.tensor_sha256 != hashlib.sha256(moved).hexdigest()

    flipped = bytearray(raw)
    flipped[0] ^= 1
    changed = local(tmp_path, container({"W_dec": ("F32", [6, 4], bytes(flipped))}), "c.bin")
    assert load_decoder(changed, **PROVENANCE, expected_width=4)[0].tensor_sha256 != (
        source.tensor_sha256
    )

    # Identity is of the published bytes, not of their float32 decoding.
    half = values.astype("<f2").tobytes()
    named = container({"decoder": ("F16", [6, 4], half)})
    pinned, decoded = load_decoder(
        local(tmp_path, named, "d.bin"), **PROVENANCE, expected_width=4, tensor="decoder",
    )
    assert pinned.tensor == "decoder" and pinned.tensor_sha256 == hashlib.sha256(half).hexdigest()
    np.testing.assert_array_equal(decoded, values.astype("<f2").astype(np.float32))


def test_load_decoder_checks_orientation_and_never_guesses(tmp_path: Path) -> None:
    values = np.arange(24, dtype="<f4").reshape(4, 6)
    by_width = local(tmp_path, container({"W_dec": ("F32", [4, 6], values.tobytes())}), "a.bin")

    # Stored as [width, latents]: the default orientation must refuse, not transpose.
    assert code_of(lambda: load_decoder(by_width, **PROVENANCE, expected_width=4)) == (
        "atlas_mismatch"
    )
    source, decoder = load_decoder(
        by_width, **PROVENANCE, expected_width=4, orientation="width_by_latents",
    )
    assert (source.latents, source.width) == (6, 4)
    np.testing.assert_array_equal(decoder, values.T)
    assert decoder.flags.c_contiguous and decoder[5, 0] == 5.0 and decoder[0, 3] == 18.0
    assert source.tensor_sha256 == hashlib.sha256(values.tobytes()).hexdigest()

    for orientation in ("latents_by_width", "width_by_latents"):
        mismatch = code_of(
            lambda o=orientation: load_decoder(
                by_width, **PROVENANCE, expected_width=5, orientation=o,
            )
        )
        assert mismatch == "atlas_mismatch"

    # A square matrix fits either way, so only the stated orientation decides.
    square = np.arange(16, dtype="<f4").reshape(4, 4)
    squared = local(tmp_path, container({"W_dec": ("F32", [4, 4], square.tobytes())}), "b.bin")
    np.testing.assert_array_equal(
        load_decoder(squared, **PROVENANCE, expected_width=4)[1], square,
    )
    np.testing.assert_array_equal(
        load_decoder(
            squared, **PROVENANCE, expected_width=4, orientation="width_by_latents",
        )[1],
        square.T,
    )

    shapes: Stored = {
        "vector": ("F32", [4], bytes(16)),
        "cube": ("F32", [2, 2, 4], bytes(64)),
        "empty": ("F32", [0, 4], b""),
    }
    odd = local(tmp_path, container(shapes), "c.bin")
    for tensor in shapes:
        refused = code_of(
            lambda t=tensor: load_decoder(odd, **PROVENANCE, expected_width=4, tensor=t)
        )
        assert refused == "atlas_mismatch"
    assert code_of(lambda: load_decoder(odd, **PROVENANCE, expected_width=4)) == "invalid_artifact"


def test_load_decoder_rejects_bad_requests_before_reading_the_tensor(tmp_path: Path) -> None:
    raw = np.ones((6, 4), "<f4").tobytes()
    source = Recording(local(tmp_path, container({"W_dec": ("F32", [6, 4], raw)})))

    def attempt(**changes: Any) -> str:
        arguments = {**PROVENANCE, "expected_width": 4, **changes}
        return code_of(lambda: load_decoder(source, **arguments))

    assert attempt(revision="main") == "invalid_request"
    assert attempt(kind="crosscoder") == "invalid_request"
    assert attempt(license="") == "invalid_request"
    assert attempt(layer=-1) == "invalid_request"
    assert all(offset + length <= 8 + 200 for offset, length in source.reads)
    assert len(source.reads) == 8

    source.reads.clear()
    assert attempt(orientation="auto") == "invalid_request"
    assert attempt(expected_width=0) == "invalid_request"
    assert attempt(expected_width=True) == "invalid_request"
    assert source.reads == []

    hub, _, _, _ = hub_fixture()
    remote = HubFile(REPO, REVISION, FILENAME, filesystem=hub)
    recorded = {**PROVENANCE, "filename": "layer_4.safetensors"}
    assert code_of(lambda: load_decoder(remote, **recorded, expected_width=4)) == "atlas_mismatch"
    assert hub.ranges == []


def test_decoder_cache_round_trip_is_checked_and_immutable(tmp_path: Path) -> None:
    root = tmp_path / "atlas"
    (root / "index").mkdir(parents=True)
    cache = DecoderCache(root)
    decoder = np.random.default_rng(9).normal(size=(6, 4)).astype(np.float32)
    source = atlas_source(6, 4, "cd" * 32)
    assert cache.get(source.tensor_sha256) is None

    cache.put(source, decoder)
    cache.put(source, decoder.copy())
    stored = root / f"{source.tensor_sha256}.bin"
    assert stored.read_bytes() == decoder.astype("<f4").tobytes()
    manifest = json.loads((root / f"{source.tensor_sha256}.json").read_text("utf-8"))
    assert manifest == {
        "schema_version": "nrly-decoder-cache/1", "tensor_sha256": source.tensor_sha256,
        "sha256": hashlib.sha256(stored.read_bytes()).hexdigest(),
        "latents": 6, "width": 4, "byte_length": 96,
    }
    cached = cache.get(source.tensor_sha256)
    assert cached is not None and cached.dtype == np.float32 and cached.shape == (6, 4)
    np.testing.assert_array_equal(cached, decoder)
    # The cached matrix is the stored bytes, read-only, and usable as a legend's decoder.
    assert not cached.flags.writeable
    nearest = legend(decoder[3], cached, source, top_k=2)
    assert nearest == legend(decoder[3], decoder, source, top_k=2)
    assert nearest.entries[0].index == 3 and nearest.entries[0].cosine == pytest.approx(1.0)

    # Different values under the same published identity are refused and change nothing.
    assert code_of(lambda: cache.put(source, decoder + 1)) == "artifact_mismatch"
    assert stored.read_bytes() == decoder.astype("<f4").tobytes()

    assert code_of(lambda: cache.put(source, decoder[:5])) == "atlas_mismatch"
    assert code_of(lambda: cache.put(source, decoder.T)) == "atlas_mismatch"
    assert code_of(lambda: cache.put(source, decoder.astype(np.int64))) == "atlas_mismatch"
    assert code_of(lambda: cache.get("not-a-digest")) == "invalid_artifact"
    assert code_of(lambda: cache.get("../" + "c" * 61)) == "invalid_artifact"
    poisoned = decoder.copy()
    poisoned[2, 1] = np.inf
    other = atlas_source(6, 4, "ef" * 32)
    assert code_of(lambda: cache.put(other, poisoned)) == "invalid_artifact"
    assert cache.get(other.tensor_sha256) is None


def test_decoder_cache_detects_tampering(tmp_path: Path) -> None:
    cache = DecoderCache(tmp_path)
    decoder = np.random.default_rng(13).normal(size=(3, 4)).astype(np.float32)
    source = atlas_source(3, 4, "12" * 32)
    cache.put(source, decoder)
    stored = tmp_path / f"{source.tensor_sha256}.bin"
    manifest = tmp_path / f"{source.tensor_sha256}.json"
    original, description = stored.read_bytes(), manifest.read_text("utf-8")

    flipped = bytearray(original)
    flipped[17] ^= 0x40
    stored.write_bytes(bytes(flipped))
    assert code_of(lambda: cache.get(source.tensor_sha256)) == "artifact_mismatch"
    stored.write_bytes(original[:-4])
    assert code_of(lambda: cache.get(source.tensor_sha256)) == "artifact_mismatch"
    stored.write_bytes(original + bytes(4))
    assert code_of(lambda: cache.get(source.tensor_sha256)) == "invalid_artifact"
    stored.write_bytes(original)
    np.testing.assert_array_equal(cache.get(source.tensor_sha256), decoder)

    manifest.write_text(description.replace('"latents":3', '"latents":4'), "utf-8")
    assert code_of(lambda: cache.get(source.tensor_sha256)) == "invalid_artifact"
    manifest.write_text("{", "utf-8")
    assert code_of(lambda: cache.get(source.tensor_sha256)) == "invalid_artifact"

    # A manifest copied under another handle cannot vouch for that tensor.
    (tmp_path / f"{'34' * 32}.json").write_text(description, "utf-8")
    (tmp_path / f"{'34' * 32}.bin").write_bytes(original)
    assert code_of(lambda: cache.get("34" * 32)) == "artifact_mismatch"


def test_legend_names_the_two_rows_a_direction_is_built_from() -> None:
    rows = orthonormal_rows(6, 16)
    # Row lengths differ on purpose: a cosine must not reward a longer decoder row.
    decoder = (rows * np.array([1.0, 5.0, 0.25, 2.0, 3.0, 7.0])[:, None]).astype(np.float32)
    source = atlas_source(6, 16)
    direction = 0.8 * rows[4] + 0.6 * rows[1]

    two = legend(direction, decoder, source, top_k=2)
    assert [entry.index for entry in two.entries] == [4, 1]
    assert [entry.cosine for entry in two.entries] == pytest.approx([0.8, 0.6], abs=1e-6)
    assert two.explained == pytest.approx(1.0, abs=1e-6)
    assert two.status == "pencil" and two.source == source
    assert two.method == "cosine_top_k+least_squares_span"
    assert all(e.url is None and e.label is None and e.label_source is None for e in two.entries)

    one = legend(direction, decoder, source, top_k=1)
    assert [entry.index for entry in one.entries] == [4]
    assert one.explained == pytest.approx(0.64, abs=1e-6)

    # Sign is kept, rank is by magnitude, and the direction's own length is irrelevant.
    for scale in (1.0, 1e-30, 1e300):
        signed = legend(scale * (0.6 * rows[1] - 0.8 * rows[4]), decoder, source, top_k=2)
        assert [entry.index for entry in signed.entries] == [4, 1]
        assert [entry.cosine for entry in signed.entries] == pytest.approx([-0.8, 0.6], abs=1e-6)
        assert signed.explained == pytest.approx(1.0, abs=1e-6)

    every = legend(direction.astype(np.float32), decoder, source, top_k=32)
    assert [entry.index for entry in every.entries][:2] == [4, 1] and len(every.entries) == 6
    assert all(abs(entry.cosine) < 1e-6 for entry in every.entries[2:])


def test_legend_matches_a_direct_computation_across_chunk_sizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(21)
    decoder = rng.normal(size=(37, 5)).astype(np.float32)
    direction = rng.normal(size=5)
    source = atlas_source(37, 5)

    wide = decoder.astype(np.float64)
    unit = direction / np.linalg.norm(direction)
    cosines = wide @ unit / np.linalg.norm(wide, axis=1)
    order = sorted(range(37), key=lambda index: (-abs(cosines[index]), index))[:3]
    basis, _ = np.linalg.qr(wide[order].T)
    explained = float(np.sum((basis.T @ unit) ** 2))
    assert 0.3 < explained < 0.999

    results = [legend(direction, decoder, source, top_k=3)]
    for elements in (5, 12, 64):
        monkeypatch.setattr(atlas, "_BLOCK_ELEMENTS", elements)
        results.append(legend(direction, decoder, source, top_k=3))
    for result in results:
        assert [entry.index for entry in result.entries] == order
        assert [entry.cosine for entry in result.entries] == pytest.approx(
            [cosines[index] for index in order], abs=1e-12,
        )
        assert result.explained == pytest.approx(explained, abs=1e-12)


def test_legend_breaks_ties_by_index_and_treats_zero_rows_as_cosine_zero() -> None:
    e0, e1, zero = np.eye(3, dtype=np.float32)[0], np.eye(3, dtype=np.float32)[1], np.zeros(3)
    tied = np.array([e0, e1, -e0, 2 * e0, e1], dtype=np.float32)
    result = legend([1.0, 1.0, 0.0], tied, atlas_source(5, 3), top_k=5)
    assert [entry.index for entry in result.entries] == [0, 1, 2, 3, 4]
    assert [entry.cosine for entry in result.entries] == pytest.approx(
        [2**-0.5, 2**-0.5, -(2**-0.5), 2**-0.5, 2**-0.5]
    )
    assert result.explained == pytest.approx(1.0)

    later = np.array([e1, e0, -e0, e0], dtype=np.float32)
    assert [e.index for e in legend(e0, later, atlas_source(4, 3), top_k=3).entries] == [1, 2, 3]

    # Repeated rows span one dimension: the fit must not count it twice.
    repeated = np.array([e0, e0, e1], dtype=np.float32)
    twice = legend([0.8, 0.6, 0.0], repeated, atlas_source(3, 3), top_k=2)
    assert [entry.index for entry in twice.entries] == [0, 1]
    assert twice.explained == pytest.approx(0.64)

    sparse = np.array([zero, e0, zero], dtype=np.float32)
    with np.errstate(all="raise"):
        named = legend(e0, sparse, atlas_source(3, 3), top_k=3)
    assert [(e.index, e.cosine) for e in named.entries] == [(1, 1.0), (0, 0.0), (2, 0.0)]
    assert named.explained == pytest.approx(1.0)

    blank = legend(e0, np.zeros((2, 3), dtype=np.float32), atlas_source(2, 3), top_k=2)
    assert [(entry.index, entry.cosine) for entry in blank.entries] == [(0, 0.0), (1, 0.0)]
    assert blank.explained == 0.0


def test_legend_refuses_unusable_directions_decoders_and_sizes() -> None:
    decoder = np.eye(4, dtype=np.float32)
    source = atlas_source(4, 4)

    def attempt(direction: Any, matrix: Any = decoder, **options: Any) -> str:
        return code_of(lambda: legend(direction, matrix, source, **options))

    assert attempt([0.0, 0.0, 0.0, 0.0]) == "invalid_request"
    assert attempt([1.0, float("nan"), 0.0, 0.0]) == "invalid_request"
    assert attempt([1.0, float("inf"), 0.0, 0.0]) == "invalid_request"
    assert attempt([[1.0, 0.0, 0.0, 0.0]]) == "invalid_request"
    assert attempt(["a", "b", "c", "d"]) == "invalid_request"
    assert attempt([1.0, 0.0, 0.0]) == "atlas_mismatch"
    assert attempt([1.0, 0.0, 0.0, 0.0, 0.0]) == "atlas_mismatch"
    for top_k in (0, 33, -1, True, 2.0):
        assert attempt([1.0, 0.0, 0.0, 0.0], top_k=top_k) == "invalid_request"
    assert attempt([1.0, 0.0, 0.0, 0.0], decoder[:3]) == "atlas_mismatch"
    assert attempt([1.0, 0.0, 0.0, 0.0], np.eye(4, dtype=np.int64)) == "atlas_mismatch"
    broken = decoder.copy()
    broken[3, 3] = np.nan
    assert attempt([1.0, 0.0, 0.0, 0.0], broken) == "invalid_artifact"


def test_legend_records_a_label_only_with_its_source() -> None:
    rows = orthonormal_rows(4, 4).astype(np.float32)
    source = atlas_source(4, 4)
    direction = 0.8 * rows[2] + 0.6 * rows[0]
    seen: list[object] = []

    def label_for(index: int) -> tuple[str, str] | None:
        seen.append(index)
        return ("capital cities", "neuronpedia:qwen3-0.6b/3-transcoder-hp") if index == 2 else None

    result = legend(
        direction, rows, source, top_k=2, label_for=label_for,
        url_for=lambda index: neuronpedia_url("qwen3-0.6b", "3-transcoder-hp", index),
    )
    first, second = result.entries
    assert seen == [2, 0] and all(type(index) is int for index in seen)
    assert first.url == "https://www.neuronpedia.org/qwen3-0.6b/3-transcoder-hp/2"
    assert (first.label, first.label_source) == (
        "capital cities", "neuronpedia:qwen3-0.6b/3-transcoder-hp",
    )
    assert second.url == "https://www.neuronpedia.org/qwen3-0.6b/3-transcoder-hp/0"
    assert second.label is None and second.label_source is None
    assert result.status == "pencil"

    def attempt(**options: Any) -> str:
        return code_of(lambda: legend(direction, rows, source, top_k=1, **options))

    assert attempt(label_for=lambda index: ("capital cities", "")) == "invalid_request"
    assert attempt(label_for=lambda index: ("", "neuronpedia")) == "invalid_request"
    assert attempt(label_for=lambda index: "capital cities") == "invalid_request"
    assert attempt(label_for=lambda index: ("capital cities",)) == "invalid_request"
    assert attempt(label_for=lambda index: ("x" * 501, "neuronpedia")) == "invalid_request"
    assert attempt(label_for=lambda index: ("x", "s" * 201)) == "invalid_request"
    assert attempt(url_for=lambda index: "javascript:alert(1)") == "invalid_request"
    assert attempt(url_for=lambda index: "https://example.org/a b") == "invalid_request"
    assert attempt(url_for=lambda index: "https://example.org/" + "a" * 500) == "invalid_request"
    assert attempt(url_for=lambda index: 7) == "invalid_request"


def test_neuronpedia_url_validates_every_path_component() -> None:
    assert neuronpedia_url("gemma-2-2b", "12-gemmascope-res-16k", 4096) == (
        "https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/4096"
    )
    assert neuronpedia_url("gpt2-small", "0-res_post_32k-oai", 0).endswith("/0-res_post_32k-oai/0")
    assert neuronpedia_url("a" * 80, "llama3.1-8b", 2**31 - 1).endswith(f"/{2**31 - 1}")
    bad_parts: list[Any] = [
        "", "a/b", "..", ".hidden", "-flag", "a b", "gemma\n", "a?x=1", "a#b", "%2e%2e",
        "café", "a\\b", "a" * 81, None, 7,
    ]
    for part in bad_parts:
        assert code_of(lambda p=part: neuronpedia_url(p, "0-res-jb", 1)) == "invalid_request"
        assert code_of(lambda p=part: neuronpedia_url("gpt2-small", p, 1)) == "invalid_request"
    bad_indices: list[Any] = [-1, True, 1.0, "3", None, 2**31]
    for index in bad_indices:
        refused = code_of(lambda i=index: neuronpedia_url("gpt2-small", "0-res-jb", i))
        assert refused == "invalid_request"


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.asked: list[int] = []

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, amount: int = -1) -> bytes:
        self.asked.append(amount)
        return self.body if amount < 0 else self.body[:amount]


class FakeOpener:
    def __init__(self, outcome: bytes | Exception) -> None:
        self.outcome = outcome
        self.requests: list[Any] = []
        self.timeouts: list[float] = []
        self.responses: list[FakeResponse] = []

    def __call__(self, request: Any, timeout: float) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        self.responses.append(FakeResponse(self.outcome))
        return self.responses[-1]


def feature_json(*descriptions: object) -> bytes:
    explanations = [{"description": text, "explanationModelName": "m"} for text in descriptions]
    return json.dumps({"modelId": "gemma-2-2b", "explanations": explanations}).encode("utf-8")


def test_label_fetcher_reads_the_first_explanation_with_one_anonymous_get() -> None:
    opener = FakeOpener(feature_json("  references to\n capital\tcities ", "second opinion"))
    fetch = neuronpedia_labels("gemma-2-2b", "12-gemmascope-res-16k", opener=opener, timeout=2.5)
    assert opener.requests == []
    assert fetch(4096) == (
        "references to capital cities", "neuronpedia:gemma-2-2b/12-gemmascope-res-16k",
    )
    (request,) = opener.requests
    assert request.full_url == (
        "https://www.neuronpedia.org/api/feature/gemma-2-2b/12-gemmascope-res-16k/4096"
    )
    assert request.header_items() == [("User-agent", "nrly")]
    assert request.get_method() == "GET" and request.data is None
    assert opener.timeouts == [2.5]
    assert opener.responses[0].asked == [MAX_LABEL_RESPONSE_BYTES + 1]

    default = FakeOpener(feature_json("x"))
    assert neuronpedia_labels("gpt2-small", "0-res-jb", opener=default)(3) == (
        "x", "neuronpedia:gpt2-small/0-res-jb",
    )
    assert default.timeouts == [10.0]


@pytest.mark.parametrize(
    "outcome",
    [
        b'{"explanations": [{"description": "cut off',
        b"\xff\xfe not json",
        b"",
        b"[]",
        b'"capital cities"',
        b"{}",
        b'{"explanations": null}',
        b'{"explanations": []}',
        b'{"explanations": {"description": "not a list"}}',
        b'{"explanations": ["capital cities"]}',
        b'{"explanations": [{"text": "another field"}]}',
        b'{"explanations": [{"description": 7}]}',
        b'{"explanations": [{"description": " \\n\\t "}]}',
        b'{"explanations": [{"description": null}, {"description": "second"}]}',
        b"[" * 100_000,
        urllib.error.URLError("unreachable"),
        urllib.error.HTTPError("https://www.neuronpedia.org", 404, "Not Found", Message(), None),
        TimeoutError("slow"),
        ConnectionResetError("reset"),
        ValueError("unexpected"),
        RuntimeError("anything"),
    ],
)
def test_label_fetcher_returns_none_and_never_raises(outcome: bytes | Exception) -> None:
    opener = FakeOpener(outcome)
    assert neuronpedia_labels("gpt2-small", "0-res-jb", opener=opener)(7) is None
    assert len(opener.requests) == 1


def test_label_fetcher_bounds_the_response_and_the_label() -> None:
    body = feature_json("capital cities")
    padded = body + b" " * (MAX_LABEL_RESPONSE_BYTES - len(body))
    exact, over = FakeOpener(padded), FakeOpener(padded + b" ")
    assert len(padded) == MAX_LABEL_RESPONSE_BYTES
    assert neuronpedia_labels("gpt2-small", "0-res-jb", opener=exact)(7) is not None
    # The same valid JSON one byte past the bound is dropped, and no more than that is read.
    assert neuronpedia_labels("gpt2-small", "0-res-jb", opener=over)(7) is None
    assert over.responses[0].asked == [MAX_LABEL_RESPONSE_BYTES + 1]

    class Broken(FakeResponse):
        def read(self, amount: int = -1) -> bytes:
            raise OSError("connection lost mid-body")

    class BrokenOpener(FakeOpener):
        def __call__(self, request: Any, timeout: float) -> FakeResponse:
            return Broken(b"")

    assert neuronpedia_labels("gpt2-small", "0-res-jb", opener=BrokenOpener(b""))(7) is None

    long = neuronpedia_labels("gpt2-small", "0-res-jb", opener=FakeOpener(feature_json("x" * 900)))
    label = long(7)
    assert label is not None and label[0] == "x" * 497 + "..." and len(label[0]) == 500
    hostile = FakeOpener(feature_json("\x1b[31mred\x00 team‮\r\nnext"))
    assert neuronpedia_labels("gpt2-small", "0-res-jb", opener=hostile)(7) == (
        "[31mred team next", "neuronpedia:gpt2-small/0-res-jb",
    )

    # A fetched label always fits a legend entry.
    rows = np.eye(3, dtype=np.float32)
    named = legend([0.0, 1.0, 0.0], rows, atlas_source(3, 3), top_k=1, label_for=long)
    assert named.entries[0].index == 1 and named.entries[0].label == "x" * 497 + "..."
    assert named.entries[0].label_source == "neuronpedia:gpt2-small/0-res-jb"


def test_label_fetcher_validates_its_configuration_but_not_by_raising_per_feature() -> None:
    opener = FakeOpener(feature_json("x"))
    fetch = neuronpedia_labels("gpt2-small", "0-res-jb", opener=opener)
    bad_indices: list[Any] = [-1, True, 1.5, "7", None, 2**31]
    for index in bad_indices:
        assert fetch(index) is None
    assert opener.requests == []

    assert code_of(lambda: neuronpedia_labels("gpt2/small", "0-res-jb")) == "invalid_request"
    assert code_of(lambda: neuronpedia_labels("gpt2-small", "../admin")) == "invalid_request"
    bad_timeouts: list[Any] = [0, -1.0, 61, float("nan"), float("inf"), True, "10"]
    for timeout in bad_timeouts:
        refused = code_of(
            lambda t=timeout: neuronpedia_labels("gpt2-small", "0-res-jb", timeout=t)
        )
        assert refused == "invalid_request"
