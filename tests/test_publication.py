from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)


def check(root: Path) -> tuple[int, dict[str, object]]:
    process = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts/check_publication.py"), "--root", str(root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return process.returncode, json.loads(process.stdout)


def test_ignored_runtime_files_are_absent_but_force_tracked_files_are_blocked(
    tmp_path: Path,
) -> None:
    git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text((REPOSITORY / ".gitignore").read_text(), encoding="utf-8")
    runtime = tmp_path / ".nrly-private-session"
    runtime.mkdir()
    (runtime / "runs.sqlite3").write_bytes(b"private database fixture")
    (runtime / "worker.log").write_text("private worker output", encoding="utf-8")
    (tmp_path / ".env").write_text("EXAMPLE=private-fixture", encoding="utf-8")
    (tmp_path / "README.md").write_text("Public documentation", encoding="utf-8")
    code, report = check(tmp_path)
    assert code == 0 and report["findings"] == []
    git(tmp_path, "add", "-f", ".nrly-private-session/runs.sqlite3", ".env")
    code, report = check(tmp_path)
    assert code == 1
    assert {item["file"] for item in report["findings"]} == {  # type: ignore[union-attr]
        ".nrly-private-session/runs.sqlite3",
        ".env",
    }


def test_personal_paths_are_reported_without_echoing_the_private_value(tmp_path: Path) -> None:
    git(tmp_path, "init")
    sensitive = "C:/" + "Users/" + "private-review-user/credential.txt"
    (tmp_path / "notes.md").write_text(sensitive, encoding="utf-8")
    code, report = check(tmp_path)
    assert code == 1
    assert report["findings"] == [{"file": "notes.md", "line": 1, "rule": "personal_profile_path"}]
    assert "private-review-user" not in json.dumps(report)


@pytest.mark.parametrize(
    "filename", ["key.pem", "checkpoint.safetensors", "state.sqlite3-wal", "tensor.bin"],
)
def test_sensitive_untracked_files_are_rejected_without_reading_content(
    tmp_path: Path, filename: str
) -> None:
    git(tmp_path, "init")
    (tmp_path / filename).write_bytes(b"private test bytes")
    code, report = check(tmp_path)
    assert code == 1
    assert report["findings"] == [{"file": filename, "rule": "private_or_unsafe_path"}]


def test_force_tracked_tensor_blob_is_blocked_even_when_utf8_decodable(tmp_path: Path) -> None:
    git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text((REPOSITORY / ".gitignore").read_text(), encoding="utf-8")
    blob = tmp_path / "tensor.bin"
    blob.write_bytes(b"\0" * 16)
    code, report = check(tmp_path)
    assert code == 0 and report["findings"] == []
    git(tmp_path, "add", "-f", "tensor.bin")
    code, report = check(tmp_path)
    assert code == 1
    assert report["findings"] == [{"file": "tensor.bin", "rule": "private_or_unsafe_path"}]


def test_nul_bytes_are_binary_even_in_valid_utf8_text(tmp_path: Path) -> None:
    git(tmp_path, "init")
    (tmp_path / "unexpected.txt").write_bytes(b"valid UTF-8 text\0hidden binary bytes")
    code, report = check(tmp_path)
    assert code == 1
    assert report["findings"] == [{"file": "unexpected.txt", "rule": "binary_file_requires_review"}]
