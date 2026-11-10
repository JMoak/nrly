"""Fail on private state, credentials files, or profile paths in Git's publishable set.

This is a local publication guard, not a substitute for a maintained secret scanner
or dependency advisory audit. It inspects tracked files even when newly ignored.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

PRIVATE_DIRECTORIES = {
    ".git",
    ".cache",
    ".venv",
    "node_modules",
    "private_runs",
    ".aws",
    ".ssh",
    ".test-tmp",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
}
PRIVATE_FILES = (
    "*.sqlite*",
    "*.db",
    "*.db-*",
    "*.log",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.safetensors",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.pkl",
    "*.bin",
    "id_rsa",
    "id_ed25519",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    ".netrc",
    "_netrc",
    "runner.lock",
)
PROFILE_PATH = re.compile(r"(?i)(?:[A-Z]:[/\\]+Users[/\\]+|/home/|/Users/)[A-Z0-9][A-Z0-9_. -]*")
MAX_FILE_BYTES = 5 * 1024 * 1024


def publication_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    )
    return sorted({item.decode("utf-8") for item in result.stdout.split(b"\0") if item})


def inspect_files(root: Path, names: list[str]) -> list[dict[str, str | int]]:
    root = root.resolve()
    findings: list[dict[str, str | int]] = []
    for name in names:
        relative = PurePosixPath(name)
        parts = [part.lower() for part in relative.parts]
        basename = parts[-1] if parts else ""
        private = (
            any(part in PRIVATE_DIRECTORIES or part.startswith(".nrly") for part in parts)
            or any(fnmatch.fnmatchcase(basename, pattern) for pattern in PRIVATE_FILES)
            or (basename.startswith(".env") and basename not in {".env.example", ".env.sample"})
        )
        if relative.is_absolute() or ".." in relative.parts or private:
            findings.append({"file": name, "rule": "private_or_unsafe_path"})
            continue
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            findings.append({"file": name, "rule": "symlink_or_path_escape"})
            continue
        if not path.is_file():
            findings.append({"file": name, "rule": "missing_or_nonregular_file"})
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            findings.append({"file": name, "rule": "large_file_requires_review"})
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            findings.append({"file": name, "rule": "binary_file_requires_review"})
            continue
        if "\0" in content:
            findings.append({"file": name, "rule": "binary_file_requires_review"})
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            if PROFILE_PATH.search(line):
                findings.append(
                    {"file": name, "line": line_number, "rule": "personal_profile_path"}
                )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    names = publication_files(args.root)
    findings = inspect_files(args.root, names)
    print(json.dumps({"files_checked": len(names), "findings": findings}, indent=2))
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
