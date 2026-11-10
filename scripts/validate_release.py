"""Build offline, inspect archives, and exercise an installed wheel outside the checkout.

Run only after freezing source edits. Build dependencies must already be cached in
.cache/uv. Runtime dependencies come from .venv; only the wheel is installed into
the explicit --install-target. No archive is extracted by the inspection code.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from check_publication import PRIVATE_DIRECTORIES, PRIVATE_FILES

MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024

INSTALLED_CHECK = r'''
import json
import sys
from pathlib import Path

import nrly
from nrly.contracts import CaseSpec, ExperimentSpec, ProgramSpec
from nrly.runner import Runner

target, expected_version = Path(sys.argv[1]).resolve(), sys.argv[2]
module = Path(nrly.__file__).resolve()
if not module.is_relative_to(target) or nrly.__version__ != expected_version:
    raise RuntimeError("verification imported the wrong nrly installation or version")
case = CaseSpec(id="sum", input="[2,4]", expected="6", scorer="numeric")
base = ExperimentSpec(name="installed wheel original", cases=[case])
with Runner(Path("runs")) as runner:
    original = runner.wait(runner.submit(base).id)
    if original.status != "succeeded" or original.model_identity is None:
        raise RuntimeError("installed wheel baseline worker failed")
    program = ProgramSpec(
        kind="weight_scale", target="output.weight", factor=2,
        expected_base=original.model_identity.fingerprint,
    )
    repaired = runner.wait(runner.submit(ExperimentSpec(
        name="installed wheel repair", cases=[case], program=program,
    )).id)
    restored = runner.wait(runner.submit(base).id)
runs = [original, repaired, restored]
if any(run.status != "succeeded" or len(run.samples) != 1 for run in runs):
    raise RuntimeError("installed wheel worker did not finish one sample per run")
if [run.samples[0].output for run in runs] != ["3", "6", "3"]:
    raise RuntimeError("installed wheel repair/restoration outputs disagree")
if any(run.samples[0].provenance != "torch_fixture" for run in runs):
    raise RuntimeError("installed wheel did not execute the native Torch fixture")
if any(run.model_identity != original.model_identity for run in runs):
    raise RuntimeError("installed wheel changed base identity between runs")
changes = repaired.samples[0].weight_changes
if len(changes) != 1 or changes[0].changed_elements != 2:
    raise RuntimeError("installed wheel is missing real weight-change evidence")
print(json.dumps({
    "version": nrly.__version__, "module": str(module), "outside_checkout": True,
    "scope": "engineering_fixture", "outputs": [run.samples[0].output for run in runs],
    "run_ids": [run.id for run in runs], "base_fingerprint": original.model_identity.fingerprint,
    "changed_elements": changes[0].changed_elements,
}))
'''


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_snapshot(root: Path) -> dict[str, str]:
    paths = [
        path for path in (root / "src" / "nrly").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]
    paths.extend(root / name for name in ("pyproject.toml", "uv.lock", "bun.lock", "LICENSE"))
    return {path.relative_to(root).as_posix(): sha256(path.read_bytes()) for path in sorted(paths)}


def command(
    arguments: list[str], *, cwd: Path, environment: dict[str, str], log: Path,
    timeout: int = 180,
) -> str:
    completed = subprocess.run(
        arguments, cwd=cwd, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": arguments, "cwd": str(cwd)}) + "\n")
        stream.write(completed.stdout + completed.stderr + "\n")
    if completed.returncode:
        raise RuntimeError(f"command failed with exit {completed.returncode}; see {log.name}")
    return completed.stdout


def safe_member(name: str, canary_token: str) -> PurePosixPath:
    path = PurePosixPath(name)
    require(
        bool(path.parts) and not path.is_absolute() and ".." not in path.parts
        and "\\" not in name and not any(":" in part for part in path.parts),
        "archive contains an unsafe member path",
    )
    require(canary_token not in name, "archive contains a planted exclusion canary")
    parts = [part.lower() for part in path.parts]
    basename = parts[-1]
    require(
        not any(part in PRIVATE_DIRECTORIES or part.startswith(".nrly") for part in parts)
        and not any(fnmatch.fnmatchcase(basename, pattern) for pattern in PRIVATE_FILES)
        and not basename.endswith((".pyc", ".pyo", ".whl", ".tar.gz"))
        and not (basename.startswith(".env") and basename not in {".env.example", ".env.sample"}),
        "archive contains private state, a model, binary data, or an installed dependency",
    )
    return path


def inspect_archive(
    archive: Path, *, root: Path, version: str, canary_token: str,
) -> dict[str, Any]:
    license_bytes = (root / "LICENSE").read_bytes()
    require(
        license_bytes.startswith(b"MIT License"), "repository license is not the expected MIT text",
    )
    members: dict[str, bytes] = {}
    total = 0

    def accept(name: str, content: bytes) -> None:
        nonlocal total
        safe_member(name, canary_token)
        require(name not in members, "archive contains duplicate members")
        total += len(content)
        require(total <= MAX_ARCHIVE_BYTES, "archive uncompressed size exceeds inspection bound")
        require(canary_token.encode() not in content, "archive contains canary content")
        require(b"\0" not in content, "archive contains unexpected NUL-containing binary content")
        try:
            content.decode("utf-8")
        except UnicodeError as error:
            raise ValueError("archive contains unexpected non-UTF8 binary content") from error
        members[name] = content

    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as opened:
            for member in opened.infolist():
                safe_member(member.filename, canary_token)
                require(
                    (member.external_attr >> 16) & 0o170000 != 0o120000,
                    "wheel contains a symbolic link",
                )
                if member.is_dir():
                    continue
                require(
                    member.file_size <= MAX_MEMBER_BYTES, "wheel member exceeds inspection bound",
                )
                accept(member.filename, opened.read(member))
        prefix = f"nrly-{version}.dist-info/"
        require(
            members.get(prefix + "licenses/LICENSE") == license_bytes, "wheel MIT license differs",
        )
        require(prefix + "METADATA" in members, "wheel metadata/version directory is missing")
        require(
            f"Version: {version}\n".encode() in members[prefix + "METADATA"],
            "wheel metadata version differs from pyproject",
        )
        wheel_sources = {name for name in members if name.startswith("nrly/")}
        expected_sources = {
            name.removeprefix("src/")
            for name in source_snapshot(root) if name.startswith("src/nrly/")
        }
        require(wheel_sources == expected_sources, "wheel module set differs from current source")
        for name in wheel_sources:
            require(
                members[name] == (root / "src" / name).read_bytes(), "wheel source bytes differ",
            )
        source_count = len(wheel_sources)
    else:
        with tarfile.open(archive, "r:gz") as source_tar:
            for tar_member in source_tar:
                safe_member(tar_member.name, canary_token)
                require(
                    tar_member.isfile() or tar_member.isdir(),
                    "source archive contains a link/special file",
                )
                if tar_member.isdir():
                    continue
                require(
                    tar_member.size <= MAX_MEMBER_BYTES, "source member exceeds inspection bound",
                )
                stream = source_tar.extractfile(tar_member)
                require(stream is not None, "source archive member cannot be read")
                if stream is not None:
                    with stream:
                        accept(tar_member.name, stream.read(MAX_MEMBER_BYTES + 1))
        prefix = f"nrly-{version}/"
        require(
            all(name.startswith(prefix) for name in members), "source archive has unexpected roots",
        )
        require(
            members.get(prefix + "LICENSE") == license_bytes, "source archive MIT license differs",
        )
        require(
            members.get(prefix + "pyproject.toml") == (root / "pyproject.toml").read_bytes(),
            "source archive project metadata differs",
        )
        source_count = 0
        for relative in source_snapshot(root):
            if relative.startswith("src/nrly/"):
                require(
                    members.get(prefix + relative) == (root / relative).read_bytes(),
                    "source archive module bytes differ",
                )
                source_count += 1
    return {
        "path": str(archive), "sha256": sha256(archive.read_bytes()),
        "bytes": archive.stat().st_size, "members": len(members),
        "source_files_verified": source_count, "uncompressed_bytes": total,
        "mit_license_verified": True, "canaries_excluded": True,
        "private_state_excluded": True,
    }


def validate(root: Path, install_target: Path, report: dict[str, Any]) -> None:
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    require(metadata["project"]["name"] == "nrly", "validator expects the nrly project")
    version = str(metadata["project"]["version"])
    report["version"] = version
    executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    interpreter = root / ".venv" / executable
    require(interpreter.is_file(), "nrly .venv Python is unavailable")
    uv = shutil.which("uv")
    require(uv is not None, "uv must already be installed")
    require(not install_target.is_relative_to(root), "install target must be outside the checkout")
    require(
        not install_target.exists()
        or (install_target.is_dir() and not any(install_target.iterdir())),
        "install target must be absent or empty; existing installations are not deleted",
    )
    require((root / ".cache" / "uv").is_dir(), "offline build cache .cache/uv is unavailable")
    temporary = root / ".cache" / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "TEMP": str(temporary), "TMP": str(temporary), "TMPDIR": str(temporary),
        "UV_CACHE_DIR": str(root / ".cache" / "uv"), "UV_LINK_MODE": "copy",
        "UV_OFFLINE": "1", "UV_PYTHON_DOWNLOADS": "never",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
    })
    environment.pop("PYTHONPATH", None)
    log = root / ".cache" / "release-validation.log"
    log.write_text("", encoding="utf-8")
    snapshot = source_snapshot(root)
    token = uuid.uuid4().hex
    docs = root / "docs"
    canary_dir = docs / f".nrly-release-canary-{token}"
    canary_files = [
        docs / f".env.release-canary-{token}", canary_dir / "owned.json",
        docs / f"release-canary-{token}.bin",
    ]
    require(
        all(path.resolve().is_relative_to(docs) for path in canary_files), "unsafe canary paths",
    )
    created_files: list[Path] = []
    created_directory = False
    try:
        canary_dir.mkdir(exist_ok=False)
        created_directory = True
        for index, path in enumerate(canary_files):
            with path.open("xb") as stream:
                created_files.append(path)
                content = b"\0" * 16 if index == 2 else f"harmless release canary {token}".encode()
                stream.write(content)
        command(
            [str(uv), "build", "--offline", "--python", str(interpreter),
             "--wheel", "--sdist", "--out-dir", str(root / "dist"),
             "--no-create-gitignore", "--no-python-downloads"],
            cwd=root, environment=environment, log=log,
        )
        wheels = list((root / "dist").glob(f"nrly-{version}-*.whl"))
        require(len(wheels) == 1, "expected exactly one wheel for the current version")
        source_archive = root / "dist" / f"nrly-{version}.tar.gz"
        require(source_archive.is_file(), "current-version source archive is missing")
        report["archives"] = [
            inspect_archive(path, root=root, version=version, canary_token=token)
            for path in (wheels[0], source_archive)
        ]
    finally:
        # Only these UUID-owned literal paths are removed. Never recurse or clear a directory.
        for path in created_files:
            path.unlink(missing_ok=True)
        if created_directory and canary_dir.exists():
            canary_dir.rmdir()
    require(source_snapshot(root) == snapshot, "source changed during release validation")
    report["canary_files_removed"] = all(not path.exists() for path in canary_files)
    command(
        [str(uv), "pip", "install", "--offline", "--no-deps", "--python", str(interpreter),
         "--target", str(install_target), str(wheels[0]), "--no-python-downloads"],
        cwd=root, environment=environment, log=log,
    )
    scratch = install_target.parent / f".nrly-release-validation-{token}"
    scratch.mkdir(exist_ok=False)
    require(not scratch.resolve().is_relative_to(root), "installed check must run outside checkout")
    environment["PYTHONPATH"] = str(install_target)
    installed = command(
        [str(interpreter), "-c", INSTALLED_CHECK, str(install_target), version],
        cwd=scratch, environment=environment, log=log,
    )
    report["installed_check"] = json.loads(installed)
    report["installed_check"]["run_store"] = str(scratch / "runs")
    require(source_snapshot(root) == snapshot, "source changed after building the installed wheel")
    report["source_snapshot_sha256"] = sha256(json.dumps(snapshot, sort_keys=True).encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--install-target", type=Path, required=True)
    arguments = parser.parse_args()
    root, target = arguments.root.resolve(), arguments.install_target.resolve()
    report: dict[str, Any] = {"status": "running", "started_at": datetime.now(UTC).isoformat()}
    result_file = root / ".cache" / "release-validation.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        validate(root, target, report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    report["finished_at"] = datetime.now(UTC).isoformat()
    result_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
