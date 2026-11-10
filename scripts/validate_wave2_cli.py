"""Exercise the real CLI with a public engineering fixture; retain reviewable evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    root = repository / ".nrly" / ("wave2-cli-" + uuid.uuid4().hex)

    def invoke(*arguments: str) -> dict[str, Any]:
        process = subprocess.run(
            [sys.executable, "-m", "nrly.cli", "--root", str(root), *arguments],
            cwd=repository,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=180,
        )
        if process.returncode:
            raise RuntimeError(f"CLI {arguments[0]} failed: {process.stderr}\n{process.stdout}")
        data: dict[str, Any] = json.loads(process.stdout)
        return data

    def save(name: str, data: object) -> None:
        (destination / name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    demo = invoke("demo", "--export", str(destination / "repair-bundle"))
    assert demo["scope"] == "engineering_fixture" and demo["base_restored"]
    save("demo.json", demo)
    replay = invoke("replay", str(destination / "repair-bundle"))
    assert replay["replay"]["outputs_match"] and replay["replay"]["identity_matches"]
    save("replay.json", replay)
    from nrly.store import read_bundle

    candidate = read_bundle(destination / "repair-bundle")
    invoke(
        "report",
        candidate.id,
        "--baseline",
        demo["baseline"],
        "--output",
        str(destination / "repair-review.html"),
    )
    study = invoke(
        "study",
        "create",
        "recipes/studies/tiny-study.json",
        "--holdout",
        "recipes/studies/tiny-holdout.operator.json",
        "--request-id",
        "cli-study",
    )
    save("study-created.json", study)
    attempts = []
    for name in ("retain", "overscale", "repair"):
        attempt = invoke(
            "study",
            "submit",
            study["id"],
            f"recipes/studies/tiny-candidate-{name}.json",
            "--request-id",
            name,
        )
        assert attempt["status"] == "succeeded"
        attempts.append(attempt)
        save(f"attempt-{name}.json", attempt)
    assert attempts[0]["assessment"]["recommendation"] == "retain_original"
    assert attempts[1]["assessment"]["recommendation"] == "retain_original"
    assert attempts[2]["assessment"]["recommendation"] == "candidate_supported"
    final = invoke("study", "finalize", study["id"], attempts[-1]["id"])
    assert final["final_result"]["verdict"] == "observed_improvement"
    assert final["final_result"]["scope"] == "engineering_fixture"
    assert final["usage"]["runs"] == 16
    repeated = invoke("study", "finalize", study["id"], attempts[-1]["id"])
    assert repeated == final
    save("study-final.json", final)
    invoke("study", "export-context", study["id"], str(destination / "context-pack"))
    summary = {
        "status": "passed",
        "scope": "public engineering fixture; holdout is published and not unseen evidence",
        "run_store": str(root),
        "study_id": study["id"],
        "study_runs_reserved": final["usage"]["runs"],
        "attempt_recommendations": [a["assessment"]["recommendation"] for a in attempts],
        "final_verdict": final["final_result"]["verdict"],
        "finalization_idempotent": True,
        "base_restored": True,
        "bundle_replayed": True,
        "report": "repair-review.html",
        "context_pack": "context-pack",
    }
    save("summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
