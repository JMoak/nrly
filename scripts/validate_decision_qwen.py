"""Real cached Qwen execution + real sibling Ratchet adapter; authored smoke, not quality proof."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path

from nrly.artifacts import write_immutable
from nrly.authoring import WeightProgram
from nrly.contracts import ModelSpec, canonical_json, digest
from nrly.decision_contracts import (
    DecisionCase,
    DecisionFreeze,
    DecisionObservation,
    DecisionProtocol,
)
from nrly.decision_handoff import ShadowDecisionService, prepare_shadow_handoff, shadow_server
from nrly.decision_qualification import assess_decisions, run_decision_study, save_decision_run
from nrly.decision_report import render_decision_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/models"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("keep prior attempts; choose a new output directory")
    os.environ["HF_HOME"] = str(args.cache.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from nrly.backends.decision import CausalDecisionBackend
    from nrly.backends.huggingface import HuggingFaceBackend
    from nrly.kernel import tensor_digest

    rows = [
        ("fit", "The library opens at 9 am.", "The library opens at 9 am.", True),
        ("fit", "The ticket costs $12.", "The ticket price is $12.", True),
        ("fit", "The train arrives on Tuesday.", "The train arrives on Friday.", False),
        ("fit", "The box contains five red pens.", "The box contains blue pens.", False),
        ("evaluation", "The museum is closed on Mondays.", "The museum closes on Mondays.", True),
        ("evaluation", "Ravi ordered two notebooks.", "Ravi ordered two notebooks.", True),
        ("evaluation", "The meeting is in Room B.", "The meeting is in Room C.", False),
        ("evaluation", "The parcel weighs 3 kilograms.", "The parcel weighs 8 kilograms.", False),
        ("challenge", "Only members can enter.", "Nonmembers cannot enter.", True),
        (
            "challenge",
            "Delivery is free for orders over $40.",
            "A $50 order qualifies for free delivery.",
            True,
        ),
        ("challenge", "Some seats are reserved.", "All seats are reserved.", False),
        ("challenge", "The trial showed a correlation.", "The trial proved causation.", False),
    ]
    protocol = DecisionProtocol(
        question=(
            "Is the claim supported by the source, without adding or contradicting information?"
        ),
        label_origin="assistant_authored",
        max_elapsed_seconds=600,
        cases=tuple(
            DecisionCase(
                id=f"case-{i}",
                family=f"source-{i}",
                split=split,
                state=f"Source: {source}\nClaim: {claim}",
                truth=truth,
            )
            for i, (split, source, claim, truth) in enumerate(rows)
        ),
    )
    model_spec = ModelSpec.model_validate_json(Path("recipes/qwen-model.json").read_bytes())
    print(json.dumps({"stage": "loading", "model": model_spec.model_id}), flush=True)
    native = HuggingFaceBackend(model_spec)
    with native:
        backend = CausalDecisionBackend(native)
        program = (
            WeightProgram(backend.identity.fingerprint)
            .patch(
                "model.layers.0.mlp.down_proj.weight",
                [1024, 3072],
                list(range(16)),
                [0.0] * 16,
            )
            .build()
        )
        write_immutable(
            args.output,
            "private-spec.json",
            canonical_json(
                {
                    "model": model_spec.model_dump(mode="json"),
                    "program": program.model_dump(mode="json"),
                    "protocol": protocol.model_dump(mode="json"),
                }
            ).encode(),
        )
        count = 0

        def journal(observation: DecisionObservation) -> None:
            nonlocal count
            write_immutable(
                args.output / "private-journal",
                f"{count:04d}.json",
                canonical_json(observation).encode(),
            )
            count += 1
            print(
                json.dumps(
                    {"stage": "observation", "count": count, "error": observation.error_code}
                ),
                flush=True,
            )

        def freeze(value: DecisionFreeze) -> None:
            write_immutable(args.output, "freeze.json", canonical_json(value).encode())

        run = run_decision_study(backend, protocol, program, record=journal, on_freeze=freeze)
        save_decision_run(args.output, run)
        assessment = assess_decisions(run)
        write_immutable(args.output, "assessment.json", canonical_json(assessment).encode())
        write_immutable(args.output, "review.html", render_decision_report(assessment).encode())
        assert assessment.status == "complete", "inspect retained failed observations"
        assert tensor_digest(native.model) == native.identity.details["weights_sha256"]
        assert any(
            c.changed_elements > 0
            for o in run.observations
            if o.execution
            for c in o.execution.weight_changes
        )
        handoff = prepare_shadow_handoff(run)
        write_immutable(args.output, "shadow.json", canonical_json(handoff).encode())
        bun = shutil.which("bun")
        ratchet = Path("../ratchet").resolve()
        assert bun and (ratchet / "src/oracles/systemone.ts").is_file(), "Ratchet + Bun required"
        token = secrets.token_urlsafe(32)
        with shadow_server(ShadowDecisionService(backend, run), token) as url:
            result = subprocess.run(
                [bun, "tests/decision_ratchet.ts"],
                capture_output=True,
                text=True,
                timeout=90,
                env={
                    **os.environ,
                    "NRLY_TEST_ENDPOINT": url,
                    "NRLY_TEST_TOKEN": token,
                    "NRLY_TEST_MODEL": handoff.served_model,
                    "NRLY_TEST_STATE": protocol.cases[0].state,
                    "NRLY_TEST_QUESTION": protocol.question,
                    "NRLY_RATCHET_REPO": str(ratchet),
                },
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        assert result.returncode == 0, result.stderr
        conformance = json.loads(result.stdout)
        expected = next(
            o.execution for o in run.observations if o.case_id == "case-0" and o.variant == "patch"
        )
        assert expected and abs(conformance["probability"] - expected.probability) < 1e-6
        write_immutable(
            args.output,
            "ratchet-conformance.json",
            canonical_json(
                {
                    "run_digest": digest(run),
                    "result": conformance,
                    "ratchet_source_sha256": __import__("hashlib")
                    .sha256((ratchet / "src/oracles/systemone.ts").read_bytes())
                    .hexdigest(),
                }
            ).encode(),
        )
        print(
            json.dumps(
                {
                    "stage": "complete",
                    "recommendation": assessment.recommendation,
                    "ratchet_verified": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
