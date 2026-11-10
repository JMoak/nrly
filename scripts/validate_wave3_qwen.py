"""Reproduce wave3 GPU integration smoke checks from the existing pinned local cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from nrly import __version__
from nrly.contracts import (
    BudgetSpec,
    CaptureSpec,
    CaseSpec,
    ChatMessage,
    DiagnosticSpec,
    ExperimentSpec,
    GenerationSpec,
    ModelSpec,
    ProgramSpec,
    RunRecord,
    TensorSlice,
    digest,
)
from nrly.runner import Runner
from nrly.store import Store, atomic_text, now, read_bundle
from nrly.studies import StudyEngine
from nrly.study_contracts import (
    AttemptRecord,
    CandidateSpec,
    ComparatorSpec,
    StudyBudget,
    StudyRecord,
    StudySpec,
)

TARGET = "model.layers.0.mlp.down_proj"
SHAPE = [1024, 3072]


def save(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"existing evidence is immutable: {path}")
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def progress(stage: str, **details: object) -> None:
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False, allow_nan=False), flush=True)


def stable_samples(run: RunRecord) -> list[dict[str, Any]]:
    return [sample.model_dump(mode="json", exclude={"elapsed_ms"}) for sample in run.samples]


def replay(bundle: Path, root: Path, cache: Path, output: Path, label: str) -> RunRecord:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrly",
            "--root",
            str(root),
            "--model-cache",
            str(cache),
            "replay",
            str(bundle),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=240,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        check=False,
    )
    atomic_text(output / f"{label}.stdout.json", process.stdout)
    if process.stderr:
        atomic_text(output / f"{label}.stderr.txt", process.stderr)
    assert process.returncode == 0, process.stderr or process.stdout[-3000:]
    payload = json.loads(process.stdout)
    assert payload["replay"]["replay_matches"] is True
    assert payload["replay"]["identity_matches"] is True
    return RunRecord.model_validate(payload["run"])


def acceptance(root: Path, cache: Path, output: Path, repository: Path) -> None:
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from nrly.backends import load_backend
    from nrly.evaluation import evaluate_run
    from nrly.kernel import tensor_digest

    started = perf_counter()
    model = ModelSpec.model_validate_json((repository / "recipes/qwen-model.json").read_bytes())
    assert model.local_files_only and model.device == "cuda"
    legacy_bundle = repository / "docs/validation/wave2-qwen/bundles/composed"
    legacy = read_bundle(legacy_bundle)
    cases = [
        CaseSpec(
            id="arithmetic",
            messages=[ChatMessage(role="user", content="Reply with only the integer: 2 + 3 =")],
            continuation="5",
        ),
        CaseSpec(
            id="capital",
            messages=[
                ChatMessage(role="system", content="Be concise."),
                ChatMessage(role="user", content="What is the capital of France?"),
            ],
            continuation="Paris",
        ),
    ]
    generation = GenerationSpec(max_new_tokens=16, enable_thinking=False)
    captures = [
        CaptureSpec(
            target=TARGET,
            source="input",
            invocation="first",
            token_positions=[0, 2],
            max_elements=64,
        )
    ]
    diagnostics = DiagnosticSpec(top_k=5, max_tokens=16)
    progress("native_loading", model=model.model_id, revision=model.revision)
    with load_backend(model) as backend:
        identity = backend.identity
        assert identity == legacy.model_identity, "legacy model/runtime fingerprint changed"
        base_digest = tensor_digest(backend.model)
        assert base_digest == identity.details["weights_sha256"]
        shapes = {item.name: item.shape for item in backend.describe() if item.kind == "parameter"}
        assert shapes[TARGET + ".weight"] == SHAPE
        program = ProgramSpec(
            kind="weight_slice_scale",
            target=TARGET + ".weight",
            shape=SHAPE,
            factor=0,
            slices=[TensorSlice(stop=SHAPE[0]), TensorSlice(stop=16)],
            expected_base=identity.fingerprint,
        )
        baseline = [
            backend.run_case(case, ProgramSpec(), captures, generation, diagnostics)
            for case in cases
        ]
        edited = [
            backend.run_case(case, program, captures, generation, diagnostics) for case in cases
        ]
        restored = [
            backend.run_case(case, ProgramSpec(), captures, generation, diagnostics)
            for case in cases
        ]
        assert all(result.diagnostics is not None for result in [*baseline, *edited, *restored])
        assert all(result.output_tokens == 0 and result.output == "" for result in edited)
        assert all(result.captures and any(result.captures[0].values) for result in edited)
        assert all(result.weight_changes[0].changed_elements > 0 for result in edited)
        assert [result.diagnostics for result in baseline] == [
            result.diagnostics for result in restored
        ]
        assert [result.captures for result in baseline] == [result.captures for result in restored]
        assert any(
            before.diagnostics != after.diagnostics
            for before, after in zip(baseline, edited, strict=True)
        )
        restored_digest = tensor_digest(backend.model)
        assert restored_digest == base_digest
        native = {
            "created_at": now(),
            "scope": "public integration smoke; no capability claim",
            "model_identity": identity.model_dump(mode="json"),
            "program": program.model_dump(mode="json"),
            "program_digest": digest(program),
            "cases": [case.model_dump(mode="json") for case in cases],
            "base_state_sha256": base_digest,
            "restored_state_sha256": restored_digest,
            "baseline": [result.model_dump(mode="json") for result in baseline],
            "edited": [result.model_dump(mode="json") for result in edited],
            "restored": [result.model_dump(mode="json") for result in restored],
        }
        save(output / "native-diagnostics.json", native)
    progress(
        "native_complete",
        base_restored=True,
        changed_elements=edited[0].weight_changes[0].changed_elements,
    )

    spec = ExperimentSpec(
        schema_version="0.3",
        name="Qwen wave3 diagnostic integration; no quality claim",
        model=model,
        program=program,
        generation=generation,
        cases=cases,
        captures=captures,
        diagnostics=diagnostics,
        budget=BudgetSpec(
            wall_seconds=180, max_cases=2, max_output_tokens=1, max_capture_bytes=131072
        ),
    )
    save(output / "diagnostic-recipe.json", spec.model_dump(mode="json"))
    with Runner(root, cache) as runner:
        submitted = runner.submit(spec)
        progress("diagnostic_worker_submitted", run_id=submitted.id)
        run = runner.wait(submitted.id, timeout=240)
        save(output / "diagnostic-run.json", run.model_dump(mode="json"))
        assert run.status == "succeeded", run.error
        assert run.model_identity == identity
        assert [sample.diagnostics for sample in run.samples] == [
            result.diagnostics for result in edited
        ]
        assert run.output_tokens == 0 and run.capture_bytes > 0
        evaluation = evaluate_run(run)
        assert evaluation.summary.unscored == len(cases) and evaluation.summary.passed == 0
        bundle = runner.store.export(run.id, output / "bundles/diagnostics")
    replayed = replay(bundle, root, cache, output, "diagnostic-replay")
    assert stable_samples(run) == stable_samples(replayed)
    assert replayed.spec_digest == run.spec_digest
    Store(root).export(replayed.id, output / "bundles/diagnostic-replay")
    progress("diagnostic_replay_complete", run_id=replayed.id, nonempty_probabilities=True)

    development = [
        CaseSpec(
            id="dev-add",
            messages=[
                ChatMessage(role="user", content="What is 2 + 3? Give only the final number.")
            ],
            expected="5",
            scorer="final_number",
            group="addition",
        ),
        CaseSpec(
            id="dev-subtract",
            input="What is 5 - 2? Give only the final number.",
            expected="3",
            scorer="final_number",
            role="regression",
            group="subtraction",
        ),
    ]
    holdout = [
        CaseSpec(
            id="smoke-add",
            input="What is 7 + 4? Give only the final number.",
            expected="11",
            scorer="final_number",
            split="test",
            group="addition",
        ),
        CaseSpec(
            id="smoke-subtract",
            input="What is 12 - 5? Give only the final number.",
            expected="7",
            scorer="final_number",
            split="test",
            role="regression",
            group="subtraction",
        ),
    ]
    study_spec = StudySpec(
        schema_version="0.2",
        name="Qwen sampled public smoke; no research holdout claim",
        model=model,
        generation=GenerationSpec(
            max_new_tokens=64, temperature=0.7, top_p=0.9, top_k=20, enable_thinking=False
        ),
        seeds=[17, 29],
        development_cases=development,
        comparators=[ComparatorSpec(name="concise", kind="prompt", prefix="Be concise. ")],
        mutable_targets=[TARGET + ".weight"],
        run_budget=BudgetSpec(
            wall_seconds=180, max_cases=2, max_output_tokens=128, max_capture_bytes=65536
        ),
        budget=StudyBudget(
            max_attempts=1,
            max_runs=16,
            max_case_executions=32,
            max_output_tokens=2048,
            max_capture_bytes=1048576,
            max_wall_seconds=2880,
        ),
    )
    save(output / "study-recipe.json", study_spec.model_dump(mode="json"))
    save(output / "public-smoke-holdout.json", [case.model_dump(mode="json") for case in holdout])
    with Runner(root, cache) as runner:
        engine = StudyEngine(runner)
        try:
            study = engine.create(study_spec, holdout)
            save(output / "study-created.json", study.model_dump(mode="json"))
            candidate = CandidateSpec(
                name="zero first sixteen down-projection columns", program=program
            )
            attempt = engine.submit_candidate(study.id, candidate)
            progress(
                "study_development_started",
                study_id=study.id,
                attempt_id=attempt.id,
                expected_workers=8,
            )
            attempt = engine.wait(study.id, attempt.id, timeout=1800)
            save(output / "study-attempt.json", attempt.model_dump(mode="json"))
            assert attempt.status == "succeeded", attempt.error_code
            assert attempt.assessment is not None and len(attempt.assessment.arms) == 8
            assert all(arm.complete for arm in attempt.assessment.arms)
            for arm in attempt.assessment.arms:
                save(
                    output / "development" / f"{arm.arm.replace(':', '_')}.json",
                    engine.development_samples(
                        study.id, attempt.id, arm.arm, limit=2, output_limit=2000
                    ),
                )
            progress("study_development_complete", verdict=attempt.assessment.verdict)
            engine.finalize(study.id, attempt.id)
            progress("study_final_started", public_smoke_holdout=True, expected_workers=8)
            final = engine.wait_final(study.id, timeout=1800)
            save(output / "study-final.json", final.model_dump(mode="json"))
            save(
                output / "study-events.json",
                [event.model_dump(mode="json") for event in engine.events(study.id)],
            )
            assert final.status == "completed"
            assert final.final_result is not None and len(final.final_result.arms) == 8
            assert all(arm.complete for arm in final.final_result.arms)
            assert len(final.final_result.comparisons) == 3
            assert all(pair.comparable for pair in final.final_result.comparisons)
            assert final.final_error_code in {None, "study_inconclusive"}
            assert final.model_identity == identity
            assert final.usage.runs == 16 and final.usage.case_executions == 32
            assert final.usage.output_tokens == 2048 and final.usage.final_runs_reserved == 8
            assert final.usage.capture_bytes == 1048576 and final.usage.wall_seconds == 2880
            private_runs = engine.private_runner.store.list(100)
            assert len(private_runs) == 16
            assert all(
                item.status == "succeeded" and item.model_identity == identity
                for item in private_runs
            )
        finally:
            engine.close()
    progress(
        "study_final_complete",
        verdict=final.final_result.verdict,
        recommendation=final.final_result.recommendation,
        workers=16,
    )

    complete_saved_evidence(root, cache, output, repository, started=started)


def complete_saved_evidence(
    root: Path, cache: Path, output: Path, repository: Path, *, started: float | None = None
) -> None:
    """Verify existing recorded stages and append remaining evidence without rerunning a study.

    This also permits explicit recovery from an acceptance-harness failure after all
    study workers completed. Existing evidence and ledgers are never overwritten.
    """
    from nrly.evaluation import evaluate_run

    completion_started = perf_counter()
    native = json.loads((output / "native-diagnostics.json").read_text(encoding="utf-8"))
    run = read_bundle(output / "bundles/diagnostics")
    replayed = read_bundle(output / "bundles/diagnostic-replay")
    final = StudyRecord.model_validate_json((output / "study-final.json").read_bytes())
    attempt = AttemptRecord.model_validate_json((output / "study-attempt.json").read_bytes())
    identity = run.model_identity
    assert identity is not None and identity.model_dump(mode="json") == native["model_identity"]
    assert run.status == replayed.status == "succeeded"
    assert run.model_identity == replayed.model_identity
    assert stable_samples(run) == stable_samples(replayed)
    assert run.spec_digest == replayed.spec_digest
    assert all(
        sample.diagnostics and sample.captures and any(sample.captures[0].values)
        for sample in run.samples
    )
    assert native["base_state_sha256"] == native["restored_state_sha256"]
    assert native["base_state_sha256"] == identity.details["weights_sha256"]
    assert [item["diagnostics"] for item in native["baseline"]] == [
        item["diagnostics"] for item in native["restored"]
    ]
    assert final.status == "completed" and final.model_identity == identity
    assert final.final_error_code in {None, "study_inconclusive"}
    assert final.final_result and len(final.final_result.arms) == 8
    assert all(arm.complete for arm in final.final_result.arms)
    assert len(final.final_result.comparisons) == 3
    assert all(pair.comparable for pair in final.final_result.comparisons)
    assert attempt.status == "succeeded" and attempt.assessment is not None
    assert final.usage.runs == 16 and final.usage.case_executions == 32
    assert final.usage.output_tokens == 2048 and final.usage.final_runs_reserved == 8
    assert final.usage.capture_bytes == 1048576 and final.usage.wall_seconds == 2880
    private_runs = Store(root / "studies/private_runs").list(100)
    assert len(private_runs) == 16
    assert all(
        item.status == "succeeded" and item.model_identity == identity for item in private_runs
    )
    for private_run in private_runs:
        save(
            output / "public-smoke-runs" / f"{private_run.id}.json",
            private_run.model_dump(mode="json"),
        )
    legacy_bundle = repository / "docs/validation/wave2-qwen/bundles/composed"
    legacy = read_bundle(legacy_bundle)
    legacy_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in legacy_bundle.iterdir()
        if path.is_file()
    }

    legacy_replayed = replay(legacy_bundle, root, cache, output, "legacy-wave2-replay")
    assert legacy_replayed.model_identity == legacy.model_identity == identity
    assert stable_samples(legacy_replayed) == stable_samples(legacy)
    assert legacy_replayed.inspection == legacy.inspection
    assert legacy_replayed.spec_digest == legacy.spec_digest
    assert legacy_hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in legacy_bundle.iterdir()
        if path.is_file()
    }
    progress("legacy_replay_complete", unchanged_fingerprint=True, strict_sample_match=True)
    summary = {
        "created_at": now(),
        "status": "passed",
        "nrly_version": __version__,
        "scope": "public real-model GPU integration smoke; no capability or generalization claim",
        "model_identity": identity.model_dump(mode="json"),
        "base_state_sha256": native["base_state_sha256"],
        "restored_state_sha256": native["restored_state_sha256"],
        "changed_elements": native["edited"][0]["weight_changes"][0]["changed_elements"],
        "diagnostic_cases": len(run.samples),
        "captured_values_per_case": len(run.samples[0].captures[0].values),
        "baseline_sum_logprobs": [
            item["diagnostics"]["sum_logprob"] for item in native["baseline"]
        ],
        "edited_sum_logprobs": [item["diagnostics"]["sum_logprob"] for item in native["edited"]],
        "diagnostic_evaluation": evaluate_run(run).model_dump(mode="json"),
        "diagnostic_fresh_worker_and_replay_exact": True,
        "diagnostic_run_id": run.id,
        "diagnostic_replay_id": replayed.id,
        "sampled_study_id": final.id,
        "sampled_study_workers": len(private_runs),
        "seeds": final.spec.seeds,
        "development_verdict": attempt.assessment.verdict,
        "final_verdict": final.final_result.verdict,
        "final_recommendation": final.final_result.recommendation,
        "recorded_final_error_code": final.final_error_code,
        "fully_scored_comparable_final": True,
        "study_usage": final.usage.model_dump(mode="json"),
        "legacy_fingerprint_and_samples_exact": True,
        "legacy_replay_id": legacy_replayed.id,
        "legacy_bundle_sha256": legacy_hashes,
        "elapsed_ms": (perf_counter() - started) * 1000 if started is not None else None,
        "completion_stage_ms": (perf_counter() - completion_started) * 1000,
        "harness_recovered_existing_study": started is None,
        "retained_harness_failure": "failure.json" if (output / "failure.json").exists() else None,
        "limitations": [
            "Four public sampled smoke prompts, two seeds, one authored edit",
            "Final cases are disclosed integration fixtures, not a secret research test",
            "Seed summaries are descriptive, not population confidence intervals",
            "No search, training, dependency installation, or model downloads",
        ],
    }
    save(output / "summary.json", summary)
    inventory = {
        str(path.relative_to(output)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    save(output / "artifact-inventory.json", inventory)
    progress("acceptance_passed", elapsed_ms=summary["elapsed_ms"], study_workers=16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".nrly-wave3-qwen"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/models"))
    parser.add_argument("--output", type=Path, default=Path("docs/validation/wave3-qwen"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be absent or empty; prior attempts are retained")
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("root must be absent or empty; prior run stores are retained")
    output.mkdir(parents=True, exist_ok=True)
    try:
        acceptance(root, args.cache.resolve(), output, Path(__file__).resolve().parents[1])
    except BaseException as error:
        save(
            output / "failure.json",
            {"created_at": now(), "type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
