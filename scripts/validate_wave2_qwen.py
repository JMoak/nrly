"""Reproduce the bounded Qwen GPU integration acceptance; never a capability benchmark."""

from __future__ import annotations

import argparse
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
    CaseSpec,
    ExperimentSpec,
    GenerationSpec,
    ModelSpec,
    NrlyError,
    ProgramSpec,
    RunRecord,
    TensorSlice,
    WeightOperationSpec,
    WeightReadSpec,
    digest,
)
from nrly.runner import Runner
from nrly.store import Store, atomic_text, now, read_bundle

FIRST = "model.layers.0.mlp.down_proj.weight"
SECOND = "model.layers.1.mlp.down_proj.weight"
SHAPE = [1024, 3072]
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def save(path: Path, data: object) -> None:
    atomic_text(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def progress(stage: str, **details: object) -> None:
    print(json.dumps({"stage": stage, **details}, allow_nan=False), flush=True)


def candidates(fingerprint: str) -> tuple[ProgramSpec, ProgramSpec]:
    left = [0.0] * SHAPE[0]
    right = [0.0] * SHAPE[1]
    left[2] = 1
    right[17] = 1 / 1024
    composed = ProgramSpec(
        kind="pipeline",
        expected_base=fingerprint,
        operations=[
            WeightOperationSpec(
                kind="weight_low_rank",
                target=FIRST,
                shape=SHAPE,
                rank=1,
                left=left,
                right=right,
            ),
            WeightOperationSpec(
                kind="weight_slice_scale",
                target=FIRST,
                shape=SHAPE,
                factor=1.0078125,
                slices=[TensorSlice(start=2, stop=3), TensorSlice(start=17, stop=19)],
            ),
            WeightOperationSpec(
                kind="weight_sparse_add",
                target=SECOND,
                shape=SHAPE,
                indices=[3 * SHAPE[1] + 21],
                values=[1 / 2048],
            ),
        ],
    )
    cancellation = ProgramSpec(
        kind="pipeline",
        expected_base=fingerprint,
        operations=[
            WeightOperationSpec(
                kind="weight_slice_scale",
                target=FIRST,
                shape=SHAPE,
                factor=factor,
                slices=[TensorSlice(start=2, stop=3), TensorSlice(start=17, stop=19)],
            )
            for factor in (2, 0.5)
        ],
    )
    return composed, cancellation


def acceptance(root: Path, cache: Path, output: Path) -> None:
    # Set cache routing before importing transformers; no download or install occurs.
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from nrly.backends import load_backend
    from nrly.evaluation import evaluate_run
    from nrly.kernel import intervene, tensor_digest

    started = perf_counter()
    model = ModelSpec(
        backend="huggingface",
        model_id="Qwen/Qwen3-0.6B",
        revision=REVISION,
        device="cuda",
        dtype="bfloat16",
        local_files_only=True,
    )
    generation = GenerationSpec(max_new_tokens=16, temperature=0, seed=0, enable_thinking=False)
    cases = [
        CaseSpec(
            id="arithmetic",
            input="Reply with only the integer: 2 + 3 =",
            expected="5",
            split="validation",
        ),
        CaseSpec(id="format", input="Reply with exactly OK", expected="OK", split="validation"),
        CaseSpec(
            id="regression",
            input="What is the capital of France? Answer in one word.",
            expected="Paris",
            split="validation",
            role="regression",
        ),
    ]
    slice_read = WeightReadSpec(
        target=FIRST,
        max_elements=8,
        slices=[TensorSlice(start=2, stop=4), TensorSlice(start=16, stop=20)],
    )
    second_read = WeightReadSpec(target=SECOND, coordinates=[[3, 21], [3, 22]])
    progress("native_loading", model=model.model_id, revision=model.revision)
    native_started = perf_counter()
    with load_backend(model) as backend:
        identity = backend.identity
        base_digest = tensor_digest(backend.model)
        assert base_digest == identity.details["weights_sha256"]
        composed, cancellation = candidates(identity.fingerprint)
        shape_map = {
            item.name: item.shape for item in backend.describe() if item.kind == "parameter"
        }
        assert shape_map[FIRST] == shape_map[SECOND] == SHAPE
        baseline_slice = backend.inspect_weight_spec(slice_read)
        baseline_second = backend.inspect_weight_spec(second_read)
        offset = backend.inspect_weight_spec(
            WeightReadSpec(target=FIRST, offset=6160, max_elements=4)
        )
        indexed = backend.inspect_weight_spec(
            WeightReadSpec(target=FIRST, indices=[6162, 6161, 6160])
        )
        assert baseline_slice.selection_shape == [2, 4]
        assert baseline_slice.flat_indices == [6160, 6161, 6162, 6163, 9232, 9233, 9234, 9235]
        assert offset.values == baseline_slice.values[:4]
        assert indexed.values == [baseline_slice.values[i] for i in (2, 1, 0)]
        assert all(index > 4096 for index in baseline_slice.flat_indices)
        assert tensor_digest(backend.model) == base_digest
        progress(
            "native_inspected",
            fingerprint=identity.fingerprint,
            coordinates=baseline_slice.coordinates,
        )

        with intervene(backend.model, composed, identity.fingerprint) as composed_changes:
            edited_slice = backend.inspect_weight_spec(slice_read)
            edited_second = backend.inspect_weight_spec(second_read)
            edited_digest = tensor_digest(backend.model)
            assert edited_digest != base_digest
            assert edited_slice.values != baseline_slice.values
            assert edited_second.values[0] != baseline_second.values[0]
            assert composed_changes[0].after_sha256 == composed_changes[1].before_sha256
            net = sum(change.net_changed_elements or 0 for change in composed_changes)
            assert net > 0
        assert tensor_digest(backend.model) == base_digest
        assert backend.inspect_weight_spec(slice_read) == baseline_slice
        assert backend.inspect_weight_spec(second_read) == baseline_second

        with intervene(backend.model, cancellation, identity.fingerprint) as cancel_changes:
            cancellation_digest = tensor_digest(backend.model)
            assert cancellation_digest == base_digest
            assert cancel_changes[0].changed_elements > 0
            assert cancel_changes[1].changed_elements > 0
            assert cancel_changes[0].net_changed_elements is None
            assert cancel_changes[1].net_changed_elements == 0
            assert cancel_changes[1].net_before_sha256 == cancel_changes[1].net_after_sha256
        assert tensor_digest(backend.model) == base_digest

        try:
            wrong_base = cancellation.model_copy(update={"expected_base": "0" * 64})
            with intervene(backend.model, wrong_base, identity.fingerprint):
                raise AssertionError("wrong base was admitted")
        except NrlyError as error:
            assert error.code == "base_mismatch"
        assert tensor_digest(backend.model) == base_digest

        progress("native_generation", net_changed_elements=net)
        baseline_results = [backend.run_case(case, ProgramSpec(), [], generation) for case in cases]
        candidate_results = [backend.run_case(case, composed, [], generation) for case in cases]
        cancel_result = backend.run_case(cases[0], cancellation, [], generation)
        assert cancel_result.output == baseline_results[0].output
        assert all(
            result.base_unchanged
            for result in [*baseline_results, *candidate_results, cancel_result]
        )
        restored_digest = tensor_digest(backend.model)
        assert restored_digest == base_digest
        assert backend.inspect_weight_spec(slice_read) == baseline_slice
        native: dict[str, Any] = {
            "created_at": now(),
            "scope": "real-model integration acceptance; no capability claim",
            "model_identity": identity.model_dump(mode="json"),
            "generation": generation.model_dump(mode="json"),
            "cases": [case.model_dump(mode="json") for case in cases],
            "reads": {
                "offset": offset.model_dump(mode="json"),
                "indexed": indexed.model_dump(mode="json"),
                "slice_before": baseline_slice.model_dump(mode="json"),
                "slice_during": edited_slice.model_dump(mode="json"),
                "coordinates_before": baseline_second.model_dump(mode="json"),
                "coordinates_during": edited_second.model_dump(mode="json"),
            },
            "composed_program_digest": digest(composed),
            "composed_changes": [change.model_dump(mode="json") for change in composed_changes],
            "net_changed_elements": net,
            "cancellation_program": cancellation.model_dump(mode="json"),
            "cancellation_changes": [change.model_dump(mode="json") for change in cancel_changes],
            "base_state_sha256": base_digest,
            "edited_state_sha256": edited_digest,
            "cancellation_state_sha256": cancellation_digest,
            "restored_state_sha256": restored_digest,
            "wrong_base_rejected": True,
            "baseline_results": [result.model_dump(mode="json") for result in baseline_results],
            "candidate_results": [result.model_dump(mode="json") for result in candidate_results],
            "cancellation_result": cancel_result.model_dump(mode="json"),
            "elapsed_ms": (perf_counter() - native_started) * 1000,
        }
        save(output / "native.json", native)
    progress("native_complete", base_restored=True, net_changed_elements=net)

    spec = ExperimentSpec(
        name="Qwen3 0.6B wave2 composed weight integration; no improvement claimed",
        model=model,
        program=composed,
        cases=cases,
        generation=generation,
        weight_reads=[slice_read, second_read],
        budget=BudgetSpec(
            wall_seconds=180, max_cases=3, max_output_tokens=48, max_capture_bytes=131072
        ),
    )
    save(output / "recipe.json", spec.model_dump(mode="json"))
    with Runner(root, cache) as runner:
        submitted = runner.submit(spec)
        progress("runner_submitted", run_id=submitted.id)
        run = runner.wait(submitted.id, timeout=200)
        save(output / "runner-result.json", run.model_dump(mode="json"))
        assert run.status == "succeeded", run.error
        assert run.model_identity == identity
        assert run.spec.program.expected_base == identity.fingerprint
        assert [sample.output for sample in run.samples] == [
            result.output for result in candidate_results
        ]
        assert run.inspection is not None
        assert run.inspection.weights == [baseline_slice, baseline_second]
        assert all(sample.weight_changes == composed_changes for sample in run.samples)
        original_bundle = runner.store.export(run.id, output / "bundles/composed")
        assert read_bundle(original_bundle) == run
    progress(
        "runner_complete", run_id=run.id, elapsed_ms=run.elapsed_ms, capture_bytes=run.capture_bytes
    )

    replay_started = perf_counter()
    replay_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "nrly",
            "--root",
            str(root),
            "--model-cache",
            str(cache),
            "replay",
            str(original_bundle),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=200,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        check=False,
    )
    atomic_text(output / "replay.stdout.json", replay_process.stdout)
    if replay_process.stderr:
        atomic_text(output / "replay.stderr.txt", replay_process.stderr)
    assert replay_process.returncode == 0, replay_process.stderr or replay_process.stdout[-2000:]
    replay_payload = json.loads(replay_process.stdout)
    replayed = RunRecord.model_validate(replay_payload["run"])
    assert replay_payload["replay"]["identity_matches"] is True
    assert replay_payload["replay"]["outputs_match"] is True
    assert replayed.parent_run == run.id
    assert replayed.spec_digest == run.spec_digest
    assert replayed.inspection == run.inspection
    assert [sample.weight_changes for sample in replayed.samples] == [
        sample.weight_changes for sample in run.samples
    ]
    replay_bundle = Store(root).export(replayed.id, output / "bundles/replay")
    assert read_bundle(replay_bundle) == replayed
    bundle_bytes = sum(path.stat().st_size for path in (output / "bundles").rglob("*.json"))
    assert bundle_bytes < 1024 * 1024
    summary = {
        "created_at": now(),
        "status": "passed",
        "nrly_version": __version__,
        "scope": (
            "GPU integration acceptance on public smoke prompts; no demonstrated capability gain"
        ),
        "model_identity": identity.model_dump(mode="json"),
        "generation": generation.model_dump(mode="json"),
        "program_digest": digest(composed),
        "net_changed_elements": net,
        "base_state_sha256": base_digest,
        "edited_state_sha256": edited_digest,
        "restored_state_sha256": restored_digest,
        "base_restored": True,
        "cancellation_net_changed_elements": 0,
        "cancellation_state_sha256": cancellation_digest,
        "wrong_base_rejected": True,
        "baseline_outputs": [result.output for result in baseline_results],
        "candidate_outputs": [result.output for result in candidate_results],
        "candidate_evaluation": evaluate_run(run).model_dump(mode="json"),
        "runner": {
            "id": run.id,
            "elapsed_ms": run.elapsed_ms,
            "output_tokens": run.output_tokens,
            "capture_bytes": run.capture_bytes,
            "spec_digest": run.spec_digest,
        },
        "replay": {
            "id": replayed.id,
            "elapsed_ms": replayed.elapsed_ms,
            "outputs_match": True,
            "identity_matches": True,
            "inspection_matches": True,
            "weight_evidence_matches": True,
            "spec_digest_matches": True,
            "cli_wall_ms": (perf_counter() - replay_started) * 1000,
        },
        "portable_bundle_bytes": bundle_bytes,
        "elapsed_ms": (perf_counter() - started) * 1000,
        "limits": [
            "Three public smoke prompts; no sealed holdout",
            "Tiny authored edit is an execution canary",
            "Exact replay is specific to the recorded model/runtime/device fingerprint",
            "No parameter semantics, generalization, or superiority claim",
        ],
    }
    save(output / "summary.json", summary)
    progress(
        "acceptance_passed",
        runner=run.id,
        replay=replayed.id,
        bundle_bytes=bundle_bytes,
        elapsed_ms=summary["elapsed_ms"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".nrly-wave2-qwen"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/models"))
    parser.add_argument("--output", type=Path, default=Path("docs/validation/wave2-qwen"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be absent or empty; existing acceptance evidence is retained")
    output.mkdir(parents=True, exist_ok=True)
    try:
        acceptance(args.root.resolve(), args.cache.resolve(), output)
    except BaseException as error:
        save(
            output / "failure.json",
            {"created_at": now(), "type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
