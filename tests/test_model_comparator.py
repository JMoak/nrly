"""Real CPU/Transformers acceptance with engineered local causal checkpoints.

The tiny checkpoints emit a fixed word then EOS. This tests model loading/routing
and study integrity; it is engineering evidence, not a model-quality claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from nrly.api import Application
from nrly.contracts import BudgetSpec, CaseSpec, GenerationSpec, ModelSpec, NrlyError, ProgramSpec
from nrly.evaluation import compare_runs, evaluate_run
from nrly.runner import Runner
from nrly.studies import StudyEngine, StudyReader
from nrly.study_contracts import CandidateSpec, ComparatorSpec, StudyBudget, StudySpec


def checkpoint(path: Path, head_scale: float) -> ModelSpec:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    words = [
        "[UNK]",
        "[PAD]",
        "[EOS]",
        "answer",
        "target",
        "locality",
        "sealedtarget",
        "sealedlocal",
    ]
    raw = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {word: index for index, word in enumerate(words)}, unk_token="[UNK]"
        )
    )
    raw.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    config = transformers.GPT2Config(
        vocab_size=len(words),
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=1,
        bos_token_id=2,
        eos_token_id=2,
        pad_token_id=1,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = transformers.GPT2LMHeadModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        direction = torch.tensor([1.0, -1.0] * 4)
        model.transformer.wpe.weight[0].copy_(direction)
        model.transformer.wpe.weight[1:].copy_(-direction)
        model.transformer.ln_f.weight.fill_(1)
        model.lm_head.weight[3].copy_(head_scale * direction)
        model.lm_head.weight[2].copy_(-head_scale * direction)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    assert (path / "model.safetensors").is_file()
    assert not (path / "pytorch_model.bin").exists()
    return ModelSpec(
        backend="huggingface",
        model_id=str(path),
        revision="0" * 40,
        device="cpu",
        local_files_only=True,
    )


def test_local_hf_external_model_comparator_runs_workers_and_sealed_final(tmp_path: Path) -> None:
    original = checkpoint(tmp_path / "original", 2)
    external = checkpoint(tmp_path / "external", 3)
    original_bytes = (Path(original.model_id) / "model.safetensors").read_bytes()
    external_bytes = (Path(external.model_id) / "model.safetensors").read_bytes()
    assert hashlib.sha256(original_bytes).digest() != hashlib.sha256(external_bytes).digest()
    development = [
        CaseSpec(
            id="development-target", input="target", expected="answer", group="generalization"
        ),
        CaseSpec(
            id="development-local",
            input="locality",
            expected="answer",
            role="regression",
            group="locality",
        ),
    ]
    holdout = [
        CaseSpec(
            id="sealed-target",
            input="sealedtarget",
            expected="answer",
            split="test",
            group="generalization",
        ),
        CaseSpec(
            id="sealed-local",
            input="sealedlocal",
            expected="answer",
            split="test",
            role="regression",
            group="locality",
        ),
    ]
    spec = StudySpec(
        schema_version="0.2",
        name="engineered local HF comparator plumbing acceptance",
        model=original,
        generation=GenerationSpec(max_new_tokens=4),
        development_cases=development,
        comparators=[ComparatorSpec(name="external-checkpoint", kind="model", model=external)],
        mutable_targets=["lm_head.weight"],
        run_budget=BudgetSpec(wall_seconds=60, max_cases=2, max_output_tokens=8),
        budget=StudyBudget(
            max_attempts=1,
            max_runs=8,
            max_case_executions=16,
            max_output_tokens=64,
            max_wall_seconds=480,
        ),
    )
    workspace = tmp_path / "workspace"
    with Runner(workspace) as runner:
        engine = StudyEngine(runner)
        try:
            created = engine.create(spec, holdout)
            assert created.usage.final_runs_reserved == 4
            candidate = CandidateSpec(
                name="double-output-scale",
                program=ProgramSpec(kind="weight_scale", target="lm_head.weight", factor=2),
            )
            attempt = engine.wait(
                created.id, engine.submit_candidate(created.id, candidate).id, 120
            )
            assert attempt.status == "succeeded", attempt
            assessment = attempt.assessment
            assert assessment is not None and assessment.verdict == "no_change"
            assert assessment.recommendation == "retain_original"
            assert [arm.arm for arm in assessment.arms] == [
                "baseline",
                "no_op",
                "comparator:external-checkpoint",
                "candidate",
            ]
            assert all(arm.complete and arm.passed == 2 for arm in assessment.arms)
            assert all(
                pair.comparable and pair.improved_targets == pair.regressed_targets == 0
                for pair in assessment.comparisons
            )
            registered = engine.get(created.id)
            assert registered.model_identity is not None
            comparator_identity = registered.comparator_identities["external-checkpoint"]
            assert registered.model_identity.model_id == original.model_id
            assert comparator_identity.model_id == external.model_id
            assert registered.model_identity.fingerprint != comparator_identity.fingerprint
            for arm in assessment.arms:
                page = engine.development_samples(created.id, attempt.id, arm.arm)
                assert all(
                    sample["output"] == "answer"
                    and sample["finish_reason"] == "eos"
                    and sample["inference_truncated"] is False
                    for sample in page["samples"]
                )
            development_runs = engine.private_runner.store.list()
            assert len(development_runs) == 4
            external_runs = [run for run in development_runs if run.spec.model == external]
            original_runs = [run for run in development_runs if run.spec.model == original]
            assert len(external_runs) == 1 and len(original_runs) == 3
            assert not compare_runs(original_runs[0], external_runs[0]).comparable
            edited = next(run for run in original_runs if run.spec.program.kind == "weight_scale")
            assert all(sample.weight_changes[0].changed_elements == 16 for sample in edited.samples)
            engine.finalize(created.id, attempt.id)
            final = engine.wait_final(created.id, 120)
            assert final.status == "completed" and final.final_error_code is None
            assert final.final_result is not None and final.final_result.verdict == "no_change"
            assert final.model_identity == registered.model_identity
            assert final.comparator_identities == registered.comparator_identities
            assert (
                final.usage.attempts,
                final.usage.runs,
                final.usage.case_executions,
                final.usage.output_tokens,
            ) == (1, 8, 16, 64)
            all_runs = engine.private_runner.store.list()
            assert len(all_runs) == 8
            assert all(run.status == "succeeded" and evaluate_run(run).complete for run in all_runs)
            assert all(sample.output_tokens == 2 for run in all_runs for sample in run.samples)
            assert (Path(original.model_id) / "model.safetensors").read_bytes() == original_bytes
            assert (Path(external.model_id) / "model.safetensors").read_bytes() == external_bytes
            assert runner.store.list() == []
            reader = StudyReader(workspace)
            public = json.dumps(
                {
                    "record": reader.get(created.id).model_dump(mode="json"),
                    "context": reader.context_snapshot(created.id),
                    "attempt": reader.get_attempt(created.id, attempt.id).model_dump(mode="json"),
                }
            )
            assert "sealedtarget" not in public and "sealed-target" not in public
            assert all(run.id not in public for run in all_runs)
            app = Application.__new__(Application)
            app.runner, app.store = runner, runner.store
            for private in all_runs:
                with pytest.raises(NrlyError, match="no experiment run"):
                    app.get_run(private.id)
        finally:
            engine.close()
