from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nrly.backends import load_backend
from nrly.contracts import (
    CaptureSpec,
    CaseSpec,
    GenerationSpec,
    ModelSpec,
    NrlyError,
    ProgramSpec,
)


def test_tiny_baseline_repair_identity_and_replay() -> None:
    case = CaseSpec(id="sum", input="[2, 4]", expected="6", scorer="numeric")
    with load_backend(ModelSpec()) as backend:
        identity = backend.identity
        baseline = backend.run_case(case, ProgramSpec(), [], GenerationSpec())
        repaired = backend.run_case(
            case,
            ProgramSpec(kind="weight_scale", target="output.weight", factor=2),
            [],
            GenerationSpec(),
        )
        replay = backend.run_case(case, ProgramSpec(), [], GenerationSpec())
        assert baseline.output == replay.output == "3"
        assert repaired.output == "6"
        assert baseline.base_unchanged and repaired.base_unchanged and replay.base_unchanged
        assert baseline.weight_changes == replay.weight_changes == []
        change = repaired.weight_changes[0]
        assert change.target == "output.weight"
        assert change.shape == [1, 2]
        assert change.changed_elements == change.total_elements == 2
        assert change.before_sha256 != change.after_sha256
        assert change.before_preview == [0.5, 0.5]
        assert change.after_preview == [1, 1]
        assert identity.details["fixture"] is True
        assert identity.details["capture_semantics"]
        assert {item.name for item in backend.describe()} >= {"hidden", "output.weight"}
        assert backend.read_tensor("output.weight")["values"] == [0.5, 0.5]
        assert backend.inspect_weight("output.weight").values == [0.5, 0.5]
    with load_backend(ModelSpec()) as other:
        assert other.identity == identity


def test_tiny_captures_real_activation() -> None:
    with load_backend(ModelSpec()) as backend:
        result = backend.run_case(
            CaseSpec(id="sum", input="[2, 4]"),
            ProgramSpec(kind="activation_scale", target="hidden", factor=2),
            [CaptureSpec(target="hidden", max_elements=2)],
            GenerationSpec(),
        )
    assert result.output == "6"
    assert result.captures[0].values == [4, 8]
    assert result.captures[0].shape == [1, 2]
    assert not result.captures[0].truncated


def test_inspection_is_readonly_and_detects_base_mutation() -> None:
    with load_backend(ModelSpec()) as backend:
        identity = backend.identity
        backend.describe()
        assert backend.inspect_weight("output.weight", 1).values == [0.5]
        assert backend.inspect_weight("output.weight", 1).truncated
        backend.verify_base_unchanged()
        assert backend.identity == identity
        with torch.no_grad():
            next(backend.model.parameters()).add_(1)
        with pytest.raises(NrlyError) as error:
            backend.verify_base_unchanged()
        assert error.value.code == "restoration_failed"


def test_sparse_weight_patch_runs_through_backend_and_preserves_base() -> None:
    with load_backend(ModelSpec()) as backend:
        result = backend.run_case(
            CaseSpec(id="sum", input="[2, 4]"),
            ProgramSpec(
                kind="weight_patch", target="output.weight", shape=[1, 2], indices=[0], values=[1.5]
            ),
            [],
            GenerationSpec(),
        )
        assert result.output == "5"
        assert result.weight_changes[0].changed_elements == 1
        assert result.base_unchanged
        assert backend.inspect_weight("output.weight").values == [0.5, 0.5]
        backend.verify_base_unchanged()


@pytest.mark.parametrize(
    "value",
    ["oops", "[1]", "[1, -1]", "[true, 2]", "[1, NaN]", '["1", 2]', "[1e400, 1]"],
)
def test_tiny_rejects_malformed_inputs_and_restores(value: str) -> None:
    with load_backend(ModelSpec()) as backend:
        with pytest.raises(NrlyError) as error:
            backend.run_case(
                CaseSpec(id="bad", input=value),
                ProgramSpec(kind="weight_scale", target="output.weight", factor=2),
                [],
                GenerationSpec(),
            )
        assert error.value.code == "invalid_input"
        assert (
            backend.run_case(
                CaseSpec(id="good", input="[2, 4]"), ProgramSpec(), [], GenerationSpec()
            ).output
            == "3"
        )


def test_tiny_last_token_is_explicitly_unsupported() -> None:
    with load_backend(ModelSpec()) as backend:
        with pytest.raises(NrlyError) as error:
            backend.run_case(
                CaseSpec(id="sum", input="[2, 4]"),
                ProgramSpec(kind="activation_scale", target="hidden", factor=2, position="last"),
                [],
                GenerationSpec(),
            )
        assert error.value.code == "target_mismatch"
        assert (
            backend.run_case(
                CaseSpec(id="good", input="[2, 4]"), ProgramSpec(), [], GenerationSpec()
            ).output
            == "3"
        )


def test_closed_backend_cannot_execute() -> None:
    backend = load_backend(ModelSpec())
    backend.close()
    backend.close()
    with pytest.raises(NrlyError) as error:
        backend.run_case(CaseSpec(id="sum", input="[2, 4]"), ProgramSpec(), [], GenerationSpec())
    assert error.value.code == "backend_closed"


def test_native_hf_loader_generation_identity_and_weight_metadata(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "world": 4}
    raw_tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=5,
            n_positions=32,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=1,
        )
    )
    model.save_pretrained(tmp_path, safe_serialization=True)
    tokenizer.save_pretrained(tmp_path)
    spec = ModelSpec(
        backend="huggingface",
        model_id=str(tmp_path),
        revision="0" * 40,
    )
    case = CaseSpec(id="hf-fixture", input="hello world")
    generation = GenerationSpec(max_new_tokens=3)
    with load_backend(spec) as backend:
        baseline = backend.run_case(case, ProgramSpec(), [], generation)
        noop = backend.run_case(
            case,
            ProgramSpec(kind="weight_scale", target="transformer.h.0.mlp.c_proj.weight"),
            [CaptureSpec(target="transformer.h.0.mlp", max_elements=2)],
            generation,
        )
        assert baseline.output == noop.output
        assert noop.weight_changes[0].changed_elements == 0
        assert noop.weight_changes[0].before_sha256 == noop.weight_changes[0].after_sha256
        assert noop.captures[0].shape[-1] == 8
        assert noop.input_tokens == 2
        assert 1 <= noop.output_tokens <= 3
        assert noop.base_unchanged
        assert backend.identity.details["attention_implementation"] == "eager"


def test_hf_loader_rejects_pickle_checkpoint_without_deserializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(
            tokenizers.models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tmp_path)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(vocab_size=2, n_positions=8, n_embd=8, n_layer=1, n_head=1)
    )
    model.config.save_pretrained(tmp_path)
    torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
    load_calls = []

    def forbidden_load(*args: object, **kwargs: object) -> object:
        load_calls.append((args, kwargs))
        raise AssertionError("pickle deserialization must never be reached")

    monkeypatch.setattr(torch, "load", forbidden_load)
    spec = ModelSpec(backend="huggingface", model_id=str(tmp_path), revision="0" * 40)
    with pytest.raises(NrlyError) as error:
        load_backend(spec)
    assert error.value.code == "model_load_failed"
    assert load_calls == []
