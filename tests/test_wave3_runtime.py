from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from nrly.backends import load_backend
from nrly.backends.local import LocalExecution, LocalTorchBackend
from nrly.contracts import (
    BudgetSpec,
    CaptureSpec,
    CaseSpec,
    ChatMessage,
    DiagnosticSpec,
    ExperimentSpec,
    GenerationSpec,
    ModelSpec,
    NrlyError,
    ProgramSpec,
    WeightReadSpec,
    canonical_json,
)
from nrly.worker import execute


@pytest.fixture
def hf_spec(tmp_path: Path) -> ModelSpec:
    import tokenizers
    import transformers

    words = ["[UNK]", "[PAD]", "[EOS]", "hello", "world", "yes", "no", "system", "user"]
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
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        model = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=len(words),
                n_positions=64,
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
    return ModelSpec(backend="huggingface", model_id=str(tmp_path), revision="0" * 40)


def test_legacy_contract_serialization_omits_all_runtime_extensions() -> None:
    assert canonical_json(GenerationSpec()) == (
        '{"enable_thinking":false,"max_new_tokens":32,"seed":0,"temperature":0.0}'
    )
    assert canonical_json(CaseSpec(id="one", input="hello")) == (
        '{"expected":null,"id":"one","input":"hello","role":"target",'
        '"scorer":"exact","split":"discovery","tolerance":1e-06}'
    )
    assert CaptureSpec(target="x").model_dump() == {"target": "x", "max_elements": 16}
    for version in ("0.1", "0.2"):
        spec = ExperimentSpec(
            schema_version=version, name="legacy", cases=[CaseSpec(id="one", input="hello")]
        )
        assert "diagnostics" not in spec.model_dump()


@pytest.mark.parametrize(
    "generation",
    [
        GenerationSpec(max_new_tokens=513),
        GenerationSpec(top_k=50),
        GenerationSpec(top_p=1),
        GenerationSpec(min_new_tokens=0),
        GenerationSpec(repetition_penalty=1),
    ],
)
def test_runtime_features_require_new_schema(generation: GenerationSpec) -> None:
    case = CaseSpec(id="one", input="hello")
    with pytest.raises(ValidationError, match="0.3"):
        ExperimentSpec(name="old", cases=[case], generation=generation)
    ExperimentSpec(schema_version="0.3", name="new", cases=[case], generation=generation)


def test_chat_and_diagnostic_contract_ambiguities_fail() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        CaseSpec(id="missing")
    with pytest.raises(ValidationError, match="exactly one"):
        CaseSpec(id="both", input="hello", messages=[ChatMessage(role="user", content="world")])
    with pytest.raises(ValidationError, match="provided together"):
        ExperimentSpec(
            schema_version="0.3",
            name="missing",
            diagnostics=DiagnosticSpec(),
            cases=[CaseSpec(id="one", input="hello")],
        )
    with pytest.raises(ValidationError, match="activation capture selectors"):
        WeightReadSpec(target="x", source="input")
    with pytest.raises(ValidationError, match="unique"):
        CaptureSpec(target="x", token_positions=[1, 1])
    with pytest.raises(ValidationError, match="min_new_tokens"):
        GenerationSpec(max_new_tokens=2, min_new_tokens=3)


def test_teacher_forcing_matches_native_causal_logits_and_restores(hf_spec: ModelSpec) -> None:
    case = CaseSpec(id="diagnostic", input="hello world", continuation="yes no")
    with load_backend(hf_spec) as backend:
        with torch.inference_mode():
            native = backend.model(input_ids=torch.tensor([[3, 4, 5, 6]])).logits
            expected = native[0, 1:3].float().log_softmax(-1)
        result = backend.run_case(
            case, ProgramSpec(), [], GenerationSpec(), DiagnosticSpec(top_k=3)
        )
        score = result.diagnostics
        assert score is not None
        assert result.output == "" and result.output_tokens == 0
        assert result.finish_reason == "teacher_forced" and result.truncated is False
        assert score.prompt_tokens == score.continuation_tokens == 2
        assert score.tokenization == "separate_prompt_and_continuation"
        for index, observation in enumerate(score.tokens):
            token_id = index + 5
            assert observation.position == index + 2 and observation.token_id == token_id
            assert observation.logprob == pytest.approx(expected[index, token_id].item())
            assert observation.rank == int((expected[index] > expected[index, token_id]).sum()) + 1
            assert [value.token_id for value in observation.top_k] == expected[index].topk(
                3
            ).indices.tolist()
        edited = backend.run_case(
            case,
            ProgramSpec(kind="weight_scale", target="transformer.wpe.weight", factor=0),
            [],
            GenerationSpec(),
            DiagnosticSpec(),
        )
        assert edited.diagnostics.sum_logprob != pytest.approx(score.sum_logprob)
        assert edited.weight_changes[0].changed_elements > 0
        replay = backend.run_case(
            case, ProgramSpec(), [], GenerationSpec(), DiagnosticSpec(top_k=3)
        )
        assert replay.diagnostics == score
        backend.verify_base_unchanged()


def test_token_position_input_capture_matches_native_activation(hf_spec: ModelSpec) -> None:
    with load_backend(hf_spec) as backend:
        module = backend.model.get_submodule("transformer.h.0.mlp.c_proj")
        recorded = []
        hook = module.register_forward_pre_hook(
            lambda _module, args: recorded.append(args[0].clone())
        )
        with torch.inference_mode():
            backend.model(input_ids=torch.tensor([[3, 4, 5, 6]]))
        hook.remove()
        result = backend.run_case(
            CaseSpec(id="one", input="hello world", continuation="yes no"),
            ProgramSpec(),
            [
                CaptureSpec(
                    target="transformer.h.0.mlp.c_proj",
                    source="input",
                    invocation="first",
                    token_positions=[0, 2],
                    max_elements=64,
                )
            ],
            GenerationSpec(),
            DiagnosticSpec(),
        )
        capture = result.captures[0]
        assert capture.token_positions == [0, 2] and capture.invocation_index == 0
        assert capture.source == "input" and capture.selection_shape == [1, 2, 32]
        assert capture.values == pytest.approx(recorded[0][:, [0, 2], :].reshape(-1).tolist())
        assert not module._forward_pre_hooks


def test_chat_sampling_replays_seed_and_preserves_global_rng(hf_spec: ModelSpec) -> None:
    generation = GenerationSpec(
        max_new_tokens=4,
        min_new_tokens=4,
        temperature=0.8,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.1,
        seed=22,
    )
    case = CaseSpec(
        id="chat",
        messages=[
            ChatMessage(role="system", content="system"),
            ChatMessage(role="user", content="hello world"),
        ],
    )
    with load_backend(hf_spec) as backend:
        before = torch.get_rng_state().clone()
        first = backend.run_case(case, ProgramSpec(), [], generation)
        assert torch.equal(before, torch.get_rng_state())
        torch.rand(17)
        second = backend.run_case(case, ProgramSpec(), [], generation)
        assert first.output == second.output
        assert first.input_tokens == 3 and first.output_tokens == 4
        assert first.truncated is True and first.finish_reason == "length"


@pytest.mark.parametrize("mode", ["token_budget", "context", "capture", "nonfinite"])
def test_diagnostic_failure_removes_hooks_and_restores(
    hf_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    case = CaseSpec(id="bad", input="hello world", continuation="yes no")
    captures = []
    diagnostics = DiagnosticSpec()
    with load_backend(hf_spec) as backend:
        if mode == "token_budget":
            diagnostics = DiagnosticSpec(max_tokens=1)
        elif mode == "context":
            case = CaseSpec(id="bad", input="hello " * 64, continuation="yes no")
        elif mode == "capture":
            captures = [CaptureSpec(target="transformer.h.0.mlp", token_positions=[99])]
        else:
            original = backend.model.forward

            def broken(*args: object, **kwargs: object) -> object:
                output = original(*args, **kwargs)
                output.logits[:] = float("nan")
                return output

            monkeypatch.setattr(backend.model, "forward", broken)
        with pytest.raises(NrlyError) as error:
            backend.run_case(
                case,
                ProgramSpec(kind="weight_scale", target="transformer.wpe.weight", factor=2),
                captures,
                GenerationSpec(),
                diagnostics,
            )
        assert (
            error.value.code
            == {
                "token_budget": "budget_exceeded",
                "context": "context_exceeded",
                "capture": "target_mismatch",
                "nonfinite": "nonfinite_tensor",
            }[mode]
        )
        backend.verify_base_unchanged()
        assert not any(
            module._forward_hooks or module._forward_pre_hooks for module in backend.model.modules()
        )


def test_worker_diagnostics_account_as_evidence_not_generated_tokens(
    hf_spec: ModelSpec, tmp_path: Path
) -> None:
    spec = ExperimentSpec(
        schema_version="0.3",
        name="tf",
        model=hf_spec,
        cases=[CaseSpec(id="one", input="hello", continuation="world yes")],
        diagnostics=DiagnosticSpec(),
        budget=BudgetSpec(max_output_tokens=1),
    )
    spec_path, result_path = tmp_path / "spec.json", tmp_path / "result.json"
    spec_path.write_text(spec.model_dump_json(), encoding="utf-8")
    assert execute(spec_path, result_path) == 0
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["output_tokens"] == 0 and result["capture_bytes"] > 0
    assert result["samples"][0]["diagnostics"]["continuation_tokens"] == 2
    limited = spec.model_copy(update={"budget": BudgetSpec(max_capture_bytes=1)})
    spec_path.write_text(limited.model_dump_json(), encoding="utf-8")
    assert execute(spec_path, result_path) == 1
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["error"]["code"] == "budget_exceeded"
    assert result["samples"][0]["status"] == "failed"


def test_local_torch_adapter_runs_native_model_and_scoped_edits() -> None:
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)

    def forward(module: nn.Module, case: CaseSpec, _generation: GenerationSpec) -> LocalExecution:
        result = module(torch.tensor([[1.0, 2.0]]))
        if case.input == "fail":
            raise RuntimeError("callback failed after forward")
        return LocalExecution(output=str(result.item()))

    backend = LocalTorchBackend(
        model, forward, model_id="custom/network", revision="1", adapter_id="test/native-forward-v1"
    )
    assert backend.identity.backend == "local_torch"
    assert backend.identity.details["adapter_identity_source"] == "caller_declared"
    case = CaseSpec(id="one", input="run")
    with backend:
        assert backend.run_case(case, ProgramSpec(), [], GenerationSpec()).output == "3.0"
        edited = backend.run_case(
            case, ProgramSpec(kind="weight_scale", target="weight", factor=2), [], GenerationSpec()
        )
        assert edited.output == "6.0" and edited.weight_changes[0].changed_elements == 2
        with pytest.raises(RuntimeError, match="callback failed"):
            backend.run_case(
                CaseSpec(id="fail", input="fail"),
                ProgramSpec(kind="weight_scale", target="weight", factor=3),
                [],
                GenerationSpec(),
            )
        backend.verify_base_unchanged()
        assert backend.run_case(case, ProgramSpec(), [], GenerationSpec()).output == "3.0"
    assert model.weight.tolist() == [[1, 1]]
    with pytest.raises(NrlyError, match="closed"):
        backend.run_case(case, ProgramSpec(), [], GenerationSpec())


def test_local_torch_identity_and_rng_are_explicit() -> None:
    model = nn.Linear(1, 1)

    def sample(_model: nn.Module, _case: CaseSpec, _generation: GenerationSpec) -> LocalExecution:
        return LocalExecution(output=str(torch.rand(1).item()))

    with LocalTorchBackend(
        model, sample, model_id="custom/random", revision="1", adapter_id="v1"
    ) as backend:
        case = CaseSpec(id="one", input="go")
        before = torch.get_rng_state().clone()
        result = backend.run_case(case, ProgramSpec(), [], GenerationSpec(seed=42))
        assert torch.equal(before, torch.get_rng_state())
        assert (
            backend.run_case(case, ProgramSpec(), [], GenerationSpec(seed=42)).output
            == result.output
        )
        assert (
            backend.run_case(case, ProgramSpec(), [], GenerationSpec(seed=43)).output
            != result.output
        )
        with pytest.raises(NrlyError, match="causal diagnostics"):
            backend.run_case(
                CaseSpec(id="tf", input="go", continuation="next"),
                ProgramSpec(),
                [],
                GenerationSpec(),
                DiagnosticSpec(),
            )
    with pytest.raises(ValidationError):
        ModelSpec(backend="local_torch", model_id="custom/random", revision="1")


def test_diagnostic_logit_budget_preflights_native_selection_and_fallback(
    hf_spec: ModelSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = CaseSpec(id="one", input="hello world", continuation="yes no")
    with load_backend(hf_spec) as backend:
        original = backend.model.forward
        forwards = []

        def selected(*args: object, logits_to_keep: object = 0, **kwargs: object) -> object:
            forwards.append(logits_to_keep)
            return original(*args, logits_to_keep=logits_to_keep, **kwargs)

        monkeypatch.setattr(backend.model, "forward", selected)
        with pytest.raises(NrlyError, match="element budget"):
            backend.run_case(
                case, ProgramSpec(), [], GenerationSpec(), DiagnosticSpec(max_logits_elements=17)
            )
        assert forwards == []
        selective = backend.run_case(
            case, ProgramSpec(), [], GenerationSpec(), DiagnosticSpec(max_logits_elements=18)
        )
        assert forwards[0].tolist() == [1, 2]

        def fallback(*args: object, **kwargs: object) -> object:
            forwards.append(None)
            assert "logits_to_keep" not in kwargs
            return original(*args, **kwargs)

        monkeypatch.setattr(backend.model, "forward", fallback)
        with pytest.raises(NrlyError, match="element budget"):
            backend.run_case(
                case, ProgramSpec(), [], GenerationSpec(), DiagnosticSpec(max_logits_elements=35)
            )
        assert len(forwards) == 1
        full = backend.run_case(
            case, ProgramSpec(), [], GenerationSpec(), DiagnosticSpec(max_logits_elements=36)
        )
        assert full.diagnostics.sum_logprob == pytest.approx(selective.diagnostics.sum_logprob)
        assert [token.logprob for token in full.diagnostics.tokens] == pytest.approx(
            [token.logprob for token in selective.diagnostics.tokens]
        )


def test_generation_recognizes_secondary_model_eos(hf_spec: ModelSpec) -> None:
    with load_backend(hf_spec) as backend:
        backend.model.generation_config.eos_token_id = [2, 5]
        backend.model.generation_config.forced_eos_token_id = 5
        result = backend.run_case(
            CaseSpec(id="one", input="hello world"),
            ProgramSpec(),
            [],
            GenerationSpec(max_new_tokens=1),
        )
        assert result.output == "yes" and result.output_tokens == 1
        assert result.finish_reason == "eos" and result.truncated is False
