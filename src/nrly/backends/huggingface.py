from __future__ import annotations

import hashlib
import inspect
import json
from typing import cast

import torch
import transformers
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from nrly.backends.torch_backend import TorchBackend, model_device, model_dtype
from nrly.contracts import (
    CaseSpec,
    ContinuationScore,
    DiagnosticSpec,
    GenerationSpec,
    ModelSpec,
    NrlyError,
)
from nrly.kernel.diagnostics import score_continuation


class HuggingFaceBackend(TorchBackend):
    def __init__(self, spec: ModelSpec) -> None:
        device = model_device(spec)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                spec.model_id,
                revision=spec.revision,
                local_files_only=spec.local_files_only,
                trust_remote_code=False,
            )
            model = cast(
                nn.Module,
                AutoModelForCausalLM.from_pretrained(
                    spec.model_id,
                    revision=spec.revision,
                    local_files_only=spec.local_files_only,
                    trust_remote_code=False,
                    use_safetensors=True,
                    dtype=model_dtype(spec),
                    attn_implementation="eager",
                ),
            ).to(device)
        except (OSError, ValueError, RuntimeError) as error:
            raise NrlyError("model_load_failed", str(error)) from error
        configuration = cast(PretrainedConfig, model.config)
        config = json.dumps(configuration.to_dict(), sort_keys=True, default=str)
        vocabulary = json.dumps(self.tokenizer.get_vocab(), sort_keys=True)
        template = self.tokenizer.chat_template
        tokenizer_backend = getattr(self.tokenizer, "backend_tokenizer", None)
        tokenizer_json = tokenizer_backend.to_str() if tokenizer_backend is not None else None
        generation_config = getattr(model, "generation_config", None)
        generation_json = (
            generation_config.to_json_string() if generation_config is not None else None
        )
        super().__init__(
            spec,
            model,
            {
                "transformers": str(transformers.__version__),
                "config_sha256": hashlib.sha256(config.encode()).hexdigest(),
                "tokenizer_vocab_sha256": hashlib.sha256(vocabulary.encode()).hexdigest(),
                "tokenizer_backend_sha256": (
                    hashlib.sha256(tokenizer_json.encode()).hexdigest() if tokenizer_json else None
                ),
                "generation_config_sha256": (
                    hashlib.sha256(generation_json.encode()).hexdigest()
                    if generation_json
                    else None
                ),
                "chat_template_sha256": hashlib.sha256(
                    json.dumps(template, sort_keys=True).encode()
                ).hexdigest(),
                "attention_implementation": "eager",
            },
        )

    def _encode_prompt(self, case: CaseSpec, generation: GenerationSpec) -> dict[str, torch.Tensor]:
        text = case.input
        if case.messages and not self.tokenizer.chat_template:
            raise NrlyError("unsupported_operation", "explicit messages require a chat template")
        if self.tokenizer.chat_template:
            messages = (
                [{"role": message.role, "content": message.content} for message in case.messages]
                if case.messages
                else [{"role": "user", "content": case.input or ""}]
            )
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=generation.enable_thinking,
            )
            if not isinstance(rendered, str):
                raise NrlyError("tokenization_failed", "chat template did not return text")
            text = rendered
        device = next(self.model.parameters()).device
        inputs = dict(
            self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        )
        if inputs["input_ids"].shape[-1] == 0:
            raise NrlyError("tokenization_failed", "prompt must contain at least one token")
        return inputs

    def _check_context(self, input_tokens: int, extra_tokens: int) -> None:
        context = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(context, int) and input_tokens + extra_tokens > context:
            raise NrlyError("context_exceeded", "input plus requested tokens exceeds model context")

    def _execute(self, case: CaseSpec, generation: GenerationSpec) -> tuple[str, int, int, str]:
        inputs = self._encode_prompt(case, generation)
        device = inputs["input_ids"].device
        input_tokens = inputs["input_ids"].shape[-1]
        self._check_context(input_tokens, generation.max_new_tokens)
        options: dict[str, object] = {
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": generation.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id,
        }
        if generation.temperature > 0:
            options["temperature"] = generation.temperature
        for key in ("top_p", "top_k", "min_new_tokens", "repetition_penalty"):
            value = getattr(generation, key)
            if value is not None:
                options[key] = value
        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(generation.seed)
            generate = getattr(self.model, "generate", None)
            if not callable(generate):
                raise NrlyError("unsupported_operation", "loaded model does not support generation")
            result = generate(**inputs, **options)
        generated = result[0, input_tokens:]
        output = self.tokenizer.decode(generated, skip_special_tokens=True)
        if not isinstance(output, str):
            raise NrlyError("tokenization_failed", "tokenizer did not return decoded text")
        configured_eos = getattr(self.model.generation_config, "eos_token_id", None)
        eos = configured_eos if configured_eos is not None else self.tokenizer.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        finish_reason = (
            "eos"
            if len(generated) and generated[-1].item() in eos_ids
            else "length"
            if len(generated) >= generation.max_new_tokens
            else "stopped"
        )
        return output, int(input_tokens), len(generated), finish_reason

    def _diagnose(
        self, case: CaseSpec, generation: GenerationSpec, diagnostics: DiagnosticSpec
    ) -> ContinuationScore:
        if case.continuation is None:
            raise NrlyError("invalid_request", "teacher forcing requires a continuation")
        inputs = self._encode_prompt(case, generation)
        prompt = inputs["input_ids"]
        continuation = self.tokenizer(
            case.continuation, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(prompt.device)
        count = continuation.shape[-1]
        if count == 0:
            raise NrlyError("tokenization_failed", "continuation must contain at least one token")
        if count > diagnostics.max_tokens:
            raise NrlyError("budget_exceeded", "continuation exceeds the diagnostic token budget")
        self._check_context(prompt.shape[-1], count)
        combined = torch.cat((prompt, continuation), dim=-1)
        selective = "logits_to_keep" in inspect.signature(self.model.forward).parameters
        rows = count if selective else combined.shape[-1]
        vocabulary = getattr(self.model.config, "vocab_size", None)
        if not isinstance(vocabulary, int) or vocabulary <= 0:
            raise NrlyError("unsupported_operation", "model must declare its vocabulary size")
        if rows * vocabulary > diagnostics.max_logits_elements:
            raise NrlyError("budget_exceeded", "requested diagnostic logits exceed element budget")
        forward_options: dict[str, object] = {}
        logits_start = prompt.shape[-1] - 1 if selective else 0
        if selective:
            forward_options["logits_to_keep"] = torch.arange(
                logits_start, logits_start + count, device=prompt.device
            )
        result = self.model(
            input_ids=combined,
            attention_mask=torch.ones_like(combined),
            use_cache=False,
            return_dict=True,
            **forward_options,
        )
        logits = getattr(result, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise NrlyError("invalid_backend_result", "causal model did not return logits")

        def decode(token_id: int) -> str:
            value = self.tokenizer.decode([token_id], skip_special_tokens=False)
            if not isinstance(value, str):
                raise NrlyError("tokenization_failed", "tokenizer did not return decoded text")
            return value

        if logits.numel() > diagnostics.max_logits_elements:
            raise NrlyError("invalid_backend_result", "model exceeded requested diagnostic logits")
        return score_continuation(
            logits,
            continuation[0],
            prompt.shape[-1],
            diagnostics,
            decode,
            logits_start=logits_start,
        )
