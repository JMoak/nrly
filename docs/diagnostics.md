# Runtime diagnostics

Teacher forcing evaluates supplied continuation tokens under a native causal model.
It records probabilities and selected activations without treating the supplied text
as a generated answer. Use experiment schema `0.3` for these features.

## A local transformer, without downloads

The CPU tests in [test_wave3_runtime.py](../tests/test_wave3_runtime.py) construct a
small GPT-2 model and WordLevel tokenizer locally. They compare nrly's token
log probabilities with the model's own causal logits, verify input captures,
apply a real edit, and verify exact restoration. Run them with:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_wave3_runtime.py --basetemp .test-tmp/diagnostics-check -q
```

For an already saved local Hugging Face causal checkpoint:

```python
from nrly.backends import load_backend
from nrly.contracts import (
    CaptureSpec, CaseSpec, DiagnosticSpec, GenerationSpec, ModelSpec, ProgramSpec,
)

# The test fixture uses a synthetic revision label and a genuinely hashed local
# checkpoint. Remote models instead require their exact immutable commit digest.
model = ModelSpec(
    backend="huggingface", model_id="path/to/local-gpt2",
    revision="0" * 40, local_files_only=True,
)
with load_backend(model) as backend:
    result = backend.run_case(
        CaseSpec(id="tokens", input="hello world", continuation="yes no"),
        ProgramSpec(),
        [CaptureSpec(
            target="transformer.h.0.mlp.c_proj", source="input",
            invocation="first", token_positions=[0, 2], max_elements=64,
        )],
        GenerationSpec(),
        DiagnosticSpec(top_k=3, max_tokens=16),
    )
    print(result.diagnostics.model_dump())
    backend.verify_base_unchanged()
```

With the test's tokenizer, the prompt IDs are `[3, 4]`, continuation IDs are
`[5, 6]`, and observations refer to absolute token positions `2` and `3`.
Each probability comes from the preceding causal position, rather than from
the position that already contains its answer. These are execution fixtures;
their scores do not establish pretrained model quality.

## A pinned Qwen experiment

[qwen-model.json](../recipes/qwen-model.json) identifies the cached Qwen3-0.6B
checkpoint used by the real-model acceptance scripts. This example creates a
teacher-forced experiment using that exact model specification:

```python
from pathlib import Path
from nrly.contracts import (
    BudgetSpec, CaptureSpec, CaseSpec, ChatMessage, DiagnosticSpec,
    ExperimentSpec, ModelSpec,
)

spec = ExperimentSpec(
    schema_version="0.3",
    name="Qwen supplied-continuation diagnostic",
    model=ModelSpec.model_validate_json(Path("recipes/qwen-model.json").read_bytes()),
    cases=[CaseSpec(
        id="arithmetic",
        messages=[ChatMessage(role="user", content="Reply with only the integer: 2 + 3 =")],
        continuation="5",
    )],
    diagnostics=DiagnosticSpec(top_k=5, max_tokens=32),
    captures=[CaptureSpec(
        target="model.layers.0.mlp.down_proj", source="input",
        invocation="first", token_positions=[0], max_elements=16,
    )],
    budget=BudgetSpec(max_cases=1, max_capture_bytes=16384),
)
Path(".cache").mkdir(exist_ok=True)
Path(".cache/qwen-diagnostics.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
```

From the repository root, with the checkpoint already present:

```powershell
.venv/Scripts/python.exe -m nrly --root .nrly-diagnostics --model-cache .cache/models run .cache/qwen-diagnostics.json
.venv/Scripts/python.exe -m nrly --root .nrly-diagnostics diagnostics RUN_ID arithmetic
.venv/Scripts/python.exe -m nrly --root .nrly-diagnostics capture RUN_ID arithmetic model.layers.0.mlp.down_proj
```

The first command returns the run ID. The latter commands read stored evidence
without running the model. MCP exposes `get_diagnostics` and `get_capture`; the
TypeScript run handle exposes `diagnostics`, `diagnosticPages`, and `capture`.
Use `next_offset` and `next_top_k_offset` to retrieve remaining stored values.
Token text previews are capped at 256 characters; token IDs remain exact. Full
run bundles retain complete recorded text and probabilities.

The reproducible GPU smoke procedure is
[validate_wave3_qwen.py](../scripts/validate_wave3_qwen.py). It retains an explicit
edit, before/after probabilities, selected activations, restoration evidence,
fresh-worker replay, and a small sampled study. Its public smoke prompts are
integration checks, not a capability benchmark or a sealed research evaluation.

## Interpretation and bounds

- Prompt and continuation are encoded **separately**, then their token IDs are
  concatenated. This differs from tokenizing a joined string when the tokenizer
  merges across the text boundary. The result records this policy explicitly.
- Positions are zero-based absolute indices in that concatenated sequence.
  `rank` is one plus the count of strictly larger logits; tied tokens share rank.
  Top-k entries may have an implementation-dependent order when logits tie.
- Probabilities use FP32 log-softmax of native logits. Sampling temperature,
  top-p, top-k, and repetition penalties do not transform diagnostic scores.
  `sum_logprob` and `mean_logprob` summarize the supplied continuation only.
- Teacher-forced samples have `output=""`, `output_tokens=0`, and
  `finish_reason="teacher_forced"`. Evaluation marks them unscored. Their JSON
  evidence counts against `max_capture_bytes`, not generated-token usage.
- `max_tokens` limits the continuation; context limits still apply to the full
  prompt and continuation. `max_logits_elements` defaults to 67,108,864 and is
  capped at 268,435,456. Models with an explicit native `logits_to_keep` argument
  compute only needed vocabulary rows. Other models preflight the whole sequence.
  These are output-allocation bounds, not total GPU-memory guarantees.
- Activation captures require exact module names. `source="input"` reads the
  first positional tensor at the module's pre-hook; `source="output"` reads its
  tensor output after any output intervention. Tuple/mapping sources are rejected.
- `token_positions` requires a `[1, sequence, hidden]` activation. Selected
  positions retain their requested order; values are flattened and bounded by
  `max_elements`. `invocation="first"` selects the first forward call;
  `"latest"` selects the latest. Cached generation calls often contain only one
  sequence position, so absolute prompt positions normally require `"first"`.
  Teacher forcing observes the full sequence even when logits are selectively kept.
- Out-of-range positions and uninvoked requested modules fail explicitly. Hooks
  are removed and base weights checked on success and failure.

## Chat, sampling, and Python adapters

A case provides exactly one plain `input` or an explicit `messages` history.
Messages support system, user, and assistant roles and require the model's chat
template. Plain input retains the existing single-user-turn template behavior.

Generation supports up to 32,768 new tokens, subject to context and experiment
budgets. `top_p`, `top_k`, `min_new_tokens`, and `repetition_penalty` are optional:
omission retains the pinned model's generation configuration. Explicit values
require schema `0.3`. Each case resets the registered Torch seed within a scoped
RNG context; repeatability remains specific to the recorded runtime and device.
EOS detection uses the model's effective configured EOS IDs, including lists.
Length-limited output is marked truncated and cannot establish a task pass.

For trusted non-LM execution, import `LocalTorchBackend` and `LocalExecution`
from `nrly.backends.local`. Supply an existing `torch.nn.Module` and a callable
`(model, case, generation) -> LocalExecution`, plus model ID, revision, and adapter
ID. The adapter ID and revision are caller-declared; model state is independently
hashed. The callback executes in-process without a sandbox or worker timeout.
It is never loaded from a filename, serialized experiment, or MCP request.

Keep a backend open to reuse resident weights across `run_case` calls. Each call
still installs scoped hooks and verifies the full model-state digest before and
after execution. There is no vectorized batching or reduced hashing mode.
