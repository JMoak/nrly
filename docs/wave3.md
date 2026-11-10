# Diagnosis and comparative research workflows

nrly 0.3.0a1 extends the local workbench with teacher-forced observations, realistic
chat and sampling inputs, controlled comparisons, trusted Python authoring and
reusable edits. The implementation plan and its ownership boundaries are in
[wave3-plan.md](wave3-plan.md).

## Supported workflows

| Research question | Implemented support | Evidence boundary |
| --- | --- | --- |
| Did an edit change the probability of a particular continuation? | Token log probabilities, ranks, top alternatives and selected activation captures | Supplied text is not a generated correct answer |
| Does a change work under sampled decoding and different conversations? | Explicit chat messages, configurable decoding, up to 32,768 new tokens, paired study seeds | Reported seed variation is descriptive, not a population confidence interval |
| Is editing better than a prompt change or another checkpoint? | Registered prompt and materialized-model comparator arms with pinned identities | No training engine or automatic adapter loading |
| Can an author construct an edit from tensor calculations? | Python direction/projection/low-rank helpers and checksummed numeric tensor artifacts | Trusted local authoring compiles to bounded declarative programs |
| Does repair B undo repair A? | Immutable cumulative edit ancestry and original/A/A+B comparison | Coordinate separation does not establish behavioral independence |
| Can the result run outside the experiment store? | Verified patch packages and a scoped Python apply/restore context | Exact model identity required; no merged checkpoint export |
| Can another PyTorch network use the kernel? | Caller-owned model and execution callback through LocalTorchBackend | Trusted in-process API; no callback sandbox or subprocess timeout |

## Diagnose before editing

The following uses the already cached, pinned Qwen model and performs actual GPU
inference. It does not download weights or award a task score:

```powershell
.venv/Scripts/python.exe -m nrly --root .nrly/diagnostics --model-cache .cache/models run recipes/qwen-continuation.json
.venv/Scripts/python.exe -m nrly --root .nrly/diagnostics diagnostics RUN_ID capital
.venv/Scripts/python.exe -m nrly --root .nrly/diagnostics capture RUN_ID capital model.layers.0.mlp.down_proj
.venv/Scripts/python.exe -m nrly --root .nrly/diagnostics report RUN_ID --output .nrly/diagnostics/review.html
```

Each diagnostic scores an explicitly supplied continuation after the rendered prompt.
Prompt and continuation are tokenized separately and that policy is recorded. There
is no sampling, no generated answer, and no task accuracy for this operation. Bounds
cover continuation length, requested logit elements and serialized observation bytes.
These are not a hard bound on total model memory. See [diagnostics.md](diagnostics.md)
for position, capture and probability semantics.

## Compare methods under a registered study

Study schema 0.2 adds `seeds` and `comparators`. For example:

```json
{
  "schema_version": "0.2",
  "seeds": [17, 29],
  "generation": {"max_new_tokens": 128, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
  "comparators": [
    {"name": "format-prompt", "kind": "prompt", "prefix": "Follow the requested output format exactly. "}
  ]
}
```

This is a fragment to merge into a full StudySpec with model, development cases and
budgets. A `kind: model` comparator supplies a pinned ModelSpec for an independently
prepared checkpoint. All arms use the registered cases and decoding settings. Prompt
comparators change only their registered prefix/suffix. Each seed executes original,
repeated identity, registered controls/comparators and candidate. All executions,
including unsuccessful attempts and the reserved final evaluation, consume budget.

Cases can name a `group`, such as generalization or locality, while retaining their
target/regression role. Deterministic task adapters support exact multiple-choice
labels, structural JSON equality and literal/length instruction constraints. They
are not a general benchmark runner, JSON Schema validator or language-model judge.
See [evaluation.md](evaluation.md) for scoring version 2 and truncation policy.

## Reuse calculations and edits

[authoring.md](authoring.md) contains an executed Python example for building an edit,
storing tensor data, creating ancestry and applying a portable patch. The CLI includes:

```powershell
nrly edit create repair-a program-a.json
nrly edit create repair-b program-b.json --parent REVISION_A
nrly edit history REVISION_B
nrly edit compare ORIGINAL_RUN A_RUN AB_RUN REVISION_A REVISION_B
nrly edit select REVISION_B --ancestor REVISION_A
nrly patch export RUN_ID .nrly/portable-patch
nrly patch inspect .nrly/portable-patch
nrly patch run .nrly/portable-patch experiment.json
```

`edit select` returns the selected program; it never rewrites history or silently
changes a deployed model. `patch run` replaces the experiment's program with the
verified package and verifies the actual loaded base in the worker. A patch is a
program and provenance, not a certification of quality.

## Client and stack boundaries

The independent Python runtime remains authoritative. CLI and MCP share validated
application operations. The TypeScript SDK exposes diagnostic/capture pages,
development chat messages, complete case specifications, comparator identity and
complete assessments. Large JSON records are paginated and their reassembled SHA256
is checked by the client. Final `requireSuccess` rejects execution errors while still
accepting a valid negative scientific result.

Keywork's existing connector can run these operations without provider changes.
Vibrissa verifies the live tool contract. No existing sibling repository, user
configuration or service registration is modified. Plaia and external benchmark
framework adapters remain future integrations; the current task adapters run inside
nrly without those dependencies.

## Compatibility and remaining boundaries

New execution features require experiment schema 0.3; existing schema 0.1/0.2
specifications retain their canonical serialization. New study features require study
schema 0.2. Runtime identity still pins tokenizer, generation configuration, weights,
software and device. Replaying an old bundle is different from re-evaluating it under
a newer scorer: known token-limit truncations are unscored under scorer version 2.

Programs retain the 32-operation and 65,536-inline-value ceilings. Authoring tensors
are bounded to 16 MiB; storing a tensor does not enlarge the executable payload limit.
Workers retain full-state restoration checks. The local Python adapter can reuse a
loaded model across sequential calls; Runner still isolates each run in its own worker.

Vector batching, a resident cross-run service, quantized/tied-weight editing,
distributed execution, arbitrary-code isolation, adaptive editor synthesis and hosted
community access are outside this release. Real-model acceptance demonstrates
execution and evidence integrity; it does not establish a generally useful repair.
