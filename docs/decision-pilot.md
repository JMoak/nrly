# Decision repair pilot

Turn a candidate weight edit into a reviewable decision experiment: execute the original
and patched model, fit their acceptance thresholds separately, inspect coverage and errors,
and prepare an exact-identity shadow handoff. The model's original parameters are restored
and verified after every observation.

This is an initial local application slice. It adds an offline interactive report and a
small Python service boundary; it does not create a hosted app or production gate.

## What is compared

| Arm | Weights | Acceptance rule |
| --- | --- | --- |
| Original / fixed | Original | Preregistered reference threshold |
| Original / fitted | Original | Threshold fitted on fit families |
| Patch / fixed | Edited by a bounded `ProgramSpec` | Same reference threshold |
| Patch / fitted | Same edit | Separately fitted on the same fit families |

The primary comparison is patch/fitted versus original/fitted. Threshold fitting maximizes
fit coverage under a specified **empirical** unsupported-act rate. Ties choose the higher
threshold; if no threshold meets the fit requirements, the rule always requests review.
This changes the acceptance policy, not the probabilities. Brier loss and raw agreement
therefore remain identical between a model's fixed and fitted arms.

`fit`, `evaluation`, and optional `challenge` cases have disjoint source families. Duplicate
IDs and identical rendered states are rejected. Both labels must be represented in fitting.
Inference receives only state and question; labels and split names remain with the evaluator.
The candidate, inputs, adapter identity and both fitted rules freeze before evaluation.

All runs are explicitly **exploratory**. This API does not enforce a sealed one-shot holdout,
detect paraphrased duplicates, verify family assignments or certify independently reviewed
labels. A promising result authorizes only further shadow evaluation. It never promotes a gate.

## Run it

Use the existing pinned HF/CUDA environment and cached Qwen setup from the workbench guide:

```sh
uv run --no-sync python scripts/validate_decision_qwen.py --output .nrly-decision-pilot
```

This acceptance script uses twelve public, assistant-authored source/claim examples and a
diagnostic sixteen-value patch. It also requires Bun and a sibling Ratchet checkout to test
the actual `SystemOneOracle`. It uses no paid model API. See the
[recorded result](validation/decision-pilot/README.md); the patch did not improve useful coverage.

For a custom experiment, prepare JSON with `model`, `program`, and `protocol` fields.
Inspect the model first and bind `program.expected_base` to its exact fingerprint. The
authoritative contract is available as JSON Schema:

```sh
nrly schema decision-experiment
nrly --model-cache .cache/models decision run experiment.json .nrly/my-decision-run
nrly decision report .nrly/my-decision-run .nrly/my-public-review
```

Choose a new destination for every run. Failed observations are journaled before continuing;
Ctrl-C leaves those attempts intact. Incomplete executions return a nonzero CLI exit code.
The wall-clock budget is checked between model calls and after the last call; an overrun
leaves the comparison incomplete. It cannot interrupt a stuck native call.

The report command revalidates the record, checksum, input/program bindings, and fitted
rules. Its output includes `review.html`, aggregate `assessment.json`, and, for complete runs,
`shadow.json`. Open the HTML locally and switch between fixed/fitted rules and data splits.
There are no external assets, requests or analytics.

Keep `run.json`, `private-spec.json` and `private-journal/` private: they contain evaluation
material. The HTML excludes case text, labels and family identifiers. The shadow manifest
contains the task question. Only share outputs appropriate for the recipient.

## Model adapters and weight programming

`CausalDecisionBackend` wraps nrly's existing Hugging Face teacher-forcing path. It normalizes
the likelihoods of the supplied `Yes` and `No` continuations. Its output is explicitly labeled
`causal_label_likelihood`, **not** a native decision head or calibrated correctness estimate.
It performs two forwards and records token counts, elapsed time, edited tensor evidence and
verified restoration. Times include integrity checks; they are not isolated GPU kernel latency.

`LocalBinaryDecisionBackend` supports an already loaded, caller-owned PyTorch probability
head and a trusted `predict(model, DecisionRequest) -> float` function. Its output is labeled
`native_binary_probability`. The callback's identifier remains caller-declared. Boolean
labels, numeric strings, non-finite values and out-of-range probabilities are rejected.
No callback code is loaded from a public JSON request.

The **weight programming** remains in the existing kernel: inspect parameters, author a
`WeightProgram`, and pass its declarative `ProgramSpec` into `run_decision_study`. Native
head fitting or an external research tool can produce that candidate; this pilot does not
add an autonomous editor or train a model. Existing payload limits and original-weight
restoration still apply. No dependency or existing generation-study contract changed.

## Ratchet boundary

`prepare_shadow_handoff(run)` returns an immutable typed manifest. `ShadowDecisionService`
binds the already loaded backend to that exact run, adapter, question and patch. Use
`shadow_server(service, token)` as a trusted Python context manager for an ephemeral,
authenticated `127.0.0.1` endpoint at `/v1/systemone`. It closes when the context exits.

The binary endpoint rejects a mismatched model, rubric, unsupported question type or extra
criteria. It returns **raw probabilities**; Ratchet owns the fitted acceptance rule. Applying
the rule twice would change the tested behavior. Configure `SystemOneOracle` with the
manifest's `served_model`, explicit local endpoint/token, and `pricePerMillionInput: 0`.
Zero means no API billing; hardware and energy costs are unmeasured.

The bridge supplies observations. Ratchet still owns shadow-mode execution, source and
numeric checks, draft-only policy, hard floors, audits and any later promotion. The manifest
cannot enforce those consumer policies. This pilot changes no Ratchet files or configuration.
`integration_status` remains `prepared_unverified`; the acceptance script writes a separate
conformance receipt for the exact run and the tested Ratchet source hash. That receipt proves
wire compatibility, not a full Ratchet gate or production deployment.

## Implementation boundaries

| Component | Responsibility |
| --- | --- |
| `decision_contracts.py` | Requests, family splits, provenance, frozen rules, aggregates |
| `backends/decision.py` | Trusted probability execution and existing kernel evidence |
| `decision_qualification.py` | Four-arm comparison, fitting, verification and immutable records |
| `decision_report.py` | Aggregate-only offline interface |
| `decision_handoff.py` | Exact-identity SystemOne bridge and shadow manifest |
| `decision_cli.py` | CLI over the same validated application operations |

The kernel remains independent of Ratchet and Keywork. The existing MCP/TypeScript surface
is unchanged. Later app/MCP operations should call these services rather than duplicate the
scoring or fitting logic.

## Research rationale and next experiment

The current [dspy-wsp work](https://github.com/cmpnd-ai/dspy-wsp) explicitly combines direct
editing with output-rule fitting. That motivates comparing against an original model given
the same fitting opportunity. nrly does not import that optimizer's reported gains, sandbox,
or model support as its own capabilities.

The next useful step is one **pretrained native decision-model adapter**, then an independently
prepared task corpus with matched prompt/rubric and conventional head-fitting baselines.
Those comparators are not implemented in this binary slice yet. A useful result must beat
those cheaper alternatives at a fixed error tolerance, including difficult regression families.
The first application task should come from observed failures, rather than expanding the
adapter catalog before finding a real advantage.
