# nrly wave 2: programmable weights and controlled discovery

Implemented on 2026-11-10. Python package `0.2.0a1`; local TypeScript client
`0.2.0-alpha.1`. This is an experimental research workbench with real model execution.

## Research translated into software

The November 6 [WSP-Scope paper](https://arxiv.org/abs/2611.09382v1) studies localized
edits across two Qwen3.5 scales and reports both successful trajectories and decisions to
retain the original weights. Its examples combine sparse changes, low-rank updates and
scaling across layers. The useful engineering consequence is an ordered program with
exact base identity, composable operations, and evidence of the final stored change.
nrly implements those primitives; it has not reproduced that paper's benchmark gains.

[WeightLoop](https://arxiv.org/abs/2611.07319v1), November 5, separates a fixed editor
from a predictor updated using experimental feedback. Its bounded update interface,
unchanged original reference, validation-based selection, and matched controls motivate
nrly's study lifecycle. Our current implementation evaluates submitted declarative
candidates; it does not yet synthesize a reusable adaptive editor or reproduce the
paper's experimental discovery tasks.

The November 9 revision of
[Zero-Shot Weight-Space Programming Across a Difficulty-Graded Ladder of Two-Player Games](https://arxiv.org/abs/2611.09271v2)
also motivates exact-oracle evaluation: finite computations can be checked exhaustively,
while success on local tactics does not establish multistep planning. An independent
oracle benchmark adapter is a promising next addition. The current tiny-sum fixture is
an execution demonstration, not a reproduction of those game experiments.

These are recent preprints. The architectural choices below are nrly's design decisions,
not claims that those papers validate nrly or settle the field.

## Components and responsibilities

| Component | Owns | Deliberate boundary |
| --- | --- | --- |
| Typed contracts | Programs, selections, models, cases, budgets, study records | Validation never establishes an empirical claim |
| Native tensor kernel | Inspect, stage ordered edits, apply, restore, verify | No Keywork or UI dependency |
| Worker and runner | Original model execution, process lifecycle, cumulative run limits | One model worker owns execution at a time |
| Study engine | Candidate ledger, controls, development evidence, sealed final nomination | No arbitrary Python or built-in author agent |
| Evaluation | Paired case outcomes, regression checks, conservative recommendations | No generative judge or statistical-significance claim |
| CLI / MCP / TypeScript client | Shared operations, bounded pages, polling, cancellation | Transports never implement tensor math |
| Review and context export | Offline run report and portable public study handoff | No publication, service registration, or agent config edits |

## Composed weight programs

The original scale, replace and sparse replacement operations remain available. New
operations add dense or sparse deltas, rank-factorized matrix updates and rectangular
slice scaling. A pipeline holds at most 32 operations and requires `expected_base`.
Its payload ceiling is 65,536 explicit values across operations. Obtain the fingerprint
through inspection; coordinate indices are meaningful only for that exact model.

```json
{
  "kind": "pipeline",
  "expected_base": "REPLACE_WITH_INSPECTION_FINGERPRINT",
  "operations": [
    {"kind": "weight_scale", "target": "output.weight", "factor": 2},
    {"kind": "weight_sparse_add", "target": "output.weight", "shape": [1, 2],
     "indices": [1], "values": [0.125], "factor": 1}
  ]
}
```

The example is a syntax illustration, not a proposed repair. Programs are staged in
order and applied together during each inference case. Any failed precondition leaves
the model unmodified. Execution restores original tensors, including when the body
raises. This is scoped atomicity for nrly's exclusive worker, not synchronization for
arbitrary concurrent readers of the same Python model.

Low-rank updates use `W += factor * (left @ right)`, with row-major flattened left/right
factors and explicit matrix shape/rank. Arithmetic uses at least FP32 for the new
additive operations before casting to storage dtype. A BF16 edit may round away;
recorded counts describe actual stored changes. Per-operation evidence and final net
evidence are separate. Distinct target storage is capped at 512 MiB before cloning;
this is not a bound on temporary peak memory. Aliased tensor storage is rejected.

Rich reads support flat offsets, arbitrary flat indices, explicit coordinates and
rectangular slices. Statistics describe the selected values, not the whole tensor.
Tensor hashes cover the complete tensor. MCP value pages keep coordinates aligned.
Activation captures remain a diagnostic preview in MCP: `get_sample` returns the first
16 stored activation values; use CLI export for the complete captured array.

## Controlled candidate search

1. An operator registers a `StudySpec` and a separately supplied test set using
   `nrly study create SPEC --holdout CASES`. Registration records a holdout commitment,
   allowed targets, fixed generation settings, controls and cumulative budget ceilings.
2. Authors submit `CandidateSpec` files or call `submit_study_candidate`. Each admitted
   attempt executes an original baseline, repeated identity, all registered controls,
   and the candidate with identical cases and generation settings.
3. Authors can inspect only development cases and outputs through the study MCP tools.
   They may submit more candidates within the preregistered limits.
4. Finalization freezes one candidate digest. The same arm design executes once on the
   sealed cases. Repeating the identical final request returns the same lifecycle;
   another nominee is refused. Cancellation or restart does not grant another test.

`observed_improvement` requires baseline/no-op agreement, comparable complete arms,
target improvement beyond every registered control, and preserved baseline outcomes.
Other results recommend retaining the original. Register meaningful task controls:
the system executes supplied controls equally but does not automatically establish
that their norm, footprint, or search opportunity matches the candidate.

Budget usage reports **reserved ceilings**, not measured consumption. Admission charges
runs, case executions, token ceilings, capture-byte ceilings and wall-time ceilings.
The complete final evaluation is reserved when the study is created. Failed and
cancelled attempts receive no refunds, preventing adaptive search from bypassing limits.
Per-run records retain actual runtime and token observations separately.

The study database and private run store are local files. Public APIs expose no raw
holdout cases, outputs or private run handles. An agent with filesystem or shell access
to those files can bypass that boundary. Exact duplicate inputs and case IDs across
development/holdout are rejected; semantic overlap is not detected. The included
`recipes/studies` holdout is published and therefore demonstrates workflow only.

## Stack integration and current limits

The dependency-free [TypeScript client](../packages/client/README.md) provides typed
run/study handles, paged evidence, explicit deadlines and cancellation. It adapts to
Keywork's actual MCP connection and Vibrissa's public session API. Keywork acceptance
executes real inference through its existing connector; no provider or user config was
changed. Model dependencies remain isolated in nrly's Python environment.

`nrly report RUN --baseline BASELINE --output review.html` produces an offline,
HTML-escaped report with behavior, operation/net-change evidence and provenance.
`nrly study export-context STUDY DESTINATION` produces a CorePack-AI UCP-compatible
directory layout for explicit consumption by another agent. See the
[UCP assessment](research/ucp-assessment.md) for maturity and boundaries.

Existing 0.1 experiment bundles retain their canonical hashes. New specifications
default to 0.2; richer operations require 0.2. Evidence bytes now include weight-change
records, so a zero capture budget rejects weight edits. The study format has its own
independent 0.1 schema version.

Replay checks the original model identity, outputs, stable sample evidence, weight
changes, activation captures and recorded inspections. New metadata may enrich legacy
inspection records, but cannot replace or disagree with their recorded values. Runtime
durations are excluded from equivalence. CLI JSON uses UTF-8 even when redirected on
Windows; MCP development-output offsets use Unicode code points.

The next scientific milestone is a repair that beats matched controls on independently
prepared cases. Teacher-forced log-probability diagnostics, richer scorers, autonomous
author sessions, checkpoint export, multi-model alignment, and community hosting remain
unimplemented. The current Qwen acceptance proves actual composed editing, restoration
and replay, with no demonstrated quality gain.
