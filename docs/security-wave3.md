# Wave 3 security and release delta review

Reviewed 10 November 2026 against the developing `0.3.0a1` source. This review covers
the new authoring/artifact APIs, local Torch adapter, generation/diagnostics, study
comparators, and public evidence pages. The earlier
[0.2.0a2 review](security-review-2026-11-10.md) does not certify these changes.
The supported environment remains trusted, single-user local research software.

## Findings addressed

| Finding | Change and verification |
| --- | --- |
| Raw tensor `.bin` files could enter a custom publication directory; an all-zero float tensor is valid UTF-8 containing NULs and passed the former text check | Ignore and source-package-exclude `*.bin`; the publication guard rejects `.bin` paths and NUL-containing text. Tests cover ignored and force-tracked blobs, untracked blobs, and a misleading `.txt` suffix. |
| JSON structural scoring rounded distinct decimal numbers through Python floats, potentially awarding a false pass | Parse integers and decimals exactly with `Decimal`; limit numeric literals to 4,096 characters and exponents to magnitude 10,000. Tests include `0.10000000000000001` versus `0.1`, equivalent numeric spellings, and rejection bounds. |
| Generation ending on a second EOS ID configured by the model could be labeled truncated | Classify completion using the effective model generation configuration. A native CPU causal-model test forces the secondary EOS and verifies completed, untruncated evidence. |

## Reviewed boundaries

**Artifacts and executable programs.** Tensor data uses raw little-endian float32/64
bytes and strict JSON, with no pickle or code loading. Handles bind shape, dtype,
byte length, and SHA256. Reads reject missing, oversized, escaped, or linked members;
finite values and exact byte counts are checked before use. Tensor artifacts are
limited to 16 MiB, patch/edit manifests to 4 MiB, and revision ancestry to 32 steps.
Compiled programs retain the existing 32-operation/65,536-value bounds. Artifacts
cannot bypass the declarative program validator. Checksums provide integrity, not
publisher authentication, provenance of training data, or evidence of quality.

Edit loads verify every parent and the exact cumulative operation prefix. Deployment
compares the recorded model identity and independently hashes loaded tensor bytes,
then uses the scoped kernel apply/restore path. Tests cover tampering, a rehashed
child that changes its parent's operations, wrong identities and weights, existing
export destinations, harmful sequential edits, and restoration after exceptions.
CLI file paths are explicit operator inputs; MCP exposes no authoring-code or patch
filesystem-execution endpoint. Caller-owned directories remain outside an OS sandbox;
checks are not protection against an adversary concurrently replacing local files.

**Trusted Python execution.** `LocalTorchBackend` accepts an already constructed
model and a caller-supplied callable. This is deliberate arbitrary Python execution
in the caller's process, not an untrusted plugin loader. `ModelSpec` rejects
`local_torch`, so JSON experiments and MCP cannot select a callback or import path.
Adapter identifiers are explicitly caller-declared, not verified code hashes. Tests
verify this boundary, scoped weight restoration on callback failure, and RNG
restoration. A trusted callback can still access the host or mutate unrelated state;
there is no timeout/sandbox promise for this in-process API.

**Generation and diagnostic resources.** Schema 0.3 permits up to 32,768 generated
tokens and 32,768 continuation tokens. Context checks, worker wall time, aggregate
output limits, capture-byte accounting, and the 16 MiB worker-result limit still
apply. Diagnostic logits are preflighted against a default 67,108,864-element budget,
with a configurable ceiling of 268,435,456. Native selective logits are requested
when supported; otherwise the full prompt-plus-continuation logits are budgeted.
Tests verify rejection before forward execution on both paths and equal scores
between paths. These are element limits, not a peak RAM/GPU quota: model weights,
attention, activations, decoded strings, temporary buffers, and result construction
also consume memory. Capture/result byte checks occur after execution. Long requests
can still exhaust resources and must not be offered as an untrusted public service.

**Studies and public evidence.** Prompt comparators apply only registered wrappers;
model comparators use exact pinned `ModelSpec` values through the safetensors-only,
`trust_remote_code=False` loader. Model identities remain stable across repeated
seeds and attempts, including duplicate registrations of the same model. Ordinary
run comparison still refuses different model identities. Tests cover seed/case/model
mismatches, cancelled or unscored arms, immutable nomination, cumulative reservations,
and overlap between effective development and holdout prompts.

Public study methods expose development text and aggregate final assessments; private
run IDs and sealed case text stay in the private store. CPU tests exercise actual
external-model comparator workers and the sealed final evaluation. General filesystem
access can read the holdout store; this is an API boundary, not cryptographic secrecy.
Group names and other operator-provided metadata can themselves be confidential.

Diagnostic and activation pages read stored evidence only, with bounded offsets,
counts, and response size. Token previews preserve a truncation indicator and original
character count. Teacher-forced diagnostics are explicitly distinguished from generated
task outcomes, and incomplete/truncated inference remains unscored. Study assessment
pages contain aggregate evidence and a digest for reconstructing the full JSON.
The stdio MCP server still has no authentication, tenancy, or global durable-storage
quota; no network service was registered or opened by this review.

## Verification and release scope

The focused publication/artifact/legacy/evidence suite passed **86 tests**. All changed
publication-guard code and tests passed Ruff. The two earlier real-worker CLI tests
passed export/inspect/re-execution, exact evidence checks, harmful A+B detection,
and machine-readable refusals without overwriting package files.
The runtime/task-adapter/seeded-study/model-comparator/evidence suite separately
passed **93 tests** on CPU, including native local Transformers workers. These two
focused suites overlap in evidence tests and are not an additive total of unique tests.

All **six** existing public run bundles loaded with verified manifests and canonical
specification digests. All **13** files listed by the wave-2 Qwen artifact inventory
retained their exact byte counts and SHA256 values. No legacy bundle was rewritten.
This is artifact verification, not a new real-model inference claim.

The publication guard inspected **220** publishable files at the review snapshot and found no
private-path, profile-path, binary, or size findings. The managed shell rejected Git's
`-C` invocation for the root repository; the same guard inspection passed using Git's
equivalent inherited-working-directory enumeration. Guard regression tests used
temporary repositories and passed normally. This is a scoped source-set check, not
a secret scan or a guarantee about later edits.

No dependencies were added or advisory/secret scanners installed for this delta
review. Earlier advisory and secret-scan results have not been rerun or extended to
new source. Final wheel/source-archive inspection is deferred until the integrating
agent freezes the source; the new `.bin` source-package exclusion is configured but
is not yet claimed as an inspected release-artifact result. No commit, publication,
configuration registration, checkpoint export, or GPU execution was performed by
this review.
