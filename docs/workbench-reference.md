# Workbench reference

[Back to nrly](../README.md) · [Current workflow guide](wave3.md)

Detailed commands, execution semantics, study boundaries, and evidence formats.

## Inspect, edit, compare

```powershell
uv run --no-sync nrly inspect --target output.weight
uv run --no-sync nrly run recipes/tiny-baseline.json
uv run --no-sync nrly run recipes/tiny-weight-repair.json
uv run --no-sync nrly list
uv run --no-sync nrly compare BASELINE_RUN_ID CANDIDATE_RUN_ID
uv run --no-sync nrly export CANDIDATE_RUN_ID .nrly/my-bundle
uv run --no-sync nrly replay .nrly/my-bundle
uv run --no-sync nrly report CANDIDATE_RUN_ID --baseline BASELINE_RUN_ID --output .nrly/review.html
```

Inspection loads the model, catalogs its actual tensor names/shapes, and reads bounded values by flat offset, indices, coordinates, or rectangular slices. Reads include coordinates, selected-value statistics, and a full tensor hash. Use `--read-spec selections.json` for rich selections; `nrly schema weight-read` describes their format. Inspection executes no forward passes and cannot receive a scored improvement verdict.

A sparse program can address selected entries in a large tensor:

```json
{
  "kind": "weight_patch",
  "target": "output.weight",
  "shape": [1, 2],
  "indices": [0, 1],
  "values": [1.0, 1.0]
}
```

Indices use flattened logical row-major order. Shapes must match exactly; duplicate, boolean, fractional and out-of-range indices are rejected. For real experiments, set `expected_base` to the identity fingerprint returned by inspection. Replay sets this automatically.

| Operation | Meaning |
| --- | --- |
| `identity` | Unmodified model control |
| `weight_scale` | Multiply one exact parameter tensor by a finite factor |
| `weight_replace` | Replace an exact tensor with a bounded explicit payload |
| `weight_patch` | Set selected entries of an exact tensor |
| `weight_add`, `weight_sparse_add` | Add a scaled dense or indexed delta |
| `weight_low_rank` | Add `factor * (left @ right)` to a matrix without a full dense payload |
| `weight_slice_scale` | Scale a rectangular tensor slice |
| `pipeline` | Execute up to 32 ordered weight operations against a mandatory exact base fingerprint |
| `activation_scale`, `activation_add` | Temporary diagnostics during execution; never reported as weight changes |

A weight program edits parameters for the entire inference case. The change persists as a replayable declarative program; the in-memory model is restored after each case. Source checkpoints are never overwritten. `nrly` does not yet export merged model checkpoints or run arbitrary Python programs.

Each weight edit records tensor hashes before/after, the number of changed elements, shape/dtype, first-32-element previews and selected edited values. Pipelines also record final net changes once per target, so cancelling edits are not counted as persistent changes. Operations are staged and restored together. Tied storage is rejected, and distinct edited tensor storage is capped at 512 MiB; temporary buffers can require additional memory. Restoration checks hash the full model state. This costs time proportional to model size.

Serialized weight-change evidence counts toward `max_capture_bytes`, as do activation captures, diagnostics and inspections. A zero-byte capture budget cannot execute an evidenced weight edit. Extended execution features require experiment schema 0.3; legacy 0.1/0.2 bundles remain readable and verifiable.

## Real model

The validated model is `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`. Its model specification and smoke recipes live under `recipes/qwen-*`. They use local files by default: a missing model fails explicitly. To download the pinned public model, set `HF_HOME` to the cache root you intend to pass to nrly and use the official Hub client:

```powershell
$env:HF_HOME = "$PWD/.cache/models"
uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])"
uv run --no-sync nrly --model-cache .cache/models inspect --model-spec recipes/qwen-model.json --target model.layers.0.mlp.down_proj.weight
uv run --no-sync nrly --model-cache .cache/models run recipes/qwen-baseline.json
uv run --no-sync nrly --model-cache .cache/models run recipes/qwen-sparse-patch.json
```

The validated GPU configuration is RTX 5090, BF16, eager attention, PyTorch 2.15.0+cu132 and Transformers 5.22.2. The tiny Qwen experiment changes one parameter and checks identity, captures, restoration and repeatability. It demonstrates actual weight editing, **not a measured capability improvement**. It is a different model from the Qwen3.5-0.8B used in the originating WSP report. CPU execution requires changing the model device/dtype and produces a different runtime fingerprint.

## Agent interface

Start the server with the installed environment, using absolute paths when connecting from another application:

```powershell
C:/src/nrly/.venv/Scripts/python.exe -m nrly.mcp --root C:/src/nrly/.nrly/agent --model-cache C:/src/nrly/.cache/models
```

Run tools: `capabilities`, `submit_inspection`, `submit_experiment`, `get_run`, `get_inspection`, `get_samples`, `get_sample`, `list_runs`, `cancel_run`, and `compare_runs`. Study tools add discovery, candidate submission, development evidence, cancellation, and finalization; see [the client guide](../packages/client/README.md).

Submit returns immediately with a run handle. Poll `get_run` until terminal; then request paginated inspection or sample details. Input schemas expose the complete typed experiment contract. Identical `request_id` retries reuse their run; conflicting specifications are rejected. Large results return an explicit summary with an artifact handle, or an actionable pagination error. MCP responses are bounded to 12,000 JSON characters. The `nrly://runs/…` handle identifies a local run for CLI export; it is not an HTTP download URL or an MCP resource.

The MCP process owns one worker at a time and an exclusive workspace lease. A second runner on the same store is refused. Read-only CLI commands can inspect that store. Closing the server cancels active work; an unexpected restart marks unfinished runs interrupted. The child watches parent liveness. This is process isolation for supported declarative operations, not an unrestricted-code security sandbox.

Keywork consumes this stdio server through its existing MCP connector. The new [TypeScript client](../packages/client/README.md) supplies typed handles, bounded iterators, deadlines, and transport adapters for Keywork, Vibrissa, and other MCP clients. Acceptance tests call Keywork's actual connector and configuration schema. See the [integration constraints](repo-integrations.md); the user's actual configuration was not changed.

## Controlled discovery studies

An operator registers development cases, controls, editable targets, cumulative budgets, and a separate holdout. Candidate authors can inspect development evidence and submit revisions. Finalization freezes one candidate and evaluates it once against the holdout. Every attempt remains recorded; failed and cancelled attempts retain their budget charges. Final execution is reserved before discovery starts.

```powershell
uv run --no-sync nrly study create recipes/studies/tiny-study.json --holdout recipes/studies/tiny-holdout.operator.json
uv run --no-sync nrly study submit STUDY_ID recipes/studies/tiny-candidate-repair.json
uv run --no-sync nrly study attempts STUDY_ID
uv run --no-sync nrly study finalize STUDY_ID ATTEMPT_ID
uv run --no-sync nrly study show STUDY_ID
uv run --no-sync nrly study export-context STUDY_ID .nrly/study-handoff
```

The included holdout is a **published engineering fixture**, not unseen scientific evidence. For a real study, provision independently prepared cases. Study MCP tools expose aggregate final assessments, never holdout inputs, answers, outputs, or private run handles. This is an application boundary; a process with filesystem access can read the database. The optional context export follows CorePack-AI's UCP `.ai` layout and contains public study information only; consumers load it explicitly.

## Evidence and architecture

```text
CLI / MCP
    ↓
Application + Runner → SQLite metadata + immutable JSON bundles
    ↳ StudyEngine → private arm store + committed holdout + attempt ledger
    ↓ isolated worker
Native PyTorch / Hugging Face backend
    ↓
Kernel: inspect → apply → execute → restore → verify
    ↓
Independent scorers and paired comparisons
```

`contracts.py` defines versioned inputs/results. `kernel/` owns parameter and activation semantics. `backends/` loads original model implementations with remote code disabled. `runner.py` owns process lifecycle, budgets and cancellation; `store.py` owns persistence and bundle verification. `studies.py` owns candidate admission, controls, nomination and final evaluation. Public and private workers share an execution gate. `evaluation/` never executes the candidate. `api.py`, `cli.py` and `mcp.py` expose the same execution and study operations.

Ordinary comparisons require identical model/runtime identities, cases, scorers and generation settings. Studies can explicitly register prompt and other-model comparators. Missing, failed, truncated or unscored samples cannot become passes. Targets, regressions and named groups have separate summaries. Exact string, numeric, final-number and deterministic task-adapter scoring are intentionally narrow; they are not general language-model judges. Ordinary run split labels do not enforce a sealed holdout; use the study workflow for that application boundary. Numerical identity checks reject changed environments; replay does not promise determinism across arbitrary devices or library versions.

Bundles contain `spec.json`, `run.json` and a SHA256 manifest. Checksums detect corruption and inconsistent members; they are not signatures or proof that a publisher's claims are true. Replay performs fresh execution and reports agreement. Model weights are referenced by revision/fingerprint, not bundled.
