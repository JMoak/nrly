# Qwen CLI and runner acceptance — 2026-11-10

The actual CLI and isolated worker completed a pinned Qwen3 0.6B experiment: inspect weights, run a baseline, apply an identity weight control, edit one scalar, compare, export, replay, and inspect the unchanged base. This establishes a functioning real-model experiment path. The scalar edit produced **no observed behavioral improvement** on the three smoke prompts.

This is integration evidence from public smoke prompts, not a held-out capability benchmark. The direct backend smoke and this runner validation used the same prompts. They do not establish generalization, parameter semantics, or superiority to prompting or fine-tuning.

## Tested configuration

| Property | Value |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B` |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Hardware | NVIDIA GeForce RTX 5090 |
| Execution | Windows, Python 3.13.7, Torch 2.15.0+cu132, CUDA 13.2, Transformers 5.22.2 |
| Precision / attention | BF16 / eager |
| Generation | Greedy, seed 0, maximum 16 new tokens, thinking disabled |
| Model fingerprint | `f73bb3b459f878de0daa06254495c90dca296e5154d9b219eb6f1883a3af9b93` |
| Base state SHA256 | `c21f9db51aa222b09c9590c3c891e9b50eecb0b5b99368ae7f989bde6a5115d6` |

The fingerprint also records tokenizer, chat template, generation configuration, library versions, and device details. Exported replay requires the exact recorded fingerprint; this run does not certify other devices or precisions.

## Observed task results

The scorer checks exact text after trimming leading and trailing whitespace. Formatting failures remain failures.

| Case | Expected | Baseline, identity control, and patch output | Outcome |
| --- | --- | --- | --- |
| Integer-only arithmetic | `5` | `2 + 3 = 5` | Fail |
| Exact format | `OK` | `OK.` | Fail |
| Capital regression | `Paris` | `Paris` | Pass |

Every run scored **1/3**, with 0/2 target cases and 1/1 regression cases passing. Both comparisons were complete and comparable, with verdict `no_change`. These three passing execution runs include two honest task failures each.

| Run | Handle | Wall time | Generated tokens |
| --- | --- | --- | --- |
| Baseline | `e2debe3545ff4255b99aa954b060938e` | 12.849 s | 13 |
| Identity weight control | `d95c1727b7544d3e96a6fec012d7e9ee` | 13.420 s | 13 |
| Sparse weight patch | `ddb3cdd99fa248a298ce8fb93931191c` | 13.120 s | 13 |

Wall time includes worker startup, model load, execution, integrity hashing, and cleanup. These measurements are not throughput benchmarks.

## Parameter-level evidence

The program addressed `model.layers.0.mlp.down_proj.weight`, shape `[1024, 3072]`, using logical row-major flattened index `0`. The bounded inspection read `0.02099609375`. The sparse patch explicitly assigned `0.0`.

Each candidate sample recorded exactly **one changed scalar out of 3,145,728 elements in that tensor**. The identity control recorded zero changed elements. The changed tensor hash moved from `01cc5c354c8ac9cf4917157d74beceb0b3ce3f38d9bdec271ae691bded0b2e3b` to `83fe09c69fc9554fcd03bbd052112cfc0fe7cc8004453327844008d729ad1c72`.

The backend verified restoration against the complete base state after every case. A later independent inspection returned the original values and identical model fingerprint. The base model cache was preserved. The executable patch persists in the recipe and exported bundle; the original model weights are restored after each experiment.

## Export and replay

All three runs were exported as checksum-verified bundles. Replaying the sparse patch through `nrly replay` succeeded with matching model identity and exact outputs. The replay binds the program to the original fingerprint before model inspection or inference.

Evidence:

- [Machine-readable summary](qwen-runner/summary.json)
- [Identity comparison](qwen-runner/compare-identity.json)
- [Sparse patch comparison](qwen-runner/compare-patch.json)
- [Replay result](qwen-runner/replay.json)
- [Baseline bundle](qwen-runner/bundles/baseline/manifest.json)
- [Identity control bundle](qwen-runner/bundles/identity-weight/manifest.json)
- [Sparse patch bundle](qwen-runner/bundles/sparse-patch/manifest.json)

The bundles contain specifications and measured outputs, not model weight files. The revision must already be available in the local Hugging Face cache for these recipes, which set `local_files_only: true`.

Representative commands from the repository root, using the tested Windows CUDA dependency extra:

```powershell
uv run --extra cuda nrly --root .nrly-qwen --model-cache .cache/models inspect --model-spec recipes/qwen-model.json --target model.layers.0.mlp.down_proj.weight --max-elements 2
uv run --extra cuda nrly --root .nrly-qwen --model-cache .cache/models run recipes/qwen-baseline.json
uv run --extra cuda nrly --root .nrly-qwen --model-cache .cache/models run recipes/qwen-identity-weight.json
uv run --extra cuda nrly --root .nrly-qwen --model-cache .cache/models run recipes/qwen-sparse-patch.json
uv run --extra cuda nrly --root .nrly-qwen --model-cache .cache/models replay docs/validation/qwen-runner/bundles/sparse-patch
```

## Runtime issue found and fixed

Integrated execution exposed a Windows-specific startup deadlock that direct backend tests did not catch: a blocking stdin watchdog thread prevented NumPy's native extension from loading. A paired subprocess probe and Python fault-handler trace isolated the issue. The Windows watchdog now polls the inherited pipe with nonblocking `PeekNamedPipe`; POSIX uses an unbuffered pipe read. The full runner tests then passed, including actual model subprocess execution and parent-pipe closure termination.

The result is a usable foundation for controlled weight-space investigations. Discovering a useful behavioral edit remains separate scientific work.
