# Wave 2 Qwen GPU acceptance — 10 November 2026

nrly 0.2.0a1 completed a real Qwen3-0.6B GPU acceptance covering arbitrary weight reads, an ordered three-operation program across two layers, exact restoration, a cancelling control, isolated Runner execution, portable export, and fresh-worker CLI replay. All assertions passed. No production fixes were needed during this acceptance.

The edit changed three stored BF16 scalars. It produced **no observed output change or capability improvement** on three public smoke prompts. The two exact-format failures remain failures; this is integration evidence, not a held-out repair study.

## Runtime and identity

| Property | Measured value |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B` |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Device | NVIDIA GeForce RTX 5090 |
| Runtime | Windows; Python 3.13.7; Torch 2.15.0+cu132; CUDA 13.2; Transformers 5.22.2 |
| Dtype / attention | BF16 / eager |
| Generation | Greedy, seed 0, maximum 16 new tokens, thinking disabled |
| Model fingerprint | `f73bb3b459f878de0daa06254495c90dca296e5154d9b219eb6f1883a3af9b93` |
| Base state SHA256 | `c21f9db51aa222b09c9590c3c891e9b50eecb0b5b99368ae7f989bde6a5115d6` |

The fingerprint and base hash exactly match the earlier MVP Qwen acceptance. The model revision was already cached; `local_files_only` remained enabled. No model weights were downloaded, modified on disk, or exported.

## Inspection beyond the prefix

The first inspected parameter was `model.layers.0.mlp.down_proj.weight`, shape `[1024, 3072]`.

- An offset read starting at flat index `6160` returned four exact values and coordinates.
- An indexed read requested `[6162, 6161, 6160]` and preserved that order.
- A matrix slice selected rows `[2, 4)` and columns `[16, 20)`, returning eight values with selection shape `[2, 4]`. Its flat indices were `[6160, 6161, 6162, 6163, 9232, 9233, 9234, 9235]`.
- A coordinate read on `model.layers.1.mlp.down_proj.weight` selected `[[3, 21], [3, 22]]`.

Offset, indexed, and slice values agreed. Every selected value lay beyond the original first-4096 prefix. Read results included logical row-major coordinates, population statistics over the selected values, and the complete tensor hash. Full model hashing confirmed that inspection did not alter weights.

## Ordered program and stored-weight evidence

The program required the exact model fingerprint before execution:

1. A rank-one additive update to layer 0's down projection used a left factor with only row 2 nonzero and a right factor with only column 17 nonzero. It added `1/1024` at `[2, 17]`.
2. A rectangular slice scale multiplied layer 0's row 2, columns 17 and 18, by `1.0078125`.
3. A sparse additive update added `1/2048` at layer 1's `[3, 21]`, flat index `9237`.

New arithmetic used FP32 accumulation and then rounded into stored BF16. Evidence measured actual stored bytes.

| Parameter coordinate | Before the program | After the complete program |
| --- | ---: | ---: |
| Layer 0, `[2, 17]` | 0.022705078125 | 0.02392578125 |
| Layer 0, `[2, 18]` | -0.03515625 | -0.035400390625 |
| Layer 1, `[3, 21]` | 0.0030059814453125 | 0.0034942626953125 |

The three operation counts were `1`, `2`, and `1`. The first two operations share a coordinate, so the final net count was **3**, not 4. Per-operation hashes chained correctly; final per-target net hashes and counts were attached to each target's last operation.

The complete edited model state hash was `6a6d607b56418da69bcd33fa9ae3822f059abe0c020993a47e13e30eb861201d`. After leaving the intervention and after every inference case, the full model state matched the original base hash. Subsequent selected reads matched the original values and tensor hashes exactly.

A separate two-operation control scaled the same layer-0 slice by 2 and then by 0.5. Each intermediate operation reported two changed elements, while the final net count was zero. The complete state hash during and after that control equaled the base hash, and its generated output matched baseline. A mismatched expected fingerprint was rejected with `base_mismatch` before a weight write.

## Inference, Runner, and replay

| Public smoke case | Expected | Baseline, composed edit, and replay output | Exact result |
| --- | --- | --- | --- |
| Integer-only arithmetic | `5` | `2 + 3 = 5` | Fail |
| Exact format | `OK` | `OK.` | Fail |
| Capital regression | `Paris` | `Paris` | Pass |

The composed candidate scored **1/3**, with 0/2 target passes and 1/1 regression pass. Nothing in these results establishes useful parameter semantics, generalization, or an advantage over another editing method.

| Execution | Handle | Measured wall time |
| --- | --- | ---: |
| Native loading, inspection, edits, controls, inference, and integrity checks | — | 19.581 s |
| Isolated Runner | `bd4e5b0d5c4b47d0a71b2c772ee75d3d` | 14.147 s |
| Fresh worker launched through actual CLI replay | `d1248e4535864ca0b0bf2d34494eceaa` | 13.835 s |
| Complete validator | — | 48.135 s |

The Runner persisted the exact read selections, three ordered weight-change records per case, final net evidence, and 13 generated tokens. It accounted for 17,333 bytes of inspection and weight evidence. Export/import verified bundle checksums. CLI replay matched the exact model identity, output texts, inspection data, all per-operation/net weight evidence, and specification digest. The original and replay specification digest was `726f7490c615a102b188bfa1351f0c824cc8387aa1a979255d06a537676375e2`.

The two portable bundles total **249,598 bytes** and contain declarations and measured results, not weight files. These timing observations include integrity hashing and startup; they are not throughput benchmarks.

After strengthening CLI replay to reject mismatching recorded evidence even when output text agrees, a further fresh-worker replay succeeded at `2026-11-10T08:27:18.696454+00:00`. Run `f4c6d63751ba48b6bb6ce489b80866c9` took **13.063 s** and returned `true` for `outputs_match`, `inspection_matches`, `identity_matches`, `sample_evidence_matches`, `weight_evidence_matches`, `activation_evidence_matches`, and `replay_matches`. This recipe records no activation captures, so its activation flag confirms matching empty capture lists; nonempty capture mismatch detection is covered by the CLI regression tests. The original acceptance artifacts remain unchanged.

## Evidence and reproduction

- [Measured summary](summary.json)
- [Native reads, edits, controls, outputs, and hashes](native.json)
- [Pinned executable experiment](recipe.json)
- [Persisted Runner result](runner-result.json)
- [Actual CLI replay output](replay.stdout.json)
- [Subsequent strict CLI evidence replay](replay-strict.json)
- [Original portable bundle](bundles/composed/manifest.json)
- [Replay portable bundle](bundles/replay/manifest.json)
- [Reproduction script](../../../scripts/validate_wave2_qwen.py)

From the repository root, with the pinned model already in the local cache:

```powershell
.venv/Scripts/python.exe scripts/validate_wave2_qwen.py --root .nrly-wave2-qwen-repeat --cache .cache/models --output docs/validation/wave2-qwen-repeat
```

The validator refuses to overwrite a nonempty evidence directory. It checks every stage, writes partial native/Runner artifacts as they become available, and preserves a failure record if a later stage fails. Exact replay is bound to the recorded model/runtime/device fingerprint; other configurations require their own validation.
