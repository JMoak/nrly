# Authoring reusable weight programs

The Python authoring API performs local calculations and emits the same validated
`ProgramSpec` that CLI, MCP, studies, and the native kernel already execute. Python
functions, imports, callbacks, and tensor artifact paths never enter a program.
MCP does not execute authoring code. Use the trusted local Python API or CLI to
prepare a candidate, then submit its declarative program through the usual interface.

## Executed tiny example

This example runs native PyTorch inference, packages a repair, applies it for one
inference context, and checks restoration. It is an engineering fixture, not evidence
of general model improvement. Run it in a fresh working directory with nrly's Torch
environment; the output directory must not already contain a patch package.

```python
from pathlib import Path

import torch

from nrly.authoring import WeightProgram
from nrly.backends.tiny import TinyBackend
from nrly.contracts import CaseSpec, ExperimentSpec, GenerationSpec, ModelSpec, ProgramSpec
from nrly.deployment import apply_patch_package, export_patch
from nrly.lineage import EditStore

root = Path(".nrly-authoring-demo")
case = CaseSpec(id="sum", input="[2,4]", expected="6", scorer="numeric")
with TinyBackend(ModelSpec()) as backend:
    program = WeightProgram(backend.identity.fingerprint).scale("output.weight", 2).build()
    revision = EditStore(root / "edits").create("repair A", program)
    export_patch(root / "patch", revision.program, backend.identity, revision)
    before = backend.run_case(case, ProgramSpec(), [], GenerationSpec()).output
    with apply_patch_package(backend.model, backend.identity, root / "patch"):
        with torch.inference_mode():
            edited = backend.model(torch.tensor([[2.0, 4.0]])).item()
    after = backend.run_case(case, ProgramSpec(), [], GenerationSpec()).output
    assert (before, edited, after) == ("3", 6.0, "3")
    spec = ExperimentSpec(name="Packaged tiny repair", cases=[case])
    (root / "experiment.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    print(before, edited, after)
```

Observed output: `3 6.0 3`. The following commands verify the package and execute
its pinned program through the ordinary isolated worker:

```powershell
nrly --root .nrly-authoring-demo/runs patch inspect .nrly-authoring-demo/patch
nrly --root .nrly-authoring-demo/runs patch run .nrly-authoring-demo/patch .nrly-authoring-demo/experiment.json
```

`patch run` replaces the experiment's program with the verified package program.
It retains the experiment's cases, generation settings, and budgets. A different
requested model is rejected before execution; a changed runtime fingerprint is
rejected by the worker. The result contains the patch ID, complete run, and evaluation.
`nrly patch export RUN_ID DESTINATION` packages a successful measured run; export
pins its recorded model fingerprint. Export never replaces an occupied destination.

## Calculations and tensor artifacts

`WeightProgram` supports `scale`, dense `add`, sparse replacement/addition through
`patch`, and matrix `low_rank` updates. `build()` returns a detached validated
`ProgramSpec`. Invalid appended operations leave the builder unchanged.

The numerical helpers operate on finite NumPy-compatible values:

| Helper | Calculation |
| --- | --- |
| `unit_direction(vector)` | Normalize a nonzero vector |
| `mean_direction(positive, negative)` | Unit difference of the two row-vector means |
| `project_out(values, direction)` | Remove the direction from the final axis |
| `builder.remove_input_direction(target, W, d, strength=1)` | Compile `W -= strength * (W @ unit(d)) @ unit(d).T` |
| `builder.steer_key(target, W, key, desired)` | Minimum-Frobenius rank-one delta satisfying `(W + delta) @ key = desired` in local arithmetic |

Rank-one calculations compile to `weight_low_rank`; they do not materialize a dense
delta in the program. Stored dtype rounding can change or erase the intended result.
The linear constraint and direction operation make no claim about downstream behavior.
Use weights from the exact intended state: a repair computed after A must use A's
edited weights before its resulting delta is appended to A.

```python
from pathlib import Path
from nrly.artifacts import TensorArtifactStore
from nrly.authoring import mean_direction

tensors = TensorArtifactStore(Path(".nrly-authoring-data"))
reference = tensors.put([[1.0, 0.0], [3.0, 0.0]], dtype="float32")
verified = tensors.inspect(reference.id)
positive = tensors.get(verified)
direction = mean_direction(positive, [[0.0, 0.0]])
```

Tensor artifacts contain a strict JSON manifest and raw little-endian float32 or
float64 bytes. Their content IDs bind shape, dtype, byte length, and payload checksum.
Reads verify bounds, identity, finite values, and SHA256; returned arrays are copies.
Handles are hexadecimal digests, not arbitrary paths. Data files do not load models,
deserialize Python objects, or execute code. Checksums detect content changes; they
are not publisher signatures or evidence that an edit is useful.

Each tensor artifact is limited to 16 MiB. Authoring calculations accept at most
16,777,216 input elements. Executable programs still have at most **32 operations and
65,536 explicit values across all payloads**, including both low-rank factors. Artifact
data does not bypass those limits. The kernel additionally caps distinct edited target
storage at 512 MiB before cloning; temporary peak memory can be higher. A patch manifest
or edit revision is limited to 4 MiB. No merged checkpoint export is provided.

## Sequential repairs and rollback

`EditStore(root).create(name, program, parent_id=None)` treats `program` as the new
delta. With a parent, it compiles the exact parent operations followed by that delta.
The child records its parent's content ID, program digest, and exact prefix length.
Every load verifies the whole ancestry. Existing revisions are never rewritten.

```powershell
nrly --root .nrly edit create "repair A" delta-a.json
nrly --root .nrly edit create "repair B" delta-b.json --parent A_ID
nrly --root .nrly edit show B_ID
nrly --root .nrly edit history B_ID
nrly --root .nrly edit select B_ID --ancestor A_ID
nrly --root .nrly edit select B_ID
```

Use the returned cumulative `revision.program` in an experiment or `CandidateSpec`.
`select --ancestor A_ID` returns A's program; omitting the ancestor returns pinned
identity for the original model. Selection changes no revision or source checkpoint.
Study admission continues to enforce its registered target, case, and budget constraints.

After evaluating the original, A, and A+B on matching cases and generation settings:

```powershell
nrly --root .nrly edit compare ORIGINAL_RUN A_RUN AB_RUN A_ID B_ID
nrly --root .nrly edit export A_ID A_RUN patch-a
```

The comparison verifies executed programs and exact base identity, then reports A
against the original, A+B against the original, and A+B against A. A second edit that
undoes an improvement is reported in `lost_prior_improvements`. Failed, incomplete,
unscored, truncated, or mismatched arms produce `inconclusive`. `no_observed_regression`
describes only the supplied cases. Even disjoint sparse coordinates can interact
behaviorally; the footprint never establishes independence. The tested tiny A doubles
the output weights, while B halves them: A repairs the target and A+B loses that repair.

Use `apply_patch_package` only with an exclusively owned model. It independently
checks base tensor bytes before application and after restoration, and restores edited
weights when inference raises. It provides no synchronization with other model users,
no Python sandbox, and no protection against a trusted callback mutating unrelated
state. Local files and caller-owned Python remain outside the MCP trust boundary.
