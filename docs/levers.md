# Levers: a map of tested sites, and the referee that inks it

A **lever** is a place in one exact checkpoint's weights, a direction found by a re-runnable
recipe, and what happened when that direction was edited. nrly records levers on a local
**map** and decides their standing with a registered held-out study.

Three rules shape everything here:

- **Levers, not meanings.** An entry is named by its test and its tensor, for example
  `reply-only.letter@model.layers.27.mlp.down_proj.weight`. Human-readable names appear
  only as attributed hypotheses.
- **Pencil and ink.** Anything derived, claimed, or inherited from a feature atlas is
  `pencil`. An entry becomes `inked` only through a completed real-model study with a
  registered inference rule and at least one matched control. A refuted entry stays on the
  map as `struck`. Status is derived from the attached verdict and cannot be set.
- **Recipes, not coordinates.** Coordinates expire with a checkpoint. The entry keeps the
  recipe that produced its direction, so the map can be regenerated on the next one.

Entries pin **weights**, not a machine. Evidence inside an entry records the runtime that
produced it.

## Workflow

| Step | Command | What it does | Evidence it produces |
| --- | --- | --- | --- |
| Derive | `nrly lever derive REQUEST` | Runs a recipe in this process on the loaded model and pencils the result | A direction artifact, a score for every candidate site, and matched controls |
| Dial | `nrly lever dial ENTRY REQUEST` | Runs the edit at each strength on development cases through the ordinary isolated runner | Exploratory dose-response; nominates one strength or none |
| Nominate | `nrly lever nominate ENTRY STRENGTH` | Operator override when the dial finds no working window | None; the recorded dial keeps its window, so the override is visible |
| Referee | `nrly lever referee ENTRY REQUEST HOLDOUT` | Operator-only. A registered study on a sealed holdout for exactly the nominated edit | The verdict that inks or strikes the entry |
| Legend | `nrly lever legend ENTRY REQUEST` | Names the direction from an inherited dictionary | Pencil-only hypotheses with attribution |
| Map | `nrly lever map FILE.html` | Writes the offline map page | No new evidence; aggregates only |

Also: `show`, `history`, `list`, `page`, `export`, `import`, and
`referee --replication` to repeat an existing verdict on another runtime. Every revision
is a new content-addressed entry that names its parent; nothing is rewritten.

Request schemas are available from `nrly schema lever-derive-request`,
`lever-dial-request`, `lever-referee-request`, `lever-legend-request`, `recipe`, and `lever`.

### Derive

The first recipe is difference of means. `positive` prompts exhibit the behaviour and
`negative` prompts do not. For every candidate site the direction is fitted on the
even-indexed prompts and scored on the odd-indexed ones, so a site is never chosen on the
data that produced its direction. Every candidate's score is kept, including the ones not
chosen. The edit removes the direction from the output side of one weight matrix:
`W <- W - strength * d (d^T W)`, compiled to one rank-one operation.

Deriving also prepares two matched controls at the same strength: the same removal in a
random orthogonal direction, and the same direction removed at the candidate site that
separated the prompts least.

Deriving is trusted local Python. It loads the model in this process and is not an MCP
operation. It requires a Hugging Face causal model and `[out, in]` linear weights.

### Dial

A dial measures each strength against the unedited original on the same development
cases and seeds. Each point carries a paired effect on target cases and on regression
cases. The **window** is the longest run of adjacent tested strengths that are superior on
targets and non-inferior on regressions. **Selectivity** is the target gain divided by the
larger of the registered regression margin and the observed regression cost.

A dial is exploratory. It selects a strength; it never changes a status. A strength that
fails to execute is reported with its run identifier and left out of the curve.

### Referee

The referee builds one study for the nominated edit: the original, a repeated no-op, every
planned control, every registered comparator, and the candidate, on development cases and
then once on the sealed holdout. One candidate, no search. The study's assessment decides
the status. If the development attempt does not complete, the attempt stays in the study
ledger and the entry keeps its status.

A replication must come from a runtime with no recorded verdict for the entry, and is
refused before any model runs otherwise.

### Legend

nrly does not train or host feature dictionaries. A legend reads one decoder matrix from
an existing public dictionary, by ranged request, and lists the features nearest the
lever's direction with their cosines and the share of the direction they explain. The
dictionary is pinned by repository, exact revision and the SHA-256 of the tensor bytes.

A legend is always pencil. Published comparisons show dictionary features are not
reliably the best editing directions, so a legend is never evidence of an effect. Labels
are fetched from Neuronpedia only on request and are recorded with their source.

## The referee core (study schema 0.3)

These are opt-in. Study schemas 0.1 and 0.2 behave as before.

| Field | Meaning |
| --- | --- |
| `truncation: "fail"` | An output cut off at the token limit is a scored failure, and reports include finish counts. The default leaves truncation unscored. Scorer version `nrly-scorers/3`. |
| `inference` | A registered rule: confidence level, regression margin, bootstrap resamples and seed. |
| comparator `kind: "generation"` | The unedited model under different decoding settings, for example thinking disabled or a repetition penalty. |

Under an inference rule, **cases are the independent unit**. Each case contributes its mean
paired difference across seeds, so seed repeats never inflate the sample. The interval is
a percentile bootstrap over cases; the p-value is an exact two-sided sign test. A
candidate is supported only when its target effect is superior to the original and to
every registered control and comparator, and its regression effect against the original
is non-inferior within the margin. Requiring every comparison is an intersection-union
test, so the joint claim keeps the level of each one. Fewer than six cases per role
cannot reach significance and are refused at admission.

`no_change` means no supported target effect with regressions preserved. It is not an
equivalence claim.

## Portable patches

`nrly patch export RUN DESTINATION --portable` writes a package that pins the checkpoint
(`model_id`, revision, dtype, weights digest) and carries no runtime fingerprint.
`nrly patch run` on such a package first loads the model here to learn the local runtime,
refuses if the weights differ, binds the operations to the local fingerprint, then runs.
The earlier package format is unchanged and still pins one runtime.

## Boundaries

- The map is a local store. There is no hosted registry, signing, or publisher identity.
  A digest shows content has not changed; it does not show a claim is true.
- The holdout boundary is enforced by the application, as for every study. A filesystem
  owner can read it.
- One recipe kind and one edit kind exist. Additive steering, input-side removal,
  multi-site derivations, tied or quantized weights, and merged-checkpoint export are not
  implemented. Whether a lever survives quantization or fine-tuning is untested.
- Site selection uses held-out separation of prompt activations, which is a correlational
  score. The dial and the referee are the causal tests.
- Generation runs one prompt at a time in a fresh worker per run. Large benchmarks are
  slow.
- The map page excludes prompt text and case content. Exported lever bundles include the
  recipe's contrast prompts; share them accordingly.

See the [recorded acceptance](validation/lever-qwen/README.md) for what was actually
measured on a real model.
