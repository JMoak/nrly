# Evaluation semantics

Current evaluation reports and newly produced study assessments identify their
policy as `nrly-scorers/2`. Historical study assessments without this field retain
its absence when loaded and serialized. Recorded run
and experiment hashes are unchanged. Re-evaluating an older run can produce a
different assessment: an output with `finish_reason="length"` is now unscored,
even if that output happens to match the expected answer. Explicitly truncated
outputs, unsuccessful inference and teacher-forced diagnostics are also unscored.
A legacy record with no truncation or length-stop evidence remains scoreable.
Historical reports retain their recorded scorer version and are not rewritten.

The `exact`, `numeric` and `final_number` answer matching rules remain available.
An incomplete or unscored arm cannot establish a controlled-study improvement.
Teacher-forced probabilities describe a supplied continuation; they are never a
generated answer or a task-accuracy score.

## Versioned task adapters

`CaseSpec.adapter` uses `nrly-tasks/1` and requires experiment schema 0.3 or study
schema 0.2. Adapter settings cannot be combined with numeric or final-number
scoring. Each evaluated case reports its adapter version and kind.

- `multiple_choice` accepts exactly one registered label after trimming surrounding
  whitespace. This is generated-label scoring, not option log-likelihood scoring.
- `json` compares strict JSON structure. Object order is irrelevant; array order
  matters; booleans are distinct from numbers. Decimal numbers are compared exactly,
  so `0.10` equals `0.1` while `0.10000000000000001` does not. Duplicate object keys,
  non-finite values and Markdown fences are rejected. The scorer bounds input to
  1,000,000 characters, nesting to 64 levels, numeric literals to 4,096 characters,
  and the absolute decimal exponent to 10,000. An unsupported expected answer is
  unscored; an unsupported model answer fails. This is structural equality, not a
  general JSON Schema validator.
- `instruction_constraints` checks case-sensitive required and forbidden literal
  substrings and an optional character ceiling. Its registered constraints define
  correctness, so `expected` can be omitted. It does not judge arbitrary instruction
  following or semantic correctness.

## Repeated studies and comparators

Registered seeds pair the same cases across baseline, no-op, controls, comparators
and candidate. Reports distinguish cases per seed from paired executions and show
mean, sample standard deviation, minimum and maximum paired score deltas across
registered seeds. A single seed has no sample standard deviation. These are
conditional descriptive observations, not a population confidence interval or
proof of independent samples. Named groups preserve separate generalization,
locality or other operator-defined slices without claiming that the labels alone
establish those properties.

Prompt comparators wrap the original plain input or final user message with their
registered prefix and suffix. Model comparators use a registered pinned ModelSpec
and separately recorded runtime identity. A materialized checkpoint trained
elsewhere can be compared; nrly does not train or load adapters through this
interface. Generic run comparisons remain strict about model identity. All arms
and seeds consume study reservations, including final evaluation and failed
attempts. Holdout prompts remain sealed and are checked for overlap with registered
prompt transformations.

## Registered truncation scoring and inference (study schema 0.3)

Study schema 0.3 adds three opt-in registrations. Studies that do not use them, and all
0.1 and 0.2 studies, are scored exactly as described above.

- `truncation: "fail"` scores an output cut off at the token limit as a task failure
  instead of leaving it unscored. Reports then carry `finished` and `finished_passed`
  counts, so finish rate and accuracy-when-finished can be read separately from accuracy.
  Such reports and assessments identify their policy as `nrly-scorers/3`. Teacher-forced
  diagnostics, failed inference, missing samples and integrity failures remain unscored.
  Use this when running to the token limit is the behaviour under study.
- `inference` registers a paired rule. Cases are the independent unit: each contributes
  its mean paired difference across the registered seeds. The interval is a percentile
  bootstrap over cases, and the p-value is an exact two-sided sign test on the nonzero
  case differences. The verdict is `observed_improvement` only when the target effect is
  superior to the original and to every registered control and comparator, and the
  regression effect against the original is non-inferior within the registered margin.
  One flipped case among many no longer yields `regressed`. At least six target and six
  regression cases are required in both the development set and the holdout.
- A comparator of `kind: "generation"` runs the unedited model under registered decoding
  settings. Its seed is replaced by each study seed.

The count-based comparison summaries are still reported beside the inference results.
`no_change` under an inference rule means no supported target effect with regressions
preserved; it is not an equivalence claim. See [levers.md](levers.md) for how the lever
map uses these studies.

## Execution status and scientific verdict

A completed, fully scored and comparable study can return the scientific verdict
`inconclusive` when the candidate improves on the base but does not beat every
registered comparator. That is valid negative evidence: discovery attempts have
status `succeeded`, and final evaluations have no lifecycle error, while the
recommendation remains `retain_original`. Such a discovery attempt may be
nominated for sealed final evaluation by the operator. Completing execution never
implies that an improvement was demonstrated.

Missing or unscored arms, identity mismatches, no-op disagreement and other failures
of the registered comparison remain lifecycle failures (`study_inconclusive`).
Existing terminal attempts, final records and event ledgers are not rewritten by
this classification change.
