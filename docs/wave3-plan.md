# Research workflow implementation plan

Agreed 10 November 2026. This wave extends nrly from verified edit execution to
diagnosis, realistic experiments, comparative evaluation, and reusable edits.
Implementation and acceptance results are recorded separately; this plan does not
claim model improvements.

## Component boundaries

| Component | Responsibility | Boundary |
| --- | --- | --- |
| Runtime and diagnostics | Chat rendering, generation, teacher-forced continuation scores, selected activations, trusted Python model adapters | No task-quality verdict or arbitrary code loading through MCP |
| Studies and evaluation | Paired seeds, registered prompt/model comparators, deterministic task adapters, group metrics and sealed final evaluation | Descriptive seed variation is not a population confidence interval |
| Authoring and artifacts | Bounded tensor calculations, validated programs, immutable edit ancestry, portable patch application | Existing inline program limits remain enforced; no implicit cross-model transfer |
| Application and transports | Shared CLI/MCP operations, bounded evidence pages, generated TypeScript contracts | Tensor mathematics stays in Python runtime modules |

## Compatibility and resource invariants

- New execution features use experiment schema 0.3. Legacy 0.1/0.2 default values
  retain their canonical serialization and replay identities.
- Expanded studies use study schema 0.2. Prior study records remain readable.
- Every run restores the base, retains failures, and records model/runtime identity.
  Full-state verification remains enabled.
- Teacher-forced observations are diagnostic evidence, never generated answers or
  task accuracy. Input/output token accounting and bounded evidence remain explicit.
- Repeated seeds and all comparator arms consume reserved study budgets, including
  the final evaluation. Prompt transformations and comparator models are registered.
- Authoring is trusted local Python. Declarative remote inputs cannot load code.
  Tensor artifacts are bounded numeric data with verified hashes; no pickle loader.
- CLI/MCP exposure must page potentially large diagnostics and conversation text.

## Implementation sequence

1. Freeze runtime, scoring and study contracts across the three implementation owners.
2. Implement independent runtime, study, and artifact modules with focused CPU tests.
3. Integrate CLI, MCP, SDK, schemas, reproducible examples and public documentation.
4. Run complete Python and TypeScript suites, real MCP lifecycle tests, and backward
   artifact verification. Resolve failures before GPU acceptance.
5. Run a pinned Qwen diagnostic and sampled experiment through real workers; verify
   restoration, negative controls and fresh replay. Separate successful execution
   from any measured behavioral result.
6. Build and inspect distributions, rerun publication checks, and record limitations.

## Accepted scope

Continuation probabilities, selected activation observations, chat histories, explicit
decoding controls, paired seed studies, prompt and materialized-model comparators,
deterministic task scorers, trusted Python model execution, authoring helpers,
verified tensor artifacts, edit ancestry and portable declarative patches are in scope.

Vector batching, a resident cross-run process service, distributed inference, quantized
editing, arbitrary code sandboxes, merged checkpoint export and hosted community access
need separately validated designs. The local Python adapter can reuse a loaded model
without changing the isolated Runner's lifecycle or weakening restoration checks.

## Research basis

- [WSP-Scope](https://arxiv.org/html/2611.09382v1): teacher forcing, activation
  observations, sampled reasoning, and matched diagnostic controls.
- [Original WSP report](https://rikiyat.github.io/blog/weight-space-programming/):
  weight analysis and direct authoring across language models and smaller networks.
- [WeightLoop](https://arxiv.org/abs/2611.07319): persistent predictor changes and
  evaluation as observations arrive; adaptive rule synthesis remains beyond this wave.
- [EasyEdit](https://github.com/zjunlp/EasyEdit): separate edit reliability,
  generalization, locality and sequential-edit evaluation in adjacent model editing.

These motivate software interfaces; nrly has not reproduced their reported gains.
