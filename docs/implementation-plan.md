# nrly Implementation Plan

Implementation plan · revised 10 November 2026

nrly is a local environment for **weight-space programming**: inspecting neural-network parameters, authoring direct weight edits, executing the resulting model, and preserving the modification together with the evidence needed to assess it. Activation interventions support diagnosis; learned latent memory and prefixes are outside the first release.

The initial product is a dependable weight-programming workbench. The larger opportunity is a common workflow for executable parameter changes, explicit assumptions, reproducible runs, independent evaluations, and useful failures. The corrected research audit in `C:/src/nrly/docs/research-audit.md` reexamines direct WSP evidence and competing tools. OpenWSP already offers a kernel, CLI, SDK and MCP; nrly's differentiation must be validated through usability, dependable local execution and stack integration.

Implementation now lives in `C:/src/nrly`. The repository README and validation evidence describe the current MVP; the rest of this document retains the broader roadmap. Existing Keywork, Stead, and Plaia applications remain independent consumers. Publication is not authorized. The earlier interactive concept still uses synthetic data and is not the implemented interface.

## Product thesis and evidence

The potentially important shift is that model internals become an authored computational medium. People and agents may increasingly specify, inspect, and revise neural computation directly. nrly should preserve both the executable modification and the conditions under which it worked. Explanations remain hypotheses until the measurements support the relevant claim.

The following background references were collected in the earlier design pass. Direct WSP evidence, new follow-up studies, competing workbenches, and the corrected product conclusions are refreshed in `C:/src/nrly/docs/research-audit.md`.

| Existing work | Established contribution | Consequence for nrly |
| --- | --- | --- |
| [Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) | Reports agents reading and directly editing weights, with controlled model execution and without gradient training in the reported experiments. Its authors also describe work on a benchmark and harness. | Preserve direct tensor authoring, observation restrictions, execution budgets, and complete attempt history. Do not claim to originate this field or its first harness. |
| [NNsight](https://nnsight.net/) | Supplies access to model internals and activation interventions. | Consider interoperability after the native PyTorch backend; activation tooling is supporting infrastructure. |
| [pyvene](https://arxiv.org/abs/2403.07809) | Provides configurable static and trainable interventions on PyTorch models. | Study its intervention abstractions and add interoperability when a concrete recipe benefits. |
| [nnterp](https://arxiv.org/abs/2511.14465) | Normalizes access across model architectures while retaining original Hugging Face implementations and providing validation tests. | Evaluate it during the backend spike; use explicit native coordinates and parity checks regardless of adapter. |
| [Neuronpedia](https://docs.neuronpedia.org/) | Provides model exploration, steering, feature data, APIs, and research presentation. | Make external research references and imports possible. Validate demand for nrly against existing workflows. |
| [ObserverBench](https://arxiv.org/abs/2609.03026) | Separates accuracy of internal estimates from the loss of actions chosen using those estimates. | Score explanations, interventions, and end-task outcomes separately. |
| [SAEScientist Bench](https://arxiv.org/abs/2609.09113) | This September 2026 preprint evaluates agent-driven feature discovery and reports measurement interpretation failures and weaker causal steering than expert baselines. | Treat agent interpretation as fallible. Use independent scoring and contrastive controls. |

The differentiation hypothesis is an integrated, local workflow for authoring neural changes with people or agents, with machine-readable execution semantics and independently reproducible evidence. Validate this with researchers before investing in a hosted community platform. General experiment logging already exists; [Inspect's evaluation logs](https://inspect.aisi.org.uk/eval-logs.html) are an interoperability reference, rather than a reason to create a competing general evaluation framework.

## First release scope

The first release supports a tiny deterministic neural fixture and one pinned, small open language model. It can run an unchanged baseline, capture selected internal tensors, apply one activation intervention or weight patch, run comparisons, and export a reproducible bundle. It supports both a local researcher and a Keywork agent using the same kernel.

Use [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) as the first real-model candidate. The feasibility milestone must select and record an exact revision, tokenizer, generation template, numerical precision, library lock, and tested device. Model size alone does not establish instrumented memory requirements. If this candidate fails the required adapter checks, record the reason and select a compatible small dense transformer.

Two reference experiments serve different purposes:

1. A deliberately perturbed tiny network with a known correction proves that trace capture, tensor targeting, editing, comparison, and restoration are correct. This is an engineering fixture, not evidence of novel model repair.
2. A real-model behavior experiment, initially repeated reasoning, measures whether an intervention helps on fresh examples without degrading other behavior. A negative result remains a valid completed experiment.

The scientific result must include task correctness as well as repetition or termination metrics. Forcing early termination is not a successful repair when answers become less correct.

The initial release excludes universal neuron labels, cross-model patch portability, distributed training, unrestricted agent code execution, automatic modification of everyday inference models, and a hosted marketplace. The existing stack can consume explicitly selected results through later deployment adapters.

## Architectural decisions

| Decision | Implementation consequence |
| --- | --- |
| Independent research tool | The nrly kernel imports neither Keywork nor Plaia. Researchers can use Python and the CLI without installing the desktop applications. |
| One execution implementation | Python owns model operations, the run state machine, and persistence. CLI and MCP transports call those same application handlers. |
| Thin TypeScript integration | Generate a TypeScript contract/client package for the existing applications. Keep tensor operations and job state transitions out of the renderer. |
| One initial backend | Use original Hugging Face/PyTorch execution with native hooks, validated directly in the MVP. NNsight remains a possible later adapter. Additional adapters must pass conformance tests. |
| Separate author and target | Keywork's usual provider conducts the investigation. A separate worker loads the model being studied. |
| Explicit program kinds | Direct weight scaling, full replacement and sparse writes are primary. Activation interventions are diagnostics with different lifetimes. Learned context, trained adapters, and procedural controllers are outside the MVP. |
| Immutable scientific inputs | Model identities, datasets, program versions, and completed run manifests are immutable. Corrections create new versions with lineage. |
| Scoped execution | Each job has an experiment workspace, supported operations, input scope, and resource budget. Imported descriptions cannot grant capabilities. |
| Local artifact format | Human-readable manifests and reports reference content-addressed payloads. The format can be used independently of a hosted service. |
| Narrow public commitments | Publish an experimental versioned schema. Stabilize fields after independent implementations and real recipes exercise them. |

The normal process arrangement is one lightweight Python runner and one isolated compute child per active device slot. The MCP entry point hosts the same runner logic; it does not introduce a second scheduler. The CLI runs the same application commands, with an exclusive writer lease for an experiment workspace. Concurrent ownership is refused until attach/resume behavior is explicitly implemented.

The first MCP transport is local stdio. A future local or remote service requires a separately specified authentication and ownership boundary. There is no need for a broker, distributed scheduler, or always-running daemon in the first release.

### Repository layout

The current MVP uses this layout. Additional client/UI packages remain roadmap items:

```text
nrly/
  pyproject.toml
  uv.lock
  src/nrly/
    contracts.py        typed manifests and public errors
    kernel/             operation semantics and validation
    backends/           original Hugging Face/PyTorch implementations
    runner.py           job ownership, budgets, cancellation
    store.py            artifact persistence and recovery
    evaluation/         sample results and comparison protocol
    cli.py              researcher entry point
    mcp.py              agent entry point
  schemas/              generated JSON Schema
  packages/client/      generated types and thin TypeScript client
  recipes/              reproducible examples
  tests/
    contracts/
    kernel/
    scientific/
    integration/
  docs/
    architecture.md
    experiment-format.md
    adapter-authoring.md
    decisions/
```

Author public typed models once, emit JSON Schema, and generate the TypeScript types. Keep semantic checks such as target shape and token alignment in the kernel. Cross-language fixtures must validate the same accepted and rejected messages. Changes to generated contracts are checked in CI.

Use uv, strict Python typing, Ruff, and pytest for the kernel. Use the existing Bun/TypeScript conventions, Biome, and Vitest for the client. Adopt Vibrissa for MCP contract tests. Exact runtime dependency versions and supported Python/device combinations are chosen from the feasibility results, then locked. Do not install nrly's ML dependencies into Plaia's environment.

## Core experiment contracts

Keep four public records small and explicit:

| Record | Required meaning |
| --- | --- |
| `ExperimentSpec` | Behavioral question, model identity, dataset and split identities, baseline definitions, metrics, permitted operations, and budgets. |
| `ProgramSpec` | Program kind, exact targets, ordered operations, activation schedule or weight payload, preconditions, and provenance. |
| `RunRecord` | Resolved inputs, program hash, actual environment, attempts, seeds, observed outputs, resource use, errors, and artifact references. |
| `EvaluationReport` | Exact runs and scorer versions compared, per-case outcomes, exclusions, uncertainty, regressions, scope of the claim, and prior evaluations. |

A nested `ModelRef` records checkpoint revision and payload digest, configuration, tokenizer and chat-template digests, dtype/quantization, backend version, and relevant execution settings. A nested target reference records the native module/tensor path, shape, axes, and aliasing information. Future SAE references also pin their dictionary and normalization; a feature ID has no standalone universal meaning.

The program manifest carries semantic operation data. Reports carry behavioral claims. A tensor coordinate is not a claim that a neuron has a particular meaning.

### Initial operations

| Family | Initial operations | Required guard |
| --- | --- | --- |
| Observe | Describe module/tensor; bounded tensor slice/statistics; capture selected activations. | Explicit location, shapes, selected examples/tokens, and capture-byte ceiling. |
| Activation | Zero, scale, add a supplied vector, or replace a selected activation from an identified source run. | Exact axes and token alignment; declared execution phase and schedule. |
| Weight | Replace, scale, or add a supplied tensor slice in a candidate branch. | Base hash, target shape/dtype, alias handling, finite values, and operation order. |
| Execute | Run a fixed case set under a specified generation policy. | Budget accounting and complete attempt identity. |
| Compare | Score unchanged, prompt-modified, sham, and candidate runs. | Compatible cases, scorers, environments, and recorded exclusions. |

For activation programs, distinguish prefill positions from generation steps and specify whether cached states are affected. Hooks are scoped to a run and removed on success, failure, or cancellation. First support one intervention per run; accept ordered multi-operation programs only after overlap and ordering semantics are tested. A matching vector dimension is insufficient to establish semantic compatibility.

For weight programs, preserve the base artifact and apply patches in an isolated candidate. Handle tied tensors explicitly. Persist compact replacements or deltas where possible instead of duplicating a whole model per attempt. The execution worker may reload the clean model between candidates; optimize reuse only after state-leak tests pass.

A trained KV prefix belongs to a later learned-context adapter. [Cartridges](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges) uses gradient-based prefix training; it must not be reported as gradient-free weight programming. Each recipe declares its use of gradients, training data, forward passes, and other information.

## Runtime and artifact reliability

The runner owns job transitions: `queued`, `running`, `cancelling`, and terminal `succeeded`, `failed`, `cancelled`, or `interrupted`. A completed experiment with no improvement can be `succeeded`; scientific benefit is a separate result.

Submission is idempotent by caller request ID and validated specification hash. An attempt has its own ID. Retrying a failed attempt does not erase its cost or evidence. On process restart, incomplete execution is marked interrupted; restarting it creates another attempt. Do not promise resumption in the middle of an arbitrary neural forward pass.

Store run metadata and transition events transactionally in local SQLite. Put large, immutable payloads in a content-addressed filesystem store. Use staged writes, checksums, and atomic publication; publish metadata references only after payload completion. Recover orphan payloads conservatively. Export completed runs as portable manifests, case records, reports, and safe tensor payloads. Exported bundles are independent of the local job database.

This database belongs to nrly's experiment service. It does not replace Keywork's file-based memory vault. Deleting an experiment follows explicit retention policy; active and pinned artifacts cannot be garbage-collected.

Resource budgets cover wall time, generated tokens, examples processed, model executions, capture bytes, artifact storage, and available device capacity. Count cumulative work across attempts. GPU memory checks are admission estimates, not guaranteed reservations against unrelated processes. Begin with serial device use and expose measured consumption.

Use cooperative cancellation first, followed by terminating the owned worker process tree after a bounded grace period. Prove cleanup on Windows and Linux for advertised environments. The worker has no authority to modify shared base-model files or other experiments. Return typed errors such as `unsupported_operation`, `target_mismatch`, `budget_exceeded`, `worker_lost`, and `artifact_incomplete`.

Provide bounded result summaries and artifact handles through MCP. Large activation arrays never flow through ordinary chat messages. Capability discovery distinguishes read access, activation intervention, weight editing, and gradient support. Unsupported combinations fail before a job is admitted where they can be detected.

## Scientific validity

The evaluator must answer three separate questions: did the program execute correctly, did it change the measured computation, and did it improve the intended task? A plausible explanation does not establish the last two.

Every reference experiment defines:

- An unchanged baseline with the same model and decoding settings.
- An identity intervention to detect instrumentation effects.
- A serious prompt baseline when the behavior can be addressed through instructions.
- Appropriate sham controls, such as a matched-magnitude intervention at another permitted location.
- Target-task quality and unrelated regression tasks.
- Discovery, validation, and final evaluation splits with frozen identities.
- Repeated trials or paired sampling where nondeterminism matters, with per-case results and uncertainty.
- Actual exploration cost, including failed and discarded candidates.

A fine-tuning baseline is required before claiming superiority to fine-tuning; it is not required to prove the workbench functions. A search or sweep baseline is required before attributing gains specifically to an agent's reasoning rather than its experiment budget.

Validation data can guide iteration. Final evaluation data is held by an evaluator outside the author's authorized workspace. Freeze the candidate before final evaluation. Repeated access is recorded, and subsequent use of revealed feedback changes the interpretation of the result. A split label or hash alone cannot enforce secrecy. Public datasets may also have prior contamination, which should be recorded when relevant.

Plaia is the first external evaluator integration. Add a task adapter and a strict real-model mode: stub fallback, missing measurements, and infrastructure errors cannot become successful scientific scores. nrly retains standalone sample scoring so that a researcher need not adopt Plaia. Later adapters can use existing evaluation frameworks without reimplementing their orchestration.

Reproducibility has explicit levels: exact deterministic fixture replay; numerical agreement within recorded tolerances on the pinned stack; and independent replication across machines or configurations. Seed recording alone does not promise bit-for-bit GPU replay.

## Agent programming and containment

The agent operates nrly through a small set of tools: discover capabilities, inspect bounded tensors, submit a trace or intervention job, read job status, read bounded results, cancel a job, and export an experiment. Each state-changing tool uses the same validated application handler as the CLI.

Begin with declarative operations. Direct tensor patch payloads are already genuine weight edits. To support the more open-ended weight reasoning in the original research, add a controlled Python authoring environment as a distinct milestone. It should permit numerical analysis and candidate construction while model execution remains routed through counted operations.

MCP and a Python subprocess do not supply a security boundary. Arbitrary code requires an enforced worker environment with scoped files, no inherited application credentials, bounded execution, and mediated model access. Reuse the existing containerized-workshop approach only after validating its actual CPU/GPU and Windows/Linux behavior. Inspection of a received bundle must never execute its code, load a pickle, or fetch an unreviewed plugin.

For benchmark tracks claiming limited observations or no gradients, record each restriction as enforced, audited, or declared. Declarative operations can enforce a restricted operation set. General Python plus raw weights can reconstruct forward computations or gradients independently, so mediation of the official model API cannot establish that all such computations were prevented. Open-ended tracks must retain code and execution records for audit and state the limits of their accounting. A trusted notebook is not an enforced track. Record the author model identity and visible tool exchange where available; do not treat an agent's narrative as faithful evidence of its internal reasoning.

## Integration with the existing stack

| Repository | Boundary and planned change |
| --- | --- |
| nrly | Independent Python kernel, runner, artifact store, CLI, MCP server, and generated client contracts. |
| Keywork | First use the [existing MCP registry](C:/src/keywork/packages/engine/src/mcp/registry.ts). Normal inference and creature permissions remain owned by Keywork. Add a small server-side experiment port only when the dedicated pane requires it. |
| keywork-app | Add a Solid workbench consuming Keywork HTTP/SSE through its existing host connection. Respect the [client-only boundary](C:/src/keywork-app/AGENTS.md); the renderer never runs model code or handles model credentials. |
| Stead | Consume the released desktop SDK and verified runtime under its [artifact ownership rules](C:/src/stead/docs/repos.md). Coordinate shared-runtime changes through the existing release procedure. |
| Plaia | Extend [benchmark tasks](C:/src/plaia/plaia/bench/runner.py) with an nrly adapter and independent scoring. Keep environments and result ownership separate. |
| Vibrissa | Use its [MCP contract testing](C:/src/vibrissa/README.md) for schemas, cancellation/status flows, error results, and manifest drift. Numerical model outcomes belong in scientific tests. |
| Ratchet | Reuse existing authorization or calibrated-decision paths where relevant. Deterministic compatibility and budget checks remain code. |
| Relly | Optional research/source adapter later. No first-release dependency. |
| Orchard | Reuse scoped work, decisions, verification, and release conventions as development tooling. It is not an end-user runtime dependency. |

The Python MCP adapter is an intentional refinement of the earlier TypeScript-package sketch. It removes duplicate orchestration and keeps a researcher's installation simple. The [TypeScript MCP template](C:/src/ts-mcp-template/README.md) still informs transport and contract-test patterns; it need not force a second server implementation.

## User experience

The first interface is a CLI with a Python API. Proposed commands are `nrly doctor`, `nrly run`, `nrly compare`, `nrly replay`, and `nrly export`; these names are a design, not installed commands. A novice can run the included fixture; a researcher can express an intervention in Python; a Keywork agent can invoke the same operations.

The later desktop view has four main activities: reproduce a behavior, inspect and edit an intervention, compare outcomes, and inspect/export the experiment. Display the exact model and program version, cost, failures, and evaluation scope. Show explanations as authored hypotheses. Preserve the negative cases next to positive results.

Keep detailed trace views on demand. An expert should be able to reach the underlying code and coordinates; a new user should be able to operate a provided recipe without first learning tensor indexing. The workbench is a projection of runner state, not a second implementation of experiments.

## Milestones and acceptance

| Milestone | Deliverable | Acceptance evidence |
| --- | --- | --- |
| M0 Feasibility | One selected real model/backend/device tuple plus the CPU fixture. Dependency locks and measured resource needs. | Native versus instrumented no-op outputs agree within specified tolerances; selected capture and intervention locations work; clean-model state survives a failed run. |
| M1 Executable experiments | Python API and CLI, initial manifests, bounded capture, activation operations, direct weight patches, export/replay. | Known fixture repair works, wrong targets are refused, original weights are preserved, and another clean environment can replay the fixture bundle. |
| M2 Reliable evidence | Durable jobs, budgets, cancellation, independent comparison, baseline and regression recipes. | Forced crash and cancellation tests pass; interrupted results cannot be scored as complete; real-model comparison reports all cases and candidate costs. |
| M3 Agent operation | MCP tools, Keywork connection, contract suite, bounded agent investigation. | A Keywork agent completes a fixture investigation using tools; replay reproduces its edits; budget limits hold; missing nrly leaves normal chat usable. |
| M4 Open-ended authoring | Controlled Python analysis and program construction with mediated execution. | Scope, model-call accounting, forbidden operations, and worker cleanup are tested; trusted and enforced tracks are distinguishable. |
| M5 Product alpha | Thin Keywork API integration, Solid workbench, release packaging, documentation, and independent trials. | UI and CLI reference the same runs; installed-product compatibility is tested; external users can run, modify, and share a recipe without author assistance. |

M1 is the first independently useful release. M3 is the first agent-driven product. M4 is needed before advertising broad autonomous weight-space programming. The desktop and community layers follow proven experiments.

### Initial implementation backlog

These planning IDs are not accepted Orchard work-item IDs. Convert them into scoped tasks in their owning repositories when implementation begins.

| ID | Work | Depends on | Owner | Completion condition |
| --- | --- | --- | --- | --- |
| N01 | Backend feasibility and hardware probe | None | nrly | Exact tested tuple, dependency lock, parity evidence, and memory/capture measurements. |
| N02 | Contracts and typed errors | N01 | nrly | Generated schemas/types and shared valid/invalid fixtures; clear version rules. |
| N03 | Native baseline and immutable run bundle | N02 | nrly | CPU fixture and one real-model baseline export and replay. |
| N04 | Capture and activation operations | N03 | nrly | Token/axis tests, identity parity, hook cleanup, and capture limit enforcement. |
| N05 | Persistent weight patches | N03 | nrly | Base-hash checks, tied-tensor handling, deterministic fixture repair, and failed-edit cleanup. |
| N06 | Job ownership, storage, budgets, cancellation | N03 | nrly | Duplicate-submit, crash/restart, partial-payload, and process-cleanup acceptance cases. |
| N07 | Comparison protocol and evaluator adapter | N04, N05, N06 | nrly and Plaia | Baseline/sham/regression reports, strict no-stub path, and frozen final evaluation flow. |
| N08 | MCP and Keywork agent recipe | N06, N07 | nrly; Keywork configuration | Vibrissa contract suite and a bounded real agent investigation. |
| N09 | Controlled Python authoring | N08 | nrly | Enforced scope and accounting with a reproducible direct-weight program. |
| N10 | Desktop server port and protocol mirror | N08 | Keywork and keywork-app | Contract fixtures, authorization behavior, and reconnection tests. |
| N11 | Workbench interface | N10 | keywork-app | Actual runs rendered with comparison, cancellation, provenance, and export. |
| N12 | Alpha packaging and independent replication | N07, N08; N11 for desktop | nrly and relevant product owners | Clean installation, documented tested environments, independently replayed recipes, and release evidence. |

After N03, N04 and N05 are separable implementation concerns. N06 can develop against the deterministic fixture while real-model work continues. A dedicated UI can use recorded contract fixtures after N02, but it cannot establish completion of the underlying runtime.

## Verification and release discipline

Use a small, fast deterministic suite on every change and more expensive reference-model acceptance when changes touch model semantics or releases advertise support. Do not require every PR to download a large model.

- Contract checks: schema generation, semantic validation, version rejection, and typed errors across Python, MCP, and TypeScript.
- Kernel checks: numerical reference tests, no-op parity, correct locations and token alignment, hook lifetime, tensor aliasing, and candidate restoration.
- Runtime checks: idempotency, concurrent ownership refusal, bounded artifacts, quota exhaustion, interruption, crash recovery, and process-tree cleanup.
- Scientific checks: deliberately ineffective and harmful interventions, correct regression accounting, scoring failures, data-split access, and reproducible baseline comparison.
- Integration checks: normal Keywork use with nrly absent or failed, permission denial, protocol mismatch, UI reconnect, and product artifact compatibility.
- Release checks: fresh installations on advertised platforms, exact dependency/model provenance, checksummed distributions, documented hardware requirements, independent recipe replay, and confirmed distribution names and licenses before publication.

A GPU job failing or being skipped is visible in the release matrix. It is not a passing backend certification. Quantized, accelerated, and multi-device backends require their own acceptance rows. Backend parity and correctness take priority over throughput during the first release.

Record major choices as short architecture decisions: state ownership, contract generation, backend selection, artifact identity, evaluator isolation, and code-execution boundary. Use existing repository governance when changing shared code. This proposal does not require modifying Orchard itself or starting work outside an accepted scope.

## Adoption and future expansion

Start with reproducible example bundles and a contribution template. Invite a small group of researchers to perform a concrete task: reproduce a result, change an intervention, and submit a report someone else can run. Measure setup time, successful independent replays, time to the first valid intervention, and failure diagnosis. These are initial product targets, not measured results.

Every contribution records model/runtime compatibility, data permissions, supported operations, costs, complete attempt lineage, and per-case evidence. Negative findings are valid contributions. Provide a compact report suitable for an issue or paper supplement. A future registry can index these bundles and independent replications; it does not need to host models or private datasets.

Future work should follow observed demand:

1. Additional adapters and imported feature dictionaries, with explicit coordinate mappings and conformance tests.
2. Learned prefixes, memory readers, and trained adapters as separate executable artifact kinds.
3. Synthesis of small networks or controllers from behavioral specifications, using explicit initial weights and executable edits. Environment adapters supply independent tests of the resulting computation.
4. Ordered program composition with interaction tests; successful individual programs do not automatically form a successful combination.
5. Cross-checkpoint studies that record candidate correspondences, alignment methods, uncertainty, and fresh causal tests. Location similarity alone is not transferred functionality.
6. Runtime deployment adapters for explicitly selected programs, separate from the experimental runner.
7. A searchable community registry once bundles are actually being exchanged and reproduced.

The format should accommodate these paths, while the first product stays focused on a complete, trustworthy experiment. nrly earns a position in the field when other people can build on its results with less effort and greater confidence.

## Immediate starting point

Begin with N01 through N03. The first demonstration is a real command that loads a pinned model, records a baseline, applies an identity operation, verifies parity, and produces a bundle that can be replayed. Add the known fixture repair and a real-model intervention next. This establishes the technical foundation before committing to a broad UI, compatibility matrix, or claims of automatic improvement.
