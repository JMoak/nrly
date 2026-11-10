# Weight-space programming: corrected research audit

Reviewed 10 November 2026. This supersedes the earlier framing of nrly as primarily a latent-space programming environment. Sources below were reopened or fetched during this pass. Published results are attributed claims; we have not reproduced their full experiments.

## What changed

**nrly is a workbench for weight-space programming:** inspect a pinned network's parameters, author a direct change, execute the resulting network, and preserve the change with evidence. Its kernel does the actual parameter writing. The investigating agent can be Keywork or another MCP client; the investigator and the network being edited are separate systems.

| Earlier direction | Correct treatment now |
| --- | --- |
| Latent representations as the primary programmable object | Parameters and explicit weight programs are the primary objects. Hidden representations are observations of execution. |
| Activation steering as the main demonstration | A supporting diagnostic/control. It does not establish persistent weight programming. |
| Learned prefixes, Cartridges, MEMORILLA, latent memory | Adjacent techniques, outside this MVP and outside the evidence for direct weight programming. |
| Stable neuron locations across Qwen generations | An unproven research question, not an implementation assumption. Targets are exact names, shapes, and coordinates inside one pinned model. |
| A generic experiment dashboard as differentiation | Insufficient by itself. Existing tools already address substantial parts of the proposed workflow. |
| A successful edit as the expected result | Unchanged, harmful, invalid, and inconclusive outcomes must remain visible. |

The terminology correction changes the roadmap: parameter inspection, sparse writes, exact change records and replay precede learned memory, cross-checkpoint transfer and a community gallery.

## Direct evidence and its limits

**Originating report, 2 November.** Takehi, Teoh, Zhang and Khattab define WSP as a general-purpose language model reading/writing weights directly around a natural-language objective, without an optimizer. Their experiments span identifying small networks, repairing policies, constructing networks and editing Qwen3.5-0.8B. Information access varies substantially: weights-only tasks, Python analysis without forward passes, and budgeted execution are distinct conditions. nrly should preserve those distinctions rather than collapse them into “the agent understood the neurons.” [Report](https://rikiyat.github.io/blog/weight-space-programming/).

**Released implementation.** The authors' README documents an external Python harness for some experiments even though built-in agent tools were disabled. It releases reported runs rather than every earlier prompt trial; intermediate artifacts have omissions. Experiment 8 includes final weight differences and an application script. Our inference: attempt history, permitted observations, executable modifications and evaluation settings are necessary parts of a reusable research artifact. [Author repository](https://github.com/RikiyaT/lm_understand-logs/blob/main/README.md), fetched README blob `63404d03e357ad223fe971740919e221d6297215`.

**WSP-Scope, 6 November.** Sixteen editing trajectories across seven benchmarks and Qwen3.5 at two scales report localized gains and uneven transfer. Six trajectories retain the original weights because candidate improvements cannot be reliably validated. This supports sparse editing and explicit retention of the baseline; it does not establish reliable improvement for arbitrary tasks or models. [Paper](https://arxiv.org/abs/2611.09382), [project](https://wsp-scope.github.io/).

**Robot policies, 6 November.** Zhang, Wang and Lin report lexical grounding repairs through 65 embedding rows. Their strongest held-out paraphrase result rises from 86.3% to 92.0%, with 48 repairs and four regressions. Twelve of thirteen further candidates are declined, and the remaining edit has no significant held-out gain. The supported opportunity is targeted repair with locality checks, not creating missing robot capabilities. [Paper](https://arxiv.org/abs/2611.09376).

**Skeptical weight-reading audit, 5 November.** Ahmed reports useful local arithmetic alongside incorrect semantic interpretations on small CNNs. Label permutation exposes reliance on familiar label order. Mostly single runs and the absence of direct forward-pass verification limit the conclusions. It nevertheless gives nrly concrete controls: blind labels, permuted coordinates, fresh examples, and separate scoring of an explanation and an effective edit. [Paper](https://arxiv.org/abs/2611.08070).

**Written scoring networks, 6 November.** Butkeraites studies LLM-written networks inside exact optimization algorithms. Some runtime improvements are matched by an inexpensive conventional rule, and performance varies with problem distribution. A comparison against a weak baseline can overstate the contribution of weight reasoning. Our experiment designs should include cheap analytic rules, matched random edits and ordinary inference settings when relevant. [Paper](https://arxiv.org/abs/2611.09559).

**WeightLoop, 5 November.** Li and colleagues describe a fixed editor authoring a reusable parameter-editing program that consumes later experimental measurements. Their offline replay reports gains across several sequential prediction tasks. The interesting extension is a program that keeps applying a bounded update rule as evidence arrives, rather than requiring another editor call for every change. This remains distinct from an editor recursively improving itself. nrly's current declarative operations are a smaller primitive; adaptive editing programs and their evaluation would need a new execution contract. [Paper](https://arxiv.org/abs/2611.07319).

**Image restoration proposal, 3 November.** Zheng and Doubao propose diagnosing and editing diffusion models for restoration quality. The manuscript explicitly identifies its claims as conjectures. It is a source of experimental ideas, not evidence that those improvements have been demonstrated. [Proposal](https://arxiv.org/abs/2611.04268).

The supplied [YouTube URL](https://www.youtube.com/watch?v=dOkKO4SoeRQ) did not yield a usable page or transcript through the available browser research tool during this pass. No new claim here depends on an unverified interpretation of its narration; the audit uses the originating report, released artifacts and direct follow-up sources.

## Existing projects change the product claim

**OpenWSP is a direct overlap.** Its documented core includes a Python weight kernel, forward service, versioned edits, CLI, SDK and MCP. It also reports live editing attempts separately from calibration of the authors' released edit. Its README's introductory “pending” language and later result tables are not fully synchronized, so this audit does not use it to assert the current completion status of every benchmark. Its existence is enough to reject any claim that nrly originates the general workbench architecture. [Repository README](https://github.com/openwsp/openwsp/blob/main/README.md), fetched blob `3e24909b9cf2fa56a5b8f910c1125b0d3508a27a`.

**WSP-Lang is another substantial neighbor.** Its project page describes quantized-model inspection, editing, MCP tooling and extracting a donor model's change into an adapter. The page also explicitly withdraws earlier behavior claims involving synthetic scoring or hardcoded outputs. Its current transfer results are author-reported and limited to stated conditions. This makes source-backed provenance and honest distinctions between training, transfer, steering and direct writing practical requirements for nrly. [Project and result-status notes](https://fivetechsoft.github.io/weights/).

The [Awesome Weight-Space Programming collection](https://github.com/brighterr/awesome-weight-space-programming) is useful for discovery, but its summaries are not independent experimental verification. It separates direct work, compilers, learned weight generation and model editing; that taxonomy is preferable to using “latent space” as an umbrella. NNsight, TransformerLens and pyvene remain relevant implementation or interoperability references for internal observations; they are not interchangeable with direct weight programming.

**Differentiation remains a hypothesis:** an accessible local tool that makes exact edits easy to inspect, reliably executes them on ordinary developer hardware, exposes a small agent API, and fits Keywork plus Vibrissa without coupling the kernel to those apps. Validate that with users and compare against OpenWSP before investing in a hosted platform. Import/export interoperability may be more valuable than duplicating a research harness.

## What this pass caused us to implement

1. An inspection job that catalogs exact tensor names/shapes and reads bounded parameter values without forward passes.
2. Direct weight scaling, full replacement and sparse writes, with exact shape and base-identity checks.
3. Before/after tensor hashes, changed-element counts and selected edited values; restoration verified against the complete original model state.
4. Distinct activation diagnostics and identity controls; activation changes never produce weight-change evidence.
5. Pinned model/runtime identities, unchanged generation settings, target/regression cases, immutable completed records and verified export/replay bundles.
6. Explicit failure, unscored, no-change, regression and mixed comparison outcomes. An infrastructure success is not a scientific improvement.

The shipped deterministic repair is an engineering fixture. The Qwen3-0.6B smoke experiment validates execution and replay on a real model; it is not a reproduction of the original Qwen3.5-0.8B reasoning-loop result.

## Research directions worth building on this foundation

**A weight debugger with testable explanations.** An agent predicts which parameter slice controls a failure, proposes a patch, and must beat matched random/analytic controls on new examples while preserving a regression set. The explanation and the behavioral result receive separate verdicts. A useful contribution would be fewer failed experiments or better diagnosis per unit of compute, not persuasive descriptions of neurons.

**An experiment protocol for reasoning about weights.** Compare weight-only inspection, analysis with no forward passes, and budgeted instrumented feedback. Record every candidate and what information the editor saw. Add label-blind/permutation controls to distinguish parameter reasoning from architectural or dataset priors. The current inspection mode supplies a zero-forward primitive; it does not yet enforce information isolation across an entire agent session.

**An interoperable patch review and replication workflow.** Share a compact program, exact model requirements, checksums, budgets, controls, failures and held-out outcomes. Independent reruns can qualify or reject a result. A community space becomes useful when it is backed by those artifacts. Current local bundles are a foundation, not a hosted registry or a signed proof of scientific truth.

Next scientific milestone: a preregistered, modest real-model repair study with a discovery set, sealed holdout, identity and placebo controls, and a reproducible report including failed attempts. The implemented study API now separates held-out cases from author-facing operations; it does not protect them from a filesystem owner. Merely labeling an ordinary run case `test` does not seal it. See the [current implementation](wave3.md) for diagnosis, paired-seed comparisons and reusable edits.
