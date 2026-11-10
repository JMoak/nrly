# CorePack Unified Context Protocol assessment

Reviewed 10 November 2026. The user identified **CorePack-AI's Unified Context Protocol**.
The useful integration is a portable, optional study handoff in its filesystem layout.
nrly continues to use MCP for execution and status operations.

## What the primary source establishes

CorePack's README labels the proposal version **1.0**, January 2026. It organizes context
under `.ai/`, with scoped module packs containing `rules.md`, `generators/`, and `knowledge/`,
plus scoped current plans and archives. Its proposed lifecycle includes agent discovery
of those files. The concrete tree scopes plans by organization and pack; a later paragraph
abbreviates that path. nrly follows the explicit scoped tree. These are documented conventions,
not evidence that every agent implements discovery.
[Pinned proposal](https://github.com/CorePack-AI/ucp-spec/blob/c8c637e554ed86cef6acaa2827fd839d68661935/README.md).

The inspected revision is `c8c637e554ed86cef6acaa2827fd839d68661935`, committed
9 January 2026; README blob `e096dabf213824f63020c5c4774916d258949277`. Its complete tree
contains a README, license, and image assets. It supplies no machine-readable schema,
reference loader, or executable conformance suite. Therefore nrly claims compatibility
with the documented layout, not certified interoperability or broad adoption.
[Commit](https://github.com/CorePack-AI/ucp-spec/commit/c8c637e554ed86cef6acaa2827fd839d68661935),
[repository at that revision](https://github.com/CorePack-AI/ucp-spec/tree/c8c637e554ed86cef6acaa2827fd839d68661935).
The upstream repository is MIT licensed; nrly's exported content is independently authored.
[License](https://github.com/CorePack-AI/ucp-spec/blob/c8c637e554ed86cef6acaa2827fd839d68661935/LICENSE).

## Fit with the existing stack

The practical benefit is keeping a study's objective, immutable identities, unsuccessful
attempts, final verdict, and next valid operations together when handing work to another
agent or person. This is a lightweight organizational choice. It does not improve inference,
establish a result's scientific validity, or enforce the instructions by itself.

Local source review found existing stdio and HTTP MCP support in Keywork's
`packages/engine/src/mcp/registry.ts` and `http.ts`, including static headers and OAuth
integration. Vibrissa already supports HTTP/stdio sessions and contract testing in
`src/session/`. Neither inspected MCP integration automatically loads this `.ai` convention.
Explicitly reading the exported pack is sufficient; a new transport or sibling-repository
dependency would add no demonstrated benefit. Relly already has permission-checked evidence
packs in `docs/agent-api.md`; this export can accompany that workflow without replacing it.
All sibling repositories were read-only during this assessment.

## Implemented optional export

```text
nrly --root <NRLY_ROOT> study export-context STUDY_ID DESTINATION

DESTINATION/
  README.md
  nrly-context-manifest.json
  .ai/modules/@nrly/weight-space/
    rules.md
    generators/identity-candidate.json
    knowledge/public-evidence.json
  .ai/context/@nrly/weight-space/
    active/PLAN.md
    archive/
```

The exporter reads a consistent SQLite snapshot without acquiring the model-runner lease.
The evidence JSON summarizes all admitted attempts, with explicit omissions; it includes
the model fingerprint, allowed targets, control digests, budget ceilings, holdout commitment,
and aggregate final result. It excludes all case text, expectations, raw outputs, private run
handles, tensor payloads, and worker logs. A context pack is deliberately not a replay bundle.

The plan names the actual study and valid follow-up commands. It recommends retaining the
original until final evidence supports a candidate, and preserves the limits of fixture or
real-model evidence. Study/candidate names remain JSON data, so they cannot become generated
Markdown instructions. The identity generator is a validated nrly candidate template.

The manifest is explicitly **nrly's own format**: it pins the upstream revision and records
file hashes. It is neither an upstream schema nor a signature. Exports are bounded to 2 MiB,
built in an owned temporary directory, and published by rename. Existing nonempty destinations
are refused. No project instructions, user configuration, dependencies, MCP tools, or model
weights change. Clients must load the pack explicitly; automatic startup discovery is not
claimed. Sealed-case protection remains the application/API boundary, not protection from
the filesystem owner.

Focused tests exercise the layout, hashes, valid template, missing-secret guarantees, large
candidate-payload omission, readonly export, admission-count integrity, size bound,
non-overwrite behavior, and cleanup after a simulated write failure.

## Acronyms kept separate

Universal **Commerce** Protocol is a different project, whose latest release observed in
this review was `v2026-08-25`; no shopping integration was added.
[Commerce release](https://github.com/Universal-Commerce-Protocol/ucp/releases/tag/v2026-08-25).
Universal **Context Package** from ucpcore is also different: its JSON specification is
draft 0.1.1 and complements MCP. Its schema, libraries, and receipt terminology are not used
by this CorePack export.
[Context Package specification](https://github.com/ucpcore/ucp/blob/89c91785f6337ca468158c5e89412f26bc31d8b2/SPEC.md).
