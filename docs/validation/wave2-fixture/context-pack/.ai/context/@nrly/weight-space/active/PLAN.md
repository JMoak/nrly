# nrly study handoff

Study: `80462b98676146f1a9d15c0930a5496a`
Snapshot status: `completed`
Final verdict: `observed_improvement`
Recommendation: `candidate_supported`

The frozen candidate has supporting evidence on the evaluated cases. Review scope and controls before any separate decision to use it.

The current objective is to evaluate bounded weight edits against registered
controls while preserving regression behavior. Consult the public evidence file
for the model fingerprint, target interface, admitted budget, and every attempt.

Read `.ai/modules/@nrly/weight-space/knowledge/public-evidence.json` for data and
`.ai/modules/@nrly/weight-space/rules.md` for the scoped workflow notes.

Possible next actions, selected by the operator or existing agent permissions:

```text
nrly --root <NRLY_ROOT> study show 80462b98676146f1a9d15c0930a5496a
nrly --root <NRLY_ROOT> study attempts 80462b98676146f1a9d15c0930a5496a --offset 0 --limit 20
```

When the study is open, an operator may review a candidate file and use
`nrly --root <NRLY_ROOT> study submit 80462b98676146f1a9d15c0930a5496a <CANDIDATE_JSON>`.
After all admitted attempts finish, nominate exactly one completed attempt with
`nrly --root <NRLY_ROOT> study finalize 80462b98676146f1a9d15c0930a5496a <ATTEMPT_ID>`.
These actions are unavailable after the nominee is frozen. If an MCP server owns
the workspace, use `get_study`, `list_study_attempts`, `submit_study_candidate`, and
`finalize_study` through that existing server instead of opening another runner.
Use `cancel_study_final` to stop an active final job; cancellation retains its freeze.

This is an evidence handoff, not a replay bundle or executable patch. All admitted
attempts are summarized; tensor payloads, case text, raw outputs, logs and the raw
ledger are omitted. The source store remains the authority. Refresh the snapshot
after new activity. Budget values are admitted ceilings, including reserved final
evaluation, rather than actual compute consumption.
