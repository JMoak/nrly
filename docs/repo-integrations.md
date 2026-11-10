# Repository integrations

nrly owns weight-space programming, model execution, persistence and evaluation. Activation
operations support diagnosis and are recorded separately from direct weight edits. Existing repositories
connect through small contracts; none is a required Python runtime dependency. This review
inspected actual code and local git origins on 2026-11-10. Existing repositories were read-only.

| Priority | Integration | Readiness | Coupling |
| --- | --- | --- | --- |
| 1 | Vibrissa MCP contract tests | Published `@jmoak/vibrissa@2.0.0`, registry verified | Development dependency only |
| 2 | Keywork orchestration | Actual connector acceptance + typed nrly client | Protocol and optional user configuration |
| 3 | Jerbs/Tendie evidence invariants | Implemented evaluation and artifact verification patterns | Design reuse, no dependency |
| 4 | Plaia evaluator adapter | Working benchmark runner; explicit adapter still needed | Optional evaluation integration |
| 5 | Relly research evidence | Working local HTTP API | Optional research adapter |

## Keywork: configuration, without provider changes

Source: `C:/src/keywork/packages/engine/src/mcp/registry.ts`,
`packages/shared/src/config/schema.ts`, `packages/cli/src/user-config.ts` and `docs/headless.md`.
The existing client supports stdio and HTTP MCP; nrly currently serves stdio. Default request timeout is 10 seconds; results are
bounded at 30,000 characters. Submit work quickly, poll its handle, and retrieve bounded
reports instead of holding a model run inside a tool call.

The user-layer file is `~/.keywork/keywork.json`. Project-layer `mcpServers` is deliberately
ignored, even for a trusted workspace (`packages/shared/src/config/load.test.ts:395`).
The stdio schema has `command`, `args`, `env` and `trusted`; it has no `cwd` field.

After installing nrly in its environment, this is the integration shape. Choose the store
path intentionally. Do not overwrite an existing user configuration with this fragment.

```json
{
  "mcpServers": {
    "nrly": {
      "transport": "stdio",
      "command": "C:/src/nrly/.venv/Scripts/python.exe",
      "args": ["-m", "nrly.mcp", "--root", "C:/src/nrly/.nrly"]
    }
  }
}
```

An isolated Windows configuration check avoids reading or changing the real user file:

```powershell
$nrlyProfile = Join-Path ([IO.Path]::GetTempPath()) ("nrly-keywork-" + [guid]::NewGuid())
$nrlyConfigDir = Join-Path $nrlyProfile '.keywork'
New-Item -ItemType Directory -Path $nrlyConfigDir | Out-Null
$nrlyConfig = @{
  mcpServers = @{ nrly = @{
    transport = 'stdio'
    command = 'C:/src/nrly/.venv/Scripts/python.exe'
    args = @('-m', 'nrly.mcp', '--root', (Join-Path $nrlyProfile 'runs'))
  } }
}
$nrlyConfig | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $nrlyConfigDir 'keywork.json')
$nrlyPreviousProfile = $env:USERPROFILE
try {
  $env:USERPROFILE = $nrlyProfile
  Push-Location -LiteralPath $nrlyProfile
  try { bun C:/src/keywork/packages/cli/src/main.ts mcp list } finally { Pop-Location }
} finally {
  $env:USERPROFILE = $nrlyPreviousProfile
}
```

This lists configuration; it does not certify transport or inference. The Vibrissa checks
below exercise transport. A full Keywork run also needs explicitly configured inference
and tool permissions in the isolated profile. `keywork run '<task>' --json` emits tool
events, but exit zero alone is not scientific evidence. No provider interface changes or
blanket trust are needed.

This isolated check was executed successfully: the existing CLI reported `nrly · stdio`.
The temporary profile was retained for inspection; the actual user configuration was untouched.

Wave 2 adds a stronger acceptance check: `bun run test:keywork` imports the real
`connectStdioServer` and configuration schema from `C:/src/keywork`, connects to nrly,
and exercises real inference, bounded evidence, idempotency and conflicting retries.
It closes the connection and never writes the real user configuration. This check needs
the sibling source checkout; it is separate from portable package tests. It does not
call a paid author model or validate a complete autonomous Keywork agent session.

The local [TypeScript SDK](../packages/client/README.md) exposes `NrlyClient`,
`keyworkTransport`, `vibrissaTransport` and a generic MCP adapter. Its configuration
helpers return fragments; the caller owns server lifecycle and deliberate registration.

## Vibrissa: the public protocol is an executable contract

Published version verified with `npm view @jmoak/vibrissa version dist-tags dist.integrity
--json`: `latest` was `2.0.0`, integrity
`sha512-Jmm6g1xZcOiLh5YRi9IR/wfs5cZQIsp6a381UO5F25EM/MgbUpupfl8CH9LAJCksLr0UMasNJQCeZCBQxUvOXw==`.
nrly pins that published development dependency, with no sibling-repository dependency.
Inspected documentation: `C:/src/vibrissa/docs/{CLI,CONFIG,CASES,MATCHERS}.md`.

From nrly, after the Python environment exists:

```powershell
bun install --frozen-lockfile
bun run test:mcp
bun run mcp:manifest:check
```

The launcher runs the actual `.venv/Scripts/python.exe -m nrly.mcp --root <temporary-dir>`
on Windows (`.venv/bin/python` on Unix). Each server gets a fresh store; tests never use
the user's experiment store. Stdio passes through unchanged. The asynchronous acceptance
test uses Vibrissa's public `McpSession` API to poll nrly jobs; MCP task polling and nrly's
custom job IDs are different mechanisms.

Update the manifest intentionally with `bun run mcp:manifest:record`, then inspect its diff.
IDs/timestamps use structural matchers. The deterministic Torch fixture establishes
engineering behavior, not general model quality.

## Jerbs/Tendie: reuse evidence invariants

Actual implementations: `C:/src/jerbs/internal/tendie/evaluate.go`,
`C:/src/jerbs/tools/smoke_artifacts.py`, and `C:/src/jerbs/docs/TENDIE.md`.

- Identify the actual evaluator and loaded model, not just caller-supplied names.
- Hash the program, corpus and individual cases; preserve exact evaluation inputs.
- Separate completed execution from supported claims. Errors, missing cases, partial results
  and invalid outputs are inconclusive or failed, never passes.
- Keep negative controls that must fail qualification.
- Validate artifact paths remain inside their root and verify size/hash on read.

Adapt the patterns without adding Go/MySQL. Tendie's current in-process execution cannot
terminate arbitrary hung candidate code; evidence checks do not establish containment.

## Plaia and Relly: explicit adapters

`C:/src/plaia/plaia/bench/runner.py` already collects task/model/repeat, score, duration,
tokens and fallback metrics. Its `WORKFLOW_MODULES` registry is hard-coded; `load_suite`
rejects unknown workflows. nrly is not already supported. Keep nrly's report independent,
then add an explicit adapter. The runner temporarily changes process environment, so
concurrent model configurations need separate processes. `bench/metrics.py` does not count
unknown provenance as fallback: strict evaluation must reject unknown evidence instead of
assuming `no_fallback` proves genuine execution.

`C:/src/relly/docs/agent-api.md` documents existing local endpoints: `POST /api/import`,
`POST /api/run`, `GET /api/pack?sessionId=...`, `GET /api/export`, and
`GET /api/omissions?sessionId=...`. An optional adapter can attach versioned research
evidence and conflicting findings to hypotheses. Relly is a local single-user service,
not an authenticated hosted community backend or a replacement for eval datasets.

## Deliberate boundaries

- `ts-mcp-template`: useful examples; no reason to force Python nrly into TypeScript.
- `tdd-dsl`: useful when multiple languages actually implement the same semantic contract.
- Ratchet: later admission policy; policy gates cannot certify numerical scientific claims.
  Avoid duplicating Keywork's existing permission machinery.
- `containerized-coding-agents/AGENTS.md` explicitly says **"Nothing here has been booted."**
  Its design is not a validated nrly sandbox or evidence of Windows/GPU isolation.
- Orchard: contribution/governance patterns, not an end-user runtime dependency.

Verified local origins: `github.com/JMoak/{keywork,vibrissa,plaia,relly,jerbs}`. Only Vibrissa's
package publication was independently verified; no hosted readiness is implied for others.
