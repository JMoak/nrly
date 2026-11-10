# MCP contract checks

These checks use the published `@jmoak/vibrissa@2.0.0` development dependency and the real
nrly Python stdio server. No sibling repository, live model download, inference API key,
or user's experiment store is used. The lifecycle check requires the Torch dependency;
a missing dependency fails the test rather than substituting fabricated results.

From the repository root, with the Python environment installed:

```sh
bun install --frozen-lockfile
bun run test:mcp
bun run mcp:manifest:check
```

`run.mjs` creates a temporary store and supplies its path and the repository's Python
executable to `vibrissa.json`. Run through the package scripts so these environment values
exist. Cleanup verifies the resolved temporary path before removing it.

JSON cases cover discoverability, malformed experiments, shape errors, unsupported backends,
and invalid handles. The Node lifecycle test uses Vibrissa's public `McpSession` API to
poll nrly jobs. It verifies a direct weight edit, exact before/after parameter evidence,
idempotency, fixture repair, regression preservation, independent scoring and restoration
of base behavior. Engineering-fixture evidence must remain explicitly labeled.

After an intentional API change, run `bun run mcp:manifest:record` and review the manifest
diff. The normal suite requires the checked-in manifest and fails on contract drift.

## Wave 2 client coverage

`sdk.test.mjs` exercises the dependency-free TypeScript client through the actual Python
MCP server: rich parameter selections, pinned multi-operation edits, bounded sample
enumeration, idempotency, and cancellation. `studies.test.mjs` provisions a study through
the operator Python API, then exercises candidate development and sealed finalization
through MCP while asserting that public responses contain no holdout canaries or private
run identifiers. Both use the published `@jmoak/vibrissa@2.0.0` session API.

`npm run test:mcp` builds the client before running all MCP tests sequentially. The
separate `npm run test:keywork` acceptance test requires `KEYWORK_REPO` and uses that
checkout's actual `connectStdioServer` and configuration schema without modifying it.
See `packages/client/README.md` for usage and limits.

