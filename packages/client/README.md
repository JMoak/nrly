# nrly TypeScript client

The local `@nrly/client` package connects Keywork, Stead, or another TypeScript caller to
nrly's application protocol. It has no runtime dependencies and performs no tensor math.
Transport ownership stays with the caller: the client never starts or closes a server.
This package is private and has not been published.

Build from the nrly repository:

```powershell
bun install --frozen-lockfile
npm run build:client
npm run test:client
```

Import the built `packages/client/dist/index.js`, or make the local package available in
your application. `mcpTransport` accepts a `callTool(name, arguments)` adapter returning
an MCP result envelope; `vibrissaTransport` accepts Vibrissa 2's public `McpSession`;
`keyworkTransport` accepts Keywork's existing `McpConnection`.

```ts
import { NrlyClient, keyworkTransport } from "./packages/client/dist/index.js";

// connection is an existing Keywork McpConnection. Its owner closes it.
const nrly = new NrlyClient(keyworkTransport(connection));
const inspection = await nrly.inspect({
  model: { backend: "tiny" },
  weight_reads: [{ target: "output.weight", coordinates: [[0, 1]], max_elements: 1 }],
});
await inspection.wait({ timeoutMs: 45_000, requireSuccess: true });
const weights = await inspection.inspection({ weight_target: "output.weight" });

const run = await nrly.submit({
  name: "Pinned weight repair",
  program: {
    kind: "pipeline",
    expected_base: weights.model_identity!.fingerprint,
    operations: [{ kind: "weight_scale", target: "output.weight", factor: 2 }],
  },
  cases: [
    { id: "repair", input: "[2,4]", expected: "6", scorer: "numeric" },
    { id: "preserve", input: "[0,0]", expected: "0", scorer: "numeric", role: "regression" },
  ],
}, "caller-owned-retry-key");
const completed = await run.wait({ timeoutMs: 45_000 });
if (completed.status === "succeeded") {
  for await (const page of run.samplePages({ limit: 5 })) {
    for (const summary of page.samples) console.log(await run.sample(summary.case_id));
  }
}
console.log(run.artifact); // nrly://runs/<id>, an application handle, not an HTTP download
```

`succeeded` means execution completed. Assessment reports establish the observed result;
the tiny backend always has engineering-fixture scope. `requireSuccess` rejects failed,
cancelled, or interrupted execution, but never substitutes for evaluation.

## Polling, cancellation, and bounds

- `wait()` defaults to a 120-second caller deadline and 250 ms polling. Each request has
  its own 10-second default deadline. Deadlines include an in-flight request.
- Timing out or aborting a wait leaves the job running. Use `cancelOnTimeout: true` or
  `cancelOnAbort: true` to request cancellation, or call the handle's `cancel()` explicitly.
  A timeout still rejects even when cancellation succeeds. A cancellation failure is kept
  in the error cause. An adapter that cannot abort its transport may finish its in-flight
  request after the caller stops waiting.
- No automatic retry of submission occurs. Supply a stable request ID to retry safely.
  A transport timeout after submission can leave an accepted job; retry with the same ID.
- `inspectionPages`, `weightPages`, `samplePages`, `sampleOutputPages`, and
  `sampleChangePages` fetch bounded pages and stop
  after at most 100 pages by default. `maxPages` is explicit; on a page-limit error resume
  from the last returned cursor. Early exit from an iterator stops further requests.
  Run inspection pages enumerate tensors; `weightPages(target)` enumerates the captured
  values for one tensor with aligned indices/coordinates. `sampleChangePages(caseId)`
  enumerates pipeline operation evidence, four changes per response by default. Final
  `net_changed_elements` fields avoid double-counting repeated edits to the same tensor.
- `getRun` can be a summary when a full run exceeds the 12,000-character response budget.
  Use `samplePages` to enumerate case IDs and `sample` for details. Unicode output offsets
  use Python code points, including when an output contains emoji.
- `NrlyClientError.kind` distinguishes `domain`, `protocol`, `transport`, `timeout`,
  `aborted`, and `execution`; `code` preserves nrly domain errors such as
  `request_conflict`. Identity, status, cursor, and response-size checks reject malformed
  responses; full server validation remains authoritative for specifications.

## Controlled studies

The operator provisions a study and sealed holdout through the Python API or CLI. There
is intentionally no MCP study-creation method. The client reads only public development
cases/results and counts-based final assessments; private study arm run IDs and raw
holdout data are not exposed.

```ts
const study = nrly.study(operatorSuppliedStudyId);
for await (const page of study.casePages({ limit: 2 })) console.log(page.cases);
const attempt = await study.submit({
  name: "Candidate A",
  program: { kind: "weight_scale", target: "output.weight", factor: 2 },
}, "candidate-a");
const development = await attempt.wait({ requireSuccess: true });
console.log(development.assessment);
for await (const page of attempt.samplePages("candidate")) console.log(page.samples);

// This is the deliberate final nomination. The sealed final evaluation is single-use.
await study.finalize(attempt.id);
const final = await study.waitFinal({ timeoutMs: 120_000 });
console.log(final.final_result?.recommendation);
```

`study.attemptPages()` enumerates retained attempts, including unsuccessful ones.
`study.targetPages()` enumerates the complete preregistered mutable interface, including
targets beyond the first 16. `attempt.sampleOutputPages("candidate", caseOffset)`
reconstructs one development answer using bounded Unicode-aware output pages. Case
inputs and expectations have separate `input_offset`/`input_limit` options on `cases()`;
inspect `more_input` before treating a returned case as complete.
`attempt.cancel()` cancels development execution; `study.cancelFinal()` cancels the final
phase. `waitFinal({requireSuccess: true})` requires a completed evaluation with an
assessment and no lifecycle error. A completed phase can recommend retaining the
original model or report an inconclusive scientific comparison. Finalization never
means scientific success by itself.

## Diagnostics and complete study evidence

Experiment schema 0.3 accepts explicit `messages`, continuation `diagnostics`, selected
activation captures and configurable sampling. Submit through the same `submit` method.

```ts
const run = client.run("RUN_ID");
for await (const page of run.diagnosticPages("CASE_ID", { limit: 4, top_k_limit: 3 })) {
  console.log(page.tokens); // supplied-continuation probabilities, not task accuracy
}
const capture = await run.capture("CASE_ID", "model.layers.0.mlp.down_proj");
const completeCase = await study.caseSpec("CASE_ID");
const comparator = await study.comparator("format-prompt");
const assessment = await attempt.assessment();
const finalAssessment = await study.finalAssessment();
```

`caseSpec`, `comparator`, and assessment methods reconstruct bounded JSON pages and
verify SHA256 over the actual UTF-8 content. A changed or corrupted page fails rather
than returning a partial record. `maxPages` bounds traversal. Only development case
specifications and public aggregate assessments are accessible. Case summaries omit
potentially large chat and adapter payloads; retrieve `caseSpec` for exact constraints.

For a study with explicit seeds, sample arm names include the seed, for example
`candidate@seed=17` or `comparator:format-prompt@seed=17`. The study summary lists
registered seeds and comparator digests; comparator details include measured model
identity. Descriptive variation across seeds is not a population confidence interval.

## Keywork configuration and acceptance

`keyworkConfig({python, root, modelCache?})` returns a fragment for the **user** layer
`mcpServers` configuration. `keyworkServer(...)` returns just the stdio server entry.
All paths must be absolute because Keywork's schema has no `cwd`. Neither helper writes
configuration or opts the server into trust. Merge the fragment deliberately into the
user configuration; project-layer `mcpServers` is ignored by Keywork.

The optional test uses the actual existing Keywork source and Python server, validates
the fragment through Keywork's actual schema, and exercises submission, polling,
pagination, retry identity, and domain errors. It creates an isolated temporary nrly
store and never reads or changes the user's Keywork configuration:

```powershell
$env:KEYWORK_REPO = 'C:/src/keywork'
npm run test:keywork
```

The ordinary `npm run test:mcp` suite depends only on the published, exact-pinned
Vibrissa package and the nrly environment. It checks both the run and study SDK against
actual MCP input schemas, rich selector/pipeline evidence, cancellation, bounded result
retrieval, and non-disclosure of sealed holdout values.

Input types are generated from Pydantic, including pipeline and selector contracts:

```powershell
.venv/Scripts/python.exe scripts/generate_client_contracts.py
.venv/Scripts/python.exe scripts/generate_client_contracts.py --check
npm run check:client
```

The checked-in generated types are a development contract, not an independently
versioned replacement for the server schemas. Regenerate them when Python contracts
change. No sibling repository is a runtime dependency.
