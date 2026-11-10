import type { McpSession } from "@jmoak/vibrissa";
import {
  NrlyClient, keyworkTransport, vibrissaTransport,
  type ExperimentSpec, type KeyworkConnection, type StudyAssessment,
} from "../src/index.js";

declare const session: McpSession;
declare const keywork: KeyworkConnection;
const clients: NrlyClient[] = [
  new NrlyClient(vibrissaTransport(session)),
  new NrlyClient(keyworkTransport(keywork)),
];

const spec: ExperimentSpec = {
  name: "Consumer contract",
  program: {
    kind: "pipeline", expected_base: "a".repeat(64),
    operations: [{ kind: "weight_low_rank", target: "output.weight", shape: [1, 2], rank: 1, left: [1], right: [0.5, 0.5] }],
  },
  cases: [{ id: "sum", input: "[2,4]", expected: "6", scorer: "numeric" }],
  weight_reads: [{ target: "output.weight", coordinates: [[0, 1]], max_elements: 1 }],
};

async function consume(client: NrlyClient): Promise<StudyAssessment | null> {
  const run = await client.submit(spec, "consumer-idempotency");
  await run.wait({ requireSuccess: true });
  for await (const page of run.sampleChangePages("sum")) page.weight_changes[0]?.net_changed_elements;
  const study = client.study("operator-provisioned-study");
  const attempt = await study.submit({ name: "Candidate", program: spec.program! });
  return (await attempt.wait()).assessment;
}

void clients;
void consume;

// @ts-expect-error Arbitrary callbacks are not declarative tensor programs.
const unsupported: ExperimentSpec = { name: "invalid", program: { kind: "javascript" } };
void unsupported;
