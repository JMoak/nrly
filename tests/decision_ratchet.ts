// Optional real sibling integration. No Ratchet files or user configuration are changed.
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
const root = process.env.NRLY_RATCHET_REPO!;
const { SystemOneOracle } = await import(pathToFileURL(resolve(root, "src/oracles/systemone.ts")).href);
const oracle = new SystemOneOracle({
  baseUrl: process.env.NRLY_TEST_ENDPOINT,
  apiKey: process.env.NRLY_TEST_TOKEN,
  model: process.env.NRLY_TEST_MODEL,
  id: "nrly-local-shadow",
  pricePerMillionInput: 0,
  maxRetries: 0,
});
const [decision] = await oracle.decide(process.env.NRLY_TEST_STATE!, [{
  kind: "judge", id: "support", prompt: process.env.NRLY_TEST_QUESTION!,
}]);
if (decision.model !== process.env.NRLY_TEST_MODEL || decision.cost !== 0 ||
    decision.verdict.kind !== "judge" || !Number.isFinite(decision.verdict.pYes)) {
  throw new Error("Local SystemOne provenance, cost or verdict failed conformance");
}
console.log(JSON.stringify({model: decision.model, probability: decision.verdict.pYes,
  cost: decision.cost, oracle: decision.oracle}));
