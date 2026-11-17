# Decision pilot acceptance — 2026-11-17

Real local execution on the existing pinned Qwen3-0.6B checkpoint
`c1899de289a04d12100db370d81485cdf75e47ca`, BF16 eager attention on the local RTX 5090.
The source/claim examples in `scripts/validate_decision_qwen.py` are public and assistant-authored.
They are engineering smoke material, not unseen application reliability evidence.

- Twelve cases: four fit, four evaluation, four challenge; one distinct source family per case.
- Twenty-four original/patched observations, each with two supplied-label likelihood forwards.
- Candidate: set the first sixteen flattened entries of layer 0 MLP `down_proj.weight` to zero.
  This is a diagnostic canary, not a discovered useful repair or the published Qwen3.5 edit.
- Both rules fit independently to threshold `0.99`. Four-case evaluation coverage was 25% for
  both; challenge coverage was 0% for both. The fixed `0.5` reference accepted 50% in each split.
- No unsupported acts were observed in this tiny authored set. This does not establish a
  population error bound; threshold fitting reduced coverage and did not demonstrate a benefit.
- Recommendation: **retain original**. Zero rescued, introduced-unsupported or lost supported acts.
- The actual sibling Ratchet `SystemOneOracle` called the authenticated local bridge and
  returned the exact served identity and probability with zero API billing. This tests the
  adapter seam, not a full Ratchet gate, policy or promotion flow.
- Original model tensor digest matched after execution. Nonzero parameter changes were recorded.

The adjacent aggregate assessment and conformance receipt contain the exact run/freeze digests
and tested Ratchet source hash. Full inputs, observations and per-forward weight evidence stay
in the local ignored run directory. Reproduce with the acceptance script and a new destination.
Timing includes hashing/restoration checks and should not be read as a serving benchmark.

## Engineering verification

- 399 Python tests passed, including 22 new decision tests: beneficial/harmful native Torch
  edits, restoration, threshold-only gains, leakage rejection, malformed probabilities,
  final-call budget overruns, artifact tampering, report redaction and real Ratchet HTTP calls.
- 22 TypeScript client tests, 15 MCP contract cases and 4 MCP integration tests passed.
- Ruff, strict mypy, 25 generated-schema checks, client-contract drift and publication checks passed.
- The public CLI repeated the real Qwen experiment successfully using `--model-cache`;
  its model identity and all 24 probabilities exactly matched the Python acceptance run.
- The report was rendered at desktop and true 390px mobile widths. Rule/split filters
  displayed the expected 2, 4 and 12 rows, with no page-width overflow on mobile.

These checks validate this implementation and integration seam. They do not establish
model improvement, probability calibration, or production readiness.
