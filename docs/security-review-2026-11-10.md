# Public release security review

Reviewed 10 November 2026. Reviewed package: `nrly 0.2.0a2`.

The reviewed source is reasonable to publish as experimental, trusted-local research
software after the fixes below. No credential exposure or remaining high/medium
static-analysis finding was identified. This is a scoped review, not a guarantee that
the software has no vulnerabilities. Deployment for untrusted remote users is outside
the supported security model described in [SECURITY.md](../SECURITY.md).

## Findings and fixes

| Finding | Change | Evidence |
| --- | --- | --- |
| A generated `.nrly-wave2-qwen-strict` store was eligible for `git add`, including its SQLite database and worker log | Ignore every default `.nrly-*` store, credentials files, logs, databases and checkpoint files; add source-package exclusions | Git publication guard passes; test proves a force-tracked private file still fails the guard |
| Hugging Face loading did not explicitly forbid fallback to pickle-based checkpoints | Require `use_safetensors=True`, preserving `trust_remote_code=False` | A valid pickle-only checkpoint is rejected without calling `torch.load`; normal safetensors fixtures and real Qwen replay pass |
| Final-number parsing could perform quadratic backtracking on malformed whitespace suffixes | Strip trailing whitespace once and remove overlapping whitespace repetitions | A 100,002-character malformed answer returns in about 0.003 seconds; bounded-process regression test and format-compatibility tests pass |
| A final-evaluation invariant relied on a Python assertion | Replace it with an explicit typed error, which remains active under optimized Python | Study and complete test suites pass |
| Public-use limits and license were missing from the release surface | Add security policy, publication guard and the owner's selected MIT license; include license in package metadata | Source and package checks; no configuration or service registration changes |

The generated store contained this project's public smoke experiment, not a discovered
credential. The exclusion defect could nevertheless expose future private experiments.
The safetensors change is preventive hardening; the review did not demonstrate arbitrary
code execution through the previous dependency version.

## Inspection and scans

The review inspected the Git publication set, package boundaries, model loading,
subprocess construction, SQL queries, path handling, study isolation, JSON parsing,
HTML output and TypeScript transport boundaries. Git had zero commits and no configured
remote, so there was no historical commit content to scan. Ignored model caches,
environments and private stores are not publication inputs.

- `detect-secrets 1.5.0` ran with credential verification disabled. Its 231 initial
  findings were high-entropy hexadecimal strings. Each was classified as an artifact
  checksum, generated run/study identifier, model fingerprint, or public revision ID.
  No credential was identified. Future scans must review new findings rather than
  suppressing all hashes or ignoring entire evidence directories.
- `pip-audit 2.10.1`, using the PyPI advisory service, checked all 87 registry package
  versions in `uv.lock`: no known advisories and no skipped packages. The pinned
  Hatchling 1.33.0 build backend was checked separately with no advisory found.
  CUDA's local version suffix was normalized to upstream Torch 2.15.0 for the query;
  this does not attest that binary build or independently audit native CUDA libraries.
- `bun audit --json` returned an empty advisory result for the locked JavaScript
  dependencies. No dependency upgrades were needed.
- `bandit 1.9.4` inspected Python runtime and scripts: no scan errors and no medium or
  high findings. The remaining 65 low findings concern fixed executable/argument-array
  subprocess calls and assertions in acceptance scripts. The runtime has no assertion
  serving as the reviewed nomination guard, no shell command interpolation, and no
  arbitrary candidate-code execution.
- The publication guard found no private-state paths, personal home-directory paths,
  unexpected binary files or oversized files in the final Git publication set.

Scanner reports and raw test diagnostics are retained locally under the ignored
`.cache/security-review` directory. Scanners were run from isolated tool environments;
their dependencies were not added to nrly's runtime.

## Verification

254 Python tests passed. Ruff and strict mypy passed, as did all 12 public JSON schema
checks and generated TypeScript contract checks. Eighteen SDK unit tests, 15 Vibrissa
cases and three real MCP lifecycle tests passed. The managed environment's default
temporary directory rejected writes on the first lifecycle attempt; rerunning with
a writable test temporary directory passed all cases.

A fresh Qwen replay under the safetensors-only loader matched outputs, identity,
inspection, stable sample evidence and weight-change evidence. The recipe has no
activation captures, so its activation equality compares empty lists; nonempty
capture mismatch checks remain covered by tests. This is regression verification,
not a new model-quality claim.

The rebuilt wheel and source archive contain the MIT license and the reviewed source.
Archive inspection found no model checkpoints, runtime databases, logs, installed
dependencies or scanner cache. Deliberately planted private-directory and environment-file
test markers were excluded from both packages. Wheel module bytes match the source files.

## Publication and operating limits

Run `python scripts/check_publication.py` and inspect `git status --short` before
staging. The guard examines already tracked files even when `.gitignore` excludes
them. Repeat secret and advisory scans before later releases; today's result cannot
certify future changes or future advisories. Review every newly shared bundle and
context pack for project-specific confidential text and metadata.

The MCP server remains stdio-only, with no authentication or tenant boundary. Workers
inherit the launching user's privileges and environment. Resource limits are not a
security sandbox or hard memory/disk quota. Holdout isolation is an application/API
boundary, not protection against filesystem access. Repository publication does not
authorize registering or exposing a network service.

The MIT license applies to nrly's authored source. Models and dependencies retain their
own terms. Checkpoints are not redistributed. No commit, remote, publication, public
service, private-reporting endpoint or actual user configuration was created by this
review. Prefer the reviewed 0.2.0a2 artifacts over the earlier 0.2.0a1 snapshots.
