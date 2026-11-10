# Security policy and supported use

Version 0.3 adds trusted local Python authoring and a caller-owned model adapter.
Its callback runs in the caller's process with that user's privileges; it has no
sandbox or worker timeout. MCP accepts declarative data and does not load callbacks,
Python modules or code from experiment artifacts. Numeric tensor artifacts are
bounded and checksummed; these checks establish integrity, not trustworthy behavior.
Diagnostic logit and evidence limits do not bound total model memory. See the
[0.3 security delta review](docs/security-wave3.md) for the expanded surface.

nrly is experimental, single-user research software. Publishing its source does not
make its MCP server a service suitable for anonymous or untrusted remote users.
The current maintained development series is 0.3. Older experimental snapshots do
not have a separate security backport commitment.

## Trust boundaries

- The provided server uses local MCP stdio and opens no listening port. It has no
  account authentication, tenant separation, global request quota or durable-storage
  quota. Do not expose it through an HTTP bridge or shared service without adding
  those controls. Anyone with access to the MCP connection can use its available tools.
- Model workers execute as the launching OS user and inherit its environment and
  filesystem permissions. Process isolation, timeouts and tensor budgets are not an
  OS security sandbox, CPU/RAM/GPU memory quota, or network egress restriction.
  Run under an appropriately restricted account and environment when needed.
- Hugging Face loading requires safetensors and disables remote Python code. Public
  Hub revisions must be pinned. Local model directories are still caller-controlled
  files; review their origin. A pinned revision or hash does not make an artifact
  trustworthy or protect against vulnerabilities in native model dependencies.
- Model loading is local-only by default. A caller who sets `local_files_only=false`
  authorizes the installed Hub client to use the network and any available Hub
  credentials. Other runtime configuration is inherited from the host environment.
- Programs are validated declarative operations, not arbitrary Python. Experiments
  can still consume substantial compute. The edited tensor-storage ceiling excludes
  original models, temporary buffers and activations.

## Data and exported artifacts

Run stores contain prompts, expected answers, outputs, diagnostics and error details.
Study stores also contain raw holdout cases. They are not encrypted. Filesystem
owners can read them; the study API's holdout isolation does not protect against that.
Use private OS permissions and keep stores out of public or synchronized folders.

The repository's checked-in validation material consists of deliberately public tiny
fixtures and public Qwen smoke prompts. The sample holdout is published, not unseen
scientific evidence. New run bundles and reports need review before sharing: they can
contain user data and local paths. Checksums establish integrity, not authenticity.
Replay executes the bundle's model and resource requests; inspect them first.

CorePack context export omits case text, raw outputs and private run handles. It still
contains study/candidate names and metadata, which may be confidential in a real
project. Treat model text and imported context as data, not authority for tool calls.
No context export automatically changes agent configuration or grants permissions.

HTML reports escape text and use a restrictive content security policy. They do not
load network resources or execute JavaScript. Keep full reports private when inputs
or generated outputs are sensitive.

## Before publishing changes

```powershell
uv run --no-sync python scripts/check_publication.py
git status --short --untracked-files=all
```

The guard inspects tracked files even if `.gitignore` now excludes them, and rejects
private-state paths, credential files, profile paths, unexpected binaries and large
files. It does not prove that text is free of secrets. Run a maintained secret scanner
without credential verification, audit locked dependencies, and inspect the actual
wheel/source-archive contents. Do not commit scanner reports containing credentials.

`.gitignore` is a default exclusion, not redaction. It cannot remove previously committed
secrets or protect a custom store outside its patterns. If a credential is exposed,
revoke it and inspect Git history; deleting the working-tree copy is insufficient.
Downloaded models and dependencies have their own licenses, separate from nrly's MIT
license. nrly does not redistribute model checkpoints.

## Reporting a vulnerability

Use the repository's **Report a vulnerability** option if the maintainer has enabled
GitHub private vulnerability reporting. Otherwise establish a private contact with the
maintainer before sending exploit details or sensitive artifacts. No dedicated reporting
service or response-time guarantee is currently established. Do not post credentials,
private model data, or raw user experiment stores in a public issue.
