# nrly CorePack context export

This optional pack follows the scoped .ai layout described by CorePack-AI's
Unified Context Protocol v1.0: https://github.com/CorePack-AI/ucp-spec/blob/c8c637e554ed86cef6acaa2827fd839d68661935/README.md

Start with `.ai/context/@nrly/weight-space/active/PLAN.md`. Read explicitly or copy the scoped
module/context directories into a project after reviewing existing files. No
automatic discovery by Keywork, Codex, or another client is asserted.

The upstream document describes a filesystem convention, not a wire transport or
an executable JSON schema. The manifest and evidence JSON here are nrly formats.
The manifest hashes exported files; it is not an upstream certificate or signature.
The scoped layout follows the upstream directory tree. Its lifecycle paragraph
abbreviates the plan path; nrly retains the explicit organization/pack scope.

`generators/identity-candidate.json` validates as an nrly CandidateSpec and is only
a template. The archive directory is intentionally empty for the initial snapshot.
Every exported file is inert data. Existing project instructions remain authoritative.
