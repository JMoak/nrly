# Retained harness failure and completion

The initial harness stopped after all sixteen sampled study workers succeeded
because it incorrectly required `final_error_code` to be empty. The then-current
engine used `study_inconclusive` for a fully scored, comparable scientific result
as well as for incomplete evidence. `failure.json` retains that assertion failure.

The recorded final verdict is `inconclusive`, with `retain_original`: the candidate
did not improve target scores beyond every registered control. It changed one
formatting-sensitive regression outcome under one seed; the prompt comparator
also passed that case. This does not support a capability-improvement claim.

All original study records, outputs, and the event ledger remain unchanged. The
engine now distinguishes a valid scientific inconclusive outcome from execution
failure, covered by focused lifecycle tests. Acceptance recovery verified that
every arm completed and every registered comparison was comparable, then
performed the remaining strict wave2 replay without rerunning the study.

`summary.json` explicitly retains the recorded `study_inconclusive` error code and
the prior harness failure. Its `status` describes the runtime integration checks,
not a successful scientific candidate. Overall elapsed time is left null because
the original harness process ended; the recovery stage duration is recorded.

`artifact-inventory.json` is the original completion snapshot. The additive
`artifact-inventory-final.json` includes this recovery explanation and the original
inventory. Both inventories preserve the original run/study evidence digests.
