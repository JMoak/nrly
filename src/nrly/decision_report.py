"""Offline review surface over the aggregate-only public assessment."""

from __future__ import annotations

import base64
import hashlib
import html

from nrly.decision_contracts import DecisionAssessment


def render_decision_report(assessment: DecisionAssessment) -> str:
    view = DecisionAssessment.model_validate(assessment.model_dump())
    rows = []
    for metric in view.metrics:
        cells = [
            metric.variant.title(),
            metric.rule.title(),
            metric.split.title(),
            "Always review" if metric.threshold is None else f"{metric.threshold:g}",
            f"{metric.scored}/{metric.cases} · {metric.families} families",
            f"{metric.coverage:.1%}",
            f"{metric.unsupported_acts}/{metric.acts}",
            "—" if metric.agreement is None else f"{metric.agreement:.1%}",
            "—" if metric.brier is None else f"{metric.brier:.4f}",
        ]
        rows.append(
            f'<tr data-rule="{metric.rule}" data-split="{metric.split}">'
            + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells)
            + "</tr>"
        )
    headline = {
        "retain_original": "Keep the original. Learn from the edit.",
        "further_shadow_evaluation": "A signal worth testing in shadow.",
        "incomplete": "The evidence is incomplete.",
    }[view.recommendation]
    script = _TEMPLATE.split("<script>", 1)[1].split("</script>", 1)[0]
    script_hash = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    return (
        _TEMPLATE.replace("__HEADLINE__", headline)
        .replace("__SCRIPT_HASH__", script_hash)
        .replace(
            "__ROWS__",
            "\n".join(rows),
        )
        .replace("__MODEL__", html.escape(view.model_id))
        .replace(
            "__SEMANTICS__",
            html.escape(view.semantics.replace("_", " ")),
        )
        .replace("__ORIGIN__", html.escape(view.label_origin.replace("_", " ")))
        .replace(
            "__RESCUED__",
            str(view.rescued_supported_acts),
        )
        .replace("__INTRODUCED__", str(view.introduced_unsupported_acts))
        .replace(
            "__LOST__",
            str(view.lost_supported_acts),
        )
        .replace("__BUDGET__", f"{view.max_unsupported_rate:.1%}")
        .replace(
            "__RUN__",
            view.run_digest,
        )
        .replace("__FREEZE__", view.freeze_digest)
    )


_TEMPLATE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none';
style-src 'unsafe-inline'; script-src 'sha256-__SCRIPT_HASH__';
base-uri 'none'; form-action 'none'">
<title>nrly · decision review</title>
<style>
:root{color-scheme:dark;--ink:#f0edf7;--muted:#aaa8bc;--line:#353146;--mint:#8ee3cf}
*{box-sizing:border-box}body{margin:0;background:#101019;color:var(--ink);
font:16px/1.6 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:48px 28px}
nav{display:flex;justify-content:space-between;align-items:center;
border-bottom:1px solid var(--line);
padding-bottom:24px}.brand{font-size:36px;font-weight:800;letter-spacing:-3px}
.pill,.eyebrow{font:12px/1.5 ui-monospace,monospace;letter-spacing:2px;text-transform:uppercase}
.pill{border:1px solid #4b655f;color:var(--mint);padding:6px 12px;border-radius:24px}
header{padding:55px 0 32px;max-width:800px}.eyebrow{color:var(--mint)}
h1{font-size:clamp(32px,5vw,60px);line-height:1.1;letter-spacing:-2px;margin:18px 0 24px}
p{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}
.card{border:1px solid var(--line);border-radius:16px;padding:22px;background:#171622}
.number{font-size:38px;display:block;color:var(--ink);font-variant-numeric:tabular-nums}
.card span{color:var(--muted)}h2{font-size:22px;margin:0 0 6px}
section{margin-top:38px}.controls{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}
select{font:inherit;color:var(--ink);background:#211f30;border:1px solid #62557a;
border-radius:8px;padding:8px 14px;margin-left:10px}label{color:var(--muted)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}
th,td{text-align:left;padding:15px 12px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500;background:#211e2d}tr:last-child td{border-bottom:0}
td:first-child{font-weight:650;color:var(--mint)}code{overflow-wrap:anywhere;font-size:12px}
details{border-top:1px solid var(--line);padding:20px 0}summary{cursor:pointer}
footer{color:var(--muted);border-top:1px solid var(--line);margin-top:44px;padding-top:24px}
[hidden]{display:none!important}@media(max-width:600px){main{padding:24px 16px}
.cards{gap:8px}.card{padding:12px}.card span{font-size:12px}.number{font-size:30px}}
</style><main><nav><span class="brand">nrly</span><span class="pill">Shadow research</span></nav>
<header><div class="eyebrow">Weight-space programming / decision review</div>
<h1>__HEADLINE__</h1><p>Compare the edit with the original model after giving both
their own fitted acceptance threshold, using the same fitting opportunity.</p></header>
<div class="cards"><div class="card"><b class="number">__RESCUED__</b>
<span>new supported acts</span></div><div class="card"><b class="number">__INTRODUCED__</b>
<span>new unsupported acts</span></div><div class="card"><b class="number">__LOST__</b>
<span>lost supported acts</span></div></div>
<p>Paired changes across evaluation and challenge cases, using separately fitted rules.
A failed execution never authorizes an act.</p>
<section><h2>Did editing buy useful coverage?</h2><p>Empirical unsupported-act budget:
__BUDGET__. This is an exploratory measurement, not a reliability guarantee.</p>
<div class="controls" hidden id="controls"><label>Decision rule<select id="rule">
<option value="fitted">Separately fitted</option><option value="fixed">Fixed reference</option>
<option value="all">Both</option></select></label><label>Data<select id="split">
<option value="evaluation">Evaluation</option><option value="challenge">Challenge</option>
<option value="fit">Fit / development</option><option value="all">All splits</option></select>
</label></div><div class="scroll"><table><thead><tr><th>Weights</th><th>Rule</th><th>Split</th>
<th>Accept at ≥</th><th>Scored / total</th><th>Coverage</th><th>Errors / acts</th>
<th>Raw agreement</th><th>Brier ↓</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p id="empty" hidden>No cases in this split.</p>
<p>Coverage uses all cases as its denominator. Agreement and Brier use scored cases only.
Threshold fitting changes which acts are accepted; it does not calibrate probabilities or
change raw agreement and Brier loss. Missing evidence makes the whole comparison incomplete.</p>
</section><section><h2>Know what this run establishes</h2><p>Model: <b>__MODEL__</b>.
Output: <b>__SEMANTICS__</b>. Label origin: <b>__ORIGIN__</b>.</p>
<p>This review contains aggregate evidence only. Private states, truth labels and source-family
identifiers are absent from this file. Source-family counts describe grouping; they do not make
rows independent statistical trials. No production gate or deployment is authorized.</p>
<details><summary>Recorded identities</summary><p>Run<br><code>__RUN__</code></p>
<p>Candidate and decision-rule freeze<br><code>__FREEZE__</code></p></details></section>
<footer>Inspect the edit. Measure the consequence. Keep what the evidence supports.</footer></main>
<script>
const rule=document.getElementById('rule'),split=document.getElementById('split');
function filter(){let count=0;document.querySelectorAll('tbody tr').forEach(row=>{
row.hidden=!(rule.value==='all'||row.dataset.rule===rule.value)||
!(split.value==='all'||row.dataset.split===split.value);if(!row.hidden)count++});
document.getElementById('empty').hidden=count>0;}
document.getElementById('controls').hidden=false;
rule.addEventListener('change',filter);split.addEventListener('change',filter);filter();
</script></html>"""
