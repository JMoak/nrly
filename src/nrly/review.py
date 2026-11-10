from __future__ import annotations

import html
import json
from pathlib import Path

from nrly.contracts import NrlyError, RunRecord
from nrly.evaluation import compare_runs, evaluate_run
from nrly.store import atomic_text


def render_run_report(run: RunRecord, baseline: RunRecord | None = None) -> str:
    evaluation = evaluate_run(run)
    comparison = compare_runs(baseline, run) if baseline else None
    verdict = comparison.verdict if comparison else run.status
    baseline_samples = {sample.case_id: sample for sample in baseline.samples} if baseline else {}
    samples = {sample.case_id: sample for sample in run.samples}
    rows: list[str] = []
    for case in evaluation.cases[:200]:
        previous = baseline_samples.get(case.case_id)
        sample = samples.get(case.case_id)
        rows.append(
            f"<tr><td><strong>{escape(case.case_id)}</strong><small>{escape(case.role)} · "
            f"{escape(case.split)}</small></td><td>{text_cell(case.expected)}</td>"
            + (f"<td>{text_cell(previous.output if previous else None)}</td>" if baseline else "")
            + f'<td>{text_cell(case.output)}</td><td><span class="badge {case.outcome}">'
            f"{case.outcome}</span><small>{escape(case.reason)}</small>"
            f"<small>{escape(sample.provenance if sample else 'missing inference')}</small>"
            "</td></tr>"
        )
    changes: list[str] = []
    seen: set[str] = set()
    for sample in run.samples:
        for change in sample.weight_changes:
            serialized = json.dumps(change.model_dump(mode="json"), sort_keys=True)
            if serialized in seen:
                continue
            seen.add(serialized)
            if len(changes) >= 64:
                continue
            selected = list(
                zip(change.edited_indices, change.edited_before, change.edited_after, strict=True)
            )
            coordinate_rows = "".join(
                f"<tr><td>{index}</td><td>{escape(before)}</td><td>{escape(after)}</td></tr>"
                for index, before, after in selected
            )
            changes.append(
                '<article class="change">'
                f"<h3>{escape(change.target)}</h3>"
                f"<p>{escape(change.operation_kind or run.spec.program.kind)} · "
                f"{change.changed_elements:,} changed in this operation of "
                f"{change.total_elements:,} elements · {escape(change.dtype)} · "
                f"{escape(change.shape)}</p><dl><dt>Before SHA256</dt>"
                f"<dd>{escape(change.before_sha256)}</dd><dt>After SHA256</dt>"
                f"<dd>{escape(change.after_sha256)}</dd></dl>"
                + (
                    f"<p><strong>Final net change: {change.net_changed_elements:,} elements"
                    "</strong> compared with the original tensor.</p>"
                    f"<dl><dt>Original SHA256</dt><dd>{escape(change.net_before_sha256)}</dd>"
                    f"<dt>Final SHA256</dt><dd>{escape(change.net_after_sha256)}</dd></dl>"
                    if change.net_changed_elements is not None
                    else ""
                )
                + (
                    "<table><thead><tr><th>Flat index</th><th>Before</th><th>After</th>"
                    "</tr></thead>"
                    f"<tbody>{coordinate_rows}</tbody></table>"
                    if selected
                    else ""
                )
                + "</article>"
            )
    issues = comparison.issues if comparison else evaluation.issues
    issue_list = "".join(f"<li>{escape(issue)}</li>" for issue in issues[:30])
    summary = evaluation.summary
    accuracy = "Unscored" if summary.accuracy is None else f"{summary.accuracy:.1%}"
    scope = (
        "Engineering fixture — demonstrates execution, not model capability."
        if evaluation.scope == "engineering_fixture"
        else "Real-model observations — this report does not establish generalization."
    )
    fingerprint = run.model_identity.fingerprint if run.model_identity else "Unavailable"
    program = json.dumps(run.spec.program.model_dump(mode="json"), indent=2, ensure_ascii=False)
    if len(program) > 24000:
        program = program[:24000] + "\n… Preview truncated. Use CLI export for the full program."
    inspection = ""
    if run.inspection:
        inspection = "<h2>Parameter inspection</h2>" + "".join(
            f"<article><h3>{escape(weight.target)}</h3>"
            f"<p>{escape(weight.shape)} · {escape(weight.dtype)}</p>"
            f"<pre>{escape(json.dumps(weight.model_dump(mode='json'), indent=2)[:6000])}</pre>"
            "</article>"
            for weight in run.inspection.weights
        )
    diagnostics = ""
    observed = [sample for sample in run.samples if sample.diagnostics is not None]
    if observed:
        diagnostics = (
            "<h2>Continuation diagnostics</h2><p>Probabilities score supplied text. "
            "They are not generated answers or task accuracy. Preview: up to 20 cases "
            "and 32 tokens per case; export or page diagnostics for full evidence.</p>"
        )
        for sample in observed[:20]:
            score = sample.diagnostics
            if score is None:
                continue
            token_rows = "".join(
                f"<tr><td>{token.position}</td><td>{escape(token.text[:256])}</td>"
                f"<td>{token.logprob:.6f}</td><td>{token.rank}</td></tr>"
                for token in score.tokens[:32]
            )
            diagnostics += (
                f"<article><h3>{escape(sample.case_id)}</h3><p>"
                f"{score.continuation_tokens} continuation tokens · "
                f"sum log probability {score.sum_logprob:.6f} · "
                f"mean {score.mean_logprob:.6f}</p>"
                "<table><thead><tr><th>Position</th><th>Token</th><th>Log probability</th>"
                f"<th>Rank</th></tr></thead><tbody>{token_rows}</tbody></table></article>"
            )
    body = (
        '<header><a class="brand" href="#top">nrly</a>'
        "<span>WEIGHT-SPACE PROGRAMMING</span></header>"
        f'<main id="top"><p class="eyebrow">Experiment review</p><h1>{escape(run.spec.name)}</h1>'
        f'<p class="scope">{escape(scope)}</p><section class="metrics">'
        f"<article><small>Result</small><strong>{escape(verdict)}</strong></article>"
        f"<article><small>Task accuracy</small><strong>{accuracy}</strong></article>"
        f"<article><small>Scored cases</small><strong>{summary.scored} / {summary.total}"
        "</strong></article>"
        f"<article><small>Runtime</small><strong>{run.elapsed_ms / 1000:.2f}s</strong></article>"
        "</section>"
        + (
            f"<aside><strong>Evidence limitations</strong><ul>{issue_list}</ul></aside>"
            if issues
            else ""
        )
        + "<h2>Behavioral evidence</h2>"
        f"<p>{summary.passed} passed · {summary.failed} failed · {summary.unscored} unscored. "
        "Target and regression outcomes remain separate in the exported evaluation.</p>"
        '<div class="table-scroll"><table><thead><tr><th>Case</th><th>Expected</th>'
        + ("<th>Baseline</th>" if baseline else "")
        + "<th>Candidate output</th><th>Outcome</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + (
            f"<p>Showing 200 of {summary.total} cases. Use CLI export for all records.</p>"
            if summary.total > 200
            else ""
        )
        + "<h2>Weight changes</h2><p>Exact parameter evidence; repeated identical observations "
        "are grouped. These records do not assign semantic labels to neurons.</p>"
        + ("".join(changes) or "<p>No weight changes recorded.</p>")
        + ("<p>Additional changes omitted from this preview.</p>" if len(seen) > 64 else "")
        + inspection
        + diagnostics
        + "<details><summary>Executable program</summary>"
        f"<pre>{escape(program)}</pre></details><details><summary>Identity and provenance</summary>"
        f"<dl><dt>Run</dt><dd>{escape(run.id)}</dd><dt>Model</dt><dd>{escape(run.spec.model.model_id)}</dd>"
        f"<dt>Revision</dt><dd>{escape(run.spec.model.revision)}</dd><dt>Fingerprint</dt>"
        f"<dd>{escape(fingerprint)}</dd><dt>Specification SHA256</dt>"
        f"<dd>{escape(run.spec_digest)}</dd>"
        f"<dt>Created</dt><dd>{escape(run.created_at)}</dd></dl></details>"
        "<footer>Local, offline report. Checksums support integrity; independent replay supplies "
        "fresh execution evidence. Full records remain in the nrly bundle.</footer></main>"
    )
    return document(run.spec.name, body)


def write_run_report(run: RunRecord, destination: Path, baseline: RunRecord | None = None) -> Path:
    if destination.exists():
        raise NrlyError("destination_exists", "report destination already exists")
    atomic_text(destination, render_run_report(run, baseline))
    return destination.resolve()


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def text_cell(value: str | None) -> str:
    if value is None:
        return '<span class="muted">Unavailable</span>'
    preview = value[:2048]
    suffix = "\n… truncated in this view" if len(value) > 2048 else ""
    return f"<pre>{escape(preview + suffix)}</pre>"


def document(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'\">"
        f"<title>{escape(title)} · nrly</title><style>"
        ":root{color-scheme:dark;font:15px/1.55 system-ui,sans-serif;"
        "background:#101418;color:#e4e9ec}"
        "*{box-sizing:border-box}body{margin:0}header{display:flex;gap:24px;align-items:center;"
        "padding:20px 5vw;border-bottom:1px solid #303b43}"
        "header span{font-size:11px;letter-spacing:2px}"
        ".brand{color:#8bf4bd;font-weight:800;font-size:25px;text-decoration:none}main{max-width:1200px;"
        "margin:auto;padding:36px 24px}h1{font-size:clamp(28px,5vw,45px);"
        "line-height:1.15;margin:8px 0 16px}"
        "h2{margin-top:40px}h3{overflow-wrap:anywhere;font:600 15px/1.5 ui-monospace,monospace}"
        ".eyebrow{color:#8bf4bd;text-transform:uppercase;letter-spacing:2px;font-size:12px}"
        ".scope{color:#b7c5cc}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;"
        "margin:28px 0}.metrics article,article.change{border:1px solid #34424c;background:#182128;"
        "border-radius:12px;padding:18px}.metrics strong{display:block;"
        "font-size:24px;overflow-wrap:anywhere}"
        "small{display:block;color:#a8bac4;font-size:12px}table{width:100%;border-collapse:collapse;"
        "font-size:13px}th{text-align:left;color:#a8bac4}th,td{padding:14px 10px;"
        "border-bottom:1px solid "
        "#303e47;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;"
        "font:12px/1.6 ui-monospace,monospace;max-height:400px;overflow:auto}"
        ".table-scroll{overflow-x:auto}"
        ".badge{display:inline-block;border-radius:99px;padding:2px 10px;background:#34424c}"
        ".pass{color:#8bf4bd}.fail{color:#ffadad}.unscored,.muted{color:#c2c8ce}.change{margin-bottom:12px}"
        "dl{font-size:12px}dt{color:#a8bac4;margin-top:8px}dd{margin:0;overflow-wrap:anywhere;"
        "font-family:ui-monospace,monospace}details{border-top:1px solid #34424c;padding:18px 0;"
        "margin-top:24px}summary{cursor:pointer;font-weight:600}details pre{margin-top:20px}"
        "aside{border-left:3px solid #e6b971;padding:14px 20px;background:#28251d}"
        "footer{margin-top:48px;color:#9aaeb9;font-size:12px}"
        "@media(max-width:650px){.metrics{grid-template-columns:1fr 1fr}"
        "header span{font-size:9px}main{padding:24px 16px}}"
        "</style></head><body>" + body + "</body></html>"
    )
