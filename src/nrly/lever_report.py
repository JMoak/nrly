"""Offline lever map: one self-contained HTML page over sealed lever entries.

The page is meant to be shared, so it carries counts and aggregates only. Prompts,
expected answers, case content and operation payloads are never written. Every string
that comes from an entry is untrusted: it is escaped and never used to build markup.

Line style is the vocabulary. Pencil (dashed graphite) is anything claimed, derived or
inherited; ink (a solid dark line that thickens with independent replications) comes only
from a registered held-out verdict; a struck entry keeps its content under a strike. The
dial is always drawn in graphite because it is exploratory development evidence. The one
accent colour marks the dial's working window and nothing else.

The stylesheet is a constant admitted by hash, so the policy allows no other style and
no script, and markup that slipped past escaping could not restyle a status.
"""

from __future__ import annotations

import base64
import hashlib
import html
import math
import re
from collections import Counter
from collections.abc import Sequence

from pydantic import ValidationError

from nrly.contracts import NrlyError, PortableBase
from nrly.lever_contracts import (
    Dial,
    LeverEntry,
    LeverStatus,
    RefereeVerdict,
    operations_digest,
    status_reason,
    verdict_status,
)
from nrly.study_contracts import EffectEstimate

MAX_ENTRIES = 500
MAX_REPORT_BYTES = 2 * 1024 * 1024

_MAX_LAYERS = 4096
# Deeper strips fold long unsurveyed runs into one labelled cell so the page stays bounded.
_FULL_STRIP = 96
_FOLD_RUN = 9
_MAX_REFERENCES = 24
_MAX_ISSUES = 5

_STATUSES: tuple[LeverStatus, ...] = ("inked", "pencil", "struck")
_LABEL: dict[LeverStatus, str] = {"inked": "Inked", "pencil": "Pencil", "struck": "Struck"}
_MEANING: dict[LeverStatus, str] = {
    "inked": "a registered held-out study supports the edit",
    "pencil": "claimed, derived or inherited; nothing established",
    "struck": "refereed on held-out cases and not supported; kept on the map",
}
_OUTCOME = {
    "observed_improvement": "The study observed an improvement on held-out target cases.",
    "no_change": (
        "The study found no supported effect on held-out target cases, with regressions "
        "preserved. This is not an equivalence claim."
    ),
    "regressed": "The study observed harm on held-out cases: targets or regressions got worse.",
    "inconclusive": "The study was inconclusive on held-out cases.",
}
_RECOMMENDATION = {
    "candidate_supported": "It supports the candidate.",
    "retain_original": "It recommends retaining the original weights.",
}

_SPACE = re.compile(r"[\t\n\r\x0b\x0c]+")
# Controls, line separators, bidirectional overrides, surrogates and noncharacters can
# corrupt the document or disguise text, so they are shown as a replacement character.
_UNSAFE = re.compile(
    "["
    + "".join(
        f"{chr(low)}-{chr(high)}"
        for low, high in (
            (0x00, 0x1F),
            (0x7F, 0x9F),
            (0x2028, 0x202E),
            (0x2066, 0x2069),
            (0xD800, 0xDFFF),
            (0xFFFE, 0xFFFF),
        )
    )
    + "]"
)
_REPLACEMENT = chr(0xFFFD)
# Only RFC 3986 characters after an explicit http(s) scheme; everything else stays text.
_HTTP_URL = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+", re.IGNORECASE)


def render_map(
    entries: Sequence[LeverEntry], *, title: str = "nrly lever map", layers: int | None = None
) -> str:
    """Render a lever map: the base, the status counts, the terrain, then one datasheet each.

    ``layers`` is the model's depth when known; without it the terrain ends at the deepest
    tested layer and says so. Entries are grouped by base and ordered by layer, and a
    repeated entry is drawn once, so the same set of entries always produces the same bytes.
    """
    if len(entries) > MAX_ENTRIES:
        raise NrlyError(
            "lever_report_too_large", f"a lever map renders at most {MAX_ENTRIES} entries"
        )
    if layers is not None and (isinstance(layers, bool) or not 1 <= layers <= _MAX_LAYERS):
        raise NrlyError("invalid_request", f"layers must be between 1 and {_MAX_LAYERS}")
    unique: dict[str, LeverEntry] = {}
    for entry in entries:
        checked = _checked(entry)
        unique.setdefault(checked.id, checked)
    groups: dict[tuple[str, ...], list[LeverEntry]] = {}
    for entry in unique.values():
        groups.setdefault(_base_key(entry.base), []).append(entry)
    for group in groups.values():
        group.sort(key=_order)
    ordered = [groups[key] for key in sorted(groups)]

    heading = _e(str(title).strip(), 200) or "nrly lever map"
    if not ordered:
        scope = "No entries. Nothing has been recorded on this map."
    elif len(ordered) == 1:
        scope = "Tested places in the weights of one exact checkpoint."
    else:
        scope = (
            f"Tested places in the weights of {len(ordered)} exact checkpoints, grouped by base."
        )
    totals = Counter(entry.status for entry in unique.values())
    parts = [
        f"<header><h1>{heading}</h1>"
        f'<p class="lede">{scope} Each entry is a site, a direction, a dose-response dial and '
        "a verdict, named by its test and its site. Any meaning attached to an entry is a "
        "hypothesis attributed to its source.</p>"
        + (_base_block(ordered[0][0].base) if len(ordered) == 1 else "")
        + _tally(totals, key=True)
        + "</header>"
    ]
    if unique:
        parts.append(_FILTER)
    size = 0
    for group in ordered:
        opening = '<section class="grp">'
        if len(ordered) > 1:
            counts = Counter(entry.status for entry in group)
            opening += "<h2>Base</h2>" + _base_block(group[0].base) + _tally(counts, key=False)
        parts.append(opening + _terrain(group, layers))
        for entry in group:
            sheet = _sheet(entry)
            size += len(sheet)
            if size > MAX_REPORT_BYTES:
                raise _too_large()
            parts.append(sheet)
        parts.append("</section>")
    parts.append(_FOOTER)
    return _document(heading, "\n".join(parts))


def render_lever(entry: LeverEntry) -> str:
    """Render one entry's datasheet on its own, with the base it is pinned to."""
    checked = _checked(entry)
    body = (
        "<header><h1>Lever datasheet</h1>"
        '<p class="lede">One tested place in the weights of one exact checkpoint: a site, a '
        "direction, a dose-response dial and a verdict. Any meaning attached to it is a "
        "hypothesis attributed to its source.</p>"
        + _base_block(checked.base)
        + "</header>\n"
        + _sheet(checked)
        + "\n"
        + _FOOTER
    )
    return _document(_e(checked.name, 120) + " &#183; nrly lever", body)


# --- untrusted text -------------------------------------------------------------------


def _e(value: object, limit: int | None = None) -> str:
    """Escape untrusted text for element content or a quoted attribute, optionally clipped."""
    text = _UNSAFE.sub(_REPLACEMENT, _SPACE.sub(" ", str(value)))
    if limit is not None and len(text) > limit:
        return html.escape(text[:limit], quote=True) + "&#8230;"
    return html.escape(text, quote=True)


def _short(digest: str) -> str:
    return _e(digest[:12])


def _link(text: str) -> str:
    """A link only for a plain http(s) URL; any other scheme or shape stays inert text."""
    if len(text) <= 500 and _HTTP_URL.fullmatch(text):
        return f'<a href="{_e(text)}" rel="noopener noreferrer">{_e(text)}</a>'
    return _e(text, 500)


def _checked(entry: LeverEntry) -> LeverEntry:
    """Re-derive identity and status so a hand-altered entry cannot draw itself in ink."""
    if not isinstance(entry, LeverEntry):
        raise NrlyError("invalid_request", "a lever report renders sealed lever entries only")
    try:
        return LeverEntry.model_validate(entry.model_dump())
    except ValidationError as error:
        raise NrlyError(
            "invalid_request", "a lever entry does not match its sealed identity"
        ) from error


def _too_large() -> NrlyError:
    return NrlyError(
        "lever_report_too_large",
        f"the rendered lever report exceeds {MAX_REPORT_BYTES} bytes; render fewer entries",
    )


# --- numbers --------------------------------------------------------------------------


def _num(value: float) -> str:
    return f"{value + 0.0:g}"


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _pp(difference: float) -> str:
    """Signed percentage points; adding zero keeps a rounded negative zero from printing."""
    return f"{round(difference * 100, 1) + 0.0:+.1f}"


def _effect(estimate: EffectEstimate) -> str:
    return f"{_pp(estimate.mean_difference)} [{_pp(estimate.ci_low)}, {_pp(estimate.ci_high)}]"


def _p(value: float) -> str:
    return "&lt;0.0001" if value < 0.0001 else f"{value:.4f}"


def _level(confidence: float) -> str:
    return f"{confidence * 100:g}%"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


# --- small components -----------------------------------------------------------------


def _td(content: str, classes: str = "") -> str:
    """A cell; numbers align right, ``l`` aligns left, ``w`` and ``n`` let text wrap."""
    return f'<td class="{classes}">{content}</td>' if classes else f"<td>{content}</td>"


def _table(
    caption: str,
    head: Sequence[str],
    rows: Sequence[str],
    *,
    left: Sequence[int] = (0,),
    pinned: bool = False,
) -> str:
    """A table that scrolls inside its own box, so the page itself never scrolls sideways.

    ``head`` and ``rows`` are escaped markup and ``left`` lists the text columns. The
    caption sits outside the scrolling box so it stays readable on a narrow screen, and
    ``pinned`` keeps a short first column in view while the rest scrolls.
    """
    table = (
        ('<div class="scroll"><table class="pin">' if pinned else '<div class="scroll"><table>')
        + "<thead><tr>"
        + "".join(
            ('<th scope="col" class="l">' if index in left else '<th scope="col">') + cell + "</th>"
            for index, cell in enumerate(head)
        )
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return f"<figure><figcaption>{caption}</figcaption>{table}</figure>" if caption else table


def _figures(pairs: Sequence[tuple[str, str]], *, big: bool = False) -> str:
    return (
        ('<dl class="figs big">' if big else '<dl class="figs">')
        + "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in pairs)
        + "</dl>"
    )


def _section(label: str, kind: str, body: str, *, extra: str = "") -> str:
    return (
        f'<section class="sec{extra}"><h4>{label} <small>{kind}</small></h4>'
        f"<div>{body}</div></section>"
    )


def _base_key(base: PortableBase) -> tuple[str, ...]:
    return (base.model_id, base.revision, base.weights_sha256, base.dtype, base.backend)


def _order(entry: LeverEntry) -> tuple[int, str, str]:
    layers = [site.layer for site in entry.sites if site.layer is not None]
    return (min(layers, default=_MAX_LAYERS), entry.name, entry.id)


def _base_block(base: PortableBase) -> str:
    return (
        '<dl class="base">'
        f"<div><dt>Model</dt><dd><code>{_e(base.model_id)}</code></dd></div>"
        f"<div><dt>Revision</dt><dd><code>{_e(base.revision)}</code></dd></div>"
        f"<div><dt>Weights, sha256 first 12</dt><dd><code>{_short(base.weights_sha256)}</code>"
        "</dd></div>"
        f"<div><dt>Load dtype and backend</dt><dd><code>{_e(base.dtype)}</code> "
        f"<code>{_e(base.backend)}</code></dd></div>"
        "</dl>"
    )


def _tally(counts: Counter[LeverStatus], *, key: bool) -> str:
    """Counts per status, each beside its line style; ``key`` adds what each status means."""
    items = []
    for status in _STATUSES:
        meaning = f" <small>{_MEANING[status]}</small>" if key else ""
        items.append(
            f'<li><i class="mk {status}"></i><b>{counts[status]}</b>'
            f"<span>{_LABEL[status].lower()}{meaning}</span></li>"
        )
    weights = ""
    if key:
        weights = (
            '<p class="wts">Ink weight: '
            + "".join(
                f'<i class="mk inked w{weight}"></i>{label}'
                for weight, label in enumerate(("0", "1", "2", "3 or more"))
            )
            + " independent replications.</p>"
        )
    return (
        f'<ul class="tally{" key" if key else ""}">'
        + "".join(items)
        + "</ul>"
        + _tally_bar(counts)
        + weights
    )


def _tally_bar(counts: Counter[LeverStatus]) -> str:
    """Share of the map in each status, drawn in that status's own line style."""
    present = [status for status in _STATUSES if counts[status]]
    if not present:
        return ""
    total = sum(counts[status] for status in present)
    free = 600 - 3 * (len(present) - 1) - 8 * len(present)
    shapes = []
    left = 0.0
    for status in present:
        width = 8 + free * counts[status] / total
        if status == "inked":
            shapes.append(
                f'<rect class="bi" x="{left:.1f}" y="0" width="{width:.1f}" height="12"/>'
            )
        else:
            shapes.append(
                f'<rect class="b{status[0]}" x="{left + 0.75:.1f}" y="0.75" '
                f'width="{width - 1.5:.1f}" height="10.5"/>'
            )
        if status == "struck":
            shapes.append(f'<path class="bs" d="M{left:.1f} 6h{width:.1f}"/>')
        left += width + 3
    return '<svg class="bar" viewBox="0 0 600 12" aria-hidden="true">' + "".join(shapes) + "</svg>"


# --- terrain --------------------------------------------------------------------------


def _support(entry: LeverEntry) -> tuple[int, int]:
    """Replications that themselves support the edit, and their distinct new runtimes."""
    if entry.verdict is None:
        return 0, 0
    runtimes = {entry.verdict.runtime_fingerprint}
    supported = independent = 0
    for record in entry.replications:
        if verdict_status(record) != "inked":
            continue
        supported += 1
        if record.runtime_fingerprint not in runtimes:
            runtimes.add(record.runtime_fingerprint)
            independent += 1
    return supported, independent


def _weight(entry: LeverEntry) -> int:
    return _support(entry)[1] if entry.status == "inked" else 0


def _mark(entry: LeverEntry) -> str:
    weight = _weight(entry)
    classes = f"mk inked w{min(weight, 3)}" if entry.status == "inked" else f"mk {entry.status}"
    note = _LABEL[entry.status].lower()
    if entry.status == "inked":
        note += ", " + _plural(weight, "independent replication")
    return (
        f'<a class="{classes}" href="#l-{_e(entry.id)}" title="{_e(entry.name, 120)} ({note})"></a>'
    )


def _terrain(group: Sequence[LeverEntry], layers: int | None) -> str:
    """Layer strip: every lever at its layer, and every unsurveyed layer left visibly empty."""
    placed: dict[int, list[str]] = {}
    unplaced: list[str] = []
    for entry in group:
        mark = _mark(entry)
        seen: set[int | None] = set()
        for site in entry.sites:
            if site.layer in seen:
                continue
            seen.add(site.layer)
            if site.layer is None:
                unplaced.append(mark)
            else:
                placed.setdefault(site.layer, []).append(mark)
    deepest = max(placed, default=-1)
    depth = max(layers or 0, deepest + 1)
    notes = []
    if depth:
        notes.append(
            f"Layers 0 to {depth - 1}. {len(placed)} of {depth} hold a tested site; "
            f"{depth - len(placed)} are unsurveyed and drawn empty."
        )
    else:
        notes.append("No site here records a layer, so nothing can be placed on the strip.")
    if layers is None and depth:
        notes.append(
            "The model's depth was not supplied: the strip ends at the deepest tested layer "
            "and says nothing about layers beyond it."
        )
    elif layers is not None and deepest >= layers:
        beyond = sum(layer >= layers for layer in placed)
        notes.append(f"Tested layers beyond the stated depth of {layers}: {beyond}.")

    def surveyed(label: str, marks: Sequence[str], roomy: bool = False) -> str:
        # A crowded layer widens its own cell instead of stretching every cell in its row.
        wide = "x" if len(marks) > 8 else "g" if roomy or len(marks) > 4 else ""
        opening = f'<li class="{wide}">' if wide else "<li>"
        return f"{opening}<b>{label}</b>" + "".join(marks) + "</li>"

    cells = []
    fold = depth > _FULL_STRIP
    layer = 0
    while layer < depth:
        if layer in placed:
            cells.append(surveyed(str(layer), placed[layer]))
            layer += 1
            continue
        end = layer
        while end + 1 < depth and end + 1 not in placed:
            end += 1
        if fold and end - layer + 1 >= _FOLD_RUN:
            cells.append(
                f'<li class="e g"><b>{layer}&#8211;{end}</b>'
                f"<span>{end - layer + 1} unsurveyed</span></li>"
            )
        else:
            cells.extend(f'<li class="e"><b>{index}</b></li>' for index in range(layer, end + 1))
        layer = end + 1
    if unplaced:
        cells.append(surveyed("no layer", unplaced, roomy=True))
        notes.append(f"Levers with a site that records no layer, last cell: {len(unplaced)}.")
    return (
        '<section class="terrain"><h2>Terrain</h2><p>'
        + " ".join(notes)
        + "</p>"
        + ('<ol class="strip">' + "".join(cells) + "</ol>" if cells else "")
        + "</section>"
    )


# --- datasheet ------------------------------------------------------------------------


def _sheet(entry: LeverEntry) -> str:
    status = entry.status
    weight = _weight(entry)
    classes = f"sheet inked w{min(weight, 3)}" if status == "inked" else f"sheet {status}"
    if status == "inked":
        note = _plural(weight, "independent replication")
    elif status == "struck":
        note = "not supported on held-out cases"
    else:
        note = "not refereed" if entry.verdict is None else "refereed, cannot be inked"
    lineage = f" &#183; revises <code>{_short(entry.parent_id)}</code>" if entry.parent_id else ""
    return (
        f'<article class="{classes}" id="l-{_e(entry.id)}">'
        + (_STRIKE if status == "struck" else "")
        + f'<header><p class="stamp"><b>{_LABEL[status]}</b> {note}</p>'
        f"<h3>{_e(entry.name)}</h3>"
        f'<p class="sub">test <code>{_e(entry.behavior)}</code> &#183; entry '
        f"<code>{_short(entry.id)}</code>{lineage}</p></header>"
        + _site_section(entry)
        + _dial_section(entry)
        + _verdict_section(entry)
        + _replication_section(entry)
        + _legend_section(entry)
        + _recipe_section(entry)
        + _attribution_section(entry)
        + "</article>"
    )


def _site_section(entry: LeverEntry) -> str:
    rows = [
        "<tr>"
        + _td(_e(site.parameter), "l w")
        + _td(_e(site.module), "l w")
        + _td(_e(site.side))
        + _td("not recorded" if site.layer is None else str(site.layer))
        + "</tr>"
        for site in entry.sites
    ]
    kinds = Counter(operation.kind for operation in entry.operations)
    edit = ", ".join(f"{_e(kind)} &#215;{count}" for kind, count in sorted(kinds.items()))
    figures = [
        ("Nominated strength", "none" if entry.strength is None else _num(entry.strength)),
        ("Edit", edit or "none recorded"),
    ]
    if entry.operations:
        edit_digest = _short(operations_digest(entry.operations))
        figures.append(("Edit digest", f"<code>{edit_digest}</code>"))
    return _section(
        "Site",
        "exact tensors",
        _table("", ("Tensor", "Module", "Side", "Layer"), rows, left=(0, 1))
        + _figures(figures, big=True),
    )


def _dial_section(entry: LeverEntry) -> str:
    dial = entry.dial
    if dial is None:
        claimed = (
            " The strength above is claimed without a recorded dial."
            if entry.strength is not None
            else ""
        )
        return _section(
            "Dial",
            "exploratory",
            f"<p>No dial recorded: no dose-response was measured for this entry.{claimed}</p>",
        )
    level = _level(1 - dial.rule.alpha)
    window = dial.window
    key = (
        f'<p class="dkey"><span>{_KEY_TARGET}target effect</span>'
        f"<span>{_KEY_REGRESSION}regression effect</span>"
        + (
            f"<span><i></i>working window {_num(window.low)} to {_num(window.high)}</span>"
            if window is not None
            else ""
        )
        + f"<br>Bands are {level} intervals over cases."
        + ("" if window is not None else " No working window was found.")
        + "</p>"
    )
    truncated = "scored as failures" if dial.truncation == "fail" else "left unscored"
    return _section(
        "Dial",
        "exploratory",
        "<p>Exploratory development evidence. The dial selects a strength; it does not "
        f"establish anything. {_plural(dial.cases, 'development case')}, "
        f"seeds {', '.join(str(seed) for seed in dial.seeds)}; outputs cut off at the token "
        f"limit are {truncated}; runtime <code>{_short(dial.runtime_fingerprint)}</code>.</p>"
        f'<div class="dial"><div>{_dial_svg(dial)}{key}</div><div>{_headline(dial)}</div></div>'
        + _dial_table(dial, level),
    )


def _headline(dial: Dial) -> str:
    point = next((item for item in dial.points if item.strength == dial.nominated), None)
    if point is None:
        return "<p>No strength was nominated from this dial.</p>"
    arrow = " &#8594; "
    selectivity = "not defined" if point.selectivity is None else f"{point.selectivity:,.2f}"
    figures = [
        (
            "Selectivity <small>target gain over the larger of the regression margin and "
            "the regression cost</small>",
            selectivity,
        ),
        ("Targets passing", _pct(dial.baseline_target_rate) + arrow + _pct(point.target_rate)),
        (
            "Regressions passing",
            _pct(dial.baseline_regression_rate) + arrow + _pct(point.regression_rate),
        ),
    ]
    if point.finish_rate is not None:
        before = (
            "not recorded" if dial.baseline_finish_rate is None else _pct(dial.baseline_finish_rate)
        )
        figures.append(("Finish rate", before + arrow + _pct(point.finish_rate)))
    if point.accuracy_when_finished is not None:
        figures.append(("Accuracy when finished, edited", _pct(point.accuracy_when_finished)))
    return (
        f"<p>At the nominated strength {_num(point.strength)}, original{arrow}edited, on "
        "development cases:</p>" + _figures(figures, big=True)
    )


def _dial_table(dial: Dial, level: str) -> str:
    finish = dial.baseline_finish_rate is not None or any(
        point.finish_rate is not None for point in dial.points
    )
    accuracy = any(point.accuracy_when_finished is not None for point in dial.points)
    blank = "&#8212;"
    head = [
        "Strength",
        f"Target effect, pp [{level}]",
        f"Regression effect, pp [{level}]",
        "Targets passing",
        "Regressions passing",
    ]
    original = [
        "original",
        blank,
        blank,
        _pct(dial.baseline_target_rate),
        _pct(dial.baseline_regression_rate),
    ]
    if finish:
        head.append("Finish rate")
        original.append(
            blank if dial.baseline_finish_rate is None else _pct(dial.baseline_finish_rate)
        )
    if accuracy:
        head.append("Accuracy when finished")
        original.append(blank)
    head += ["Selectivity", "Note"]
    original += [blank, "unedited"]

    def row(cells: Sequence[str], opening: str = "<tr>") -> str:
        numbers = "".join(_td(cell) for cell in cells[:-1])
        return opening + numbers + _td(cells[-1], "l n") + "</tr>"

    rows = [row(original)]
    window = dial.window
    for point in dial.points:
        cells = [
            _num(point.strength),
            _effect(point.inference.target),
            _effect(point.inference.regression),
            _pct(point.target_rate),
            _pct(point.regression_rate),
        ]
        if finish:
            cells.append(blank if point.finish_rate is None else _pct(point.finish_rate))
        if accuracy:
            cells.append(
                blank
                if point.accuracy_when_finished is None
                else _pct(point.accuracy_when_finished)
            )
        cells.append(blank if point.selectivity is None else f"{point.selectivity:,.2f}")
        nominated = point.strength == dial.nominated
        notes = ["nominated"] if nominated else []
        if window is not None and window.low <= point.strength <= window.high:
            notes.append("in window")
        cells.append(", ".join(notes) or blank)
        rows.append(row(cells, '<tr class="at">' if nominated else "<tr>"))
    return _table(
        "The dial in numbers: every tested strength against the unedited original, on "
        "development cases. Effects are edited minus original in percentage points.",
        head,
        rows,
        left=(0, len(head) - 1),
        pinned=True,
    )


_W, _H = 420, 240
_LEFT, _RIGHT, _TOP, _BOTTOM = 46, 66, 24, 36


def _ticks(low: float, high: float) -> tuple[int, int, int]:
    """Tick step in tenths of a percentage point, with the first and last tick index."""
    step, first, last = 1000, -1, 1
    for candidate in (10, 20, 50, 100, 200, 250, 500, 1000):
        first = math.floor(round(low * 1000, 6) / candidate)
        last = math.ceil(round(high * 1000, 6) / candidate)
        step = candidate
        if last - first <= 6:
            break
    if first == last:
        first, last = first - 1, last + 1
    return step, first, last


def _dial_svg(dial: Dial) -> str:
    """Target and regression effect against strength, on one shared axis, in graphite."""
    points = dial.points
    width, height = _W - _LEFT - _RIGHT, _H - _TOP - _BOTTOM
    right, floor = _LEFT + width, _TOP + height
    low, span = points[0].strength, points[-1].strength - points[0].strength
    single = len(points) == 1

    def x(strength: float) -> float:
        if span <= 0:
            return _LEFT + width / 2
        return _LEFT + 14 + (strength - low) / span * (width - 28)

    extent = [0.0]
    for point in points:
        for estimate in (point.inference.target, point.inference.regression):
            extent += [estimate.mean_difference, estimate.ci_low, estimate.ci_high]
    step, first, last = _ticks(min(extent), max(extent))
    top_value, value_span = last * step / 1000, (last - first) * step / 1000

    def y(value: float) -> float:
        return _TOP + (top_value - value) / value_span * height

    # Back to front: window, grid, bands, nominated strength, lines, markers, labels, and
    # last an invisible column per strength whose title gives a tooltip without any script.
    xs = [x(point.strength) for point in points]
    parts: list[str] = []
    if dial.window is not None:
        start = max(float(_LEFT), x(dial.window.low) - 9)
        stop = min(float(right), x(dial.window.high) + 9)
        parts.append(
            f'<rect class="wn" x="{start:.1f}" y="{_TOP}" width="{stop - start:.1f}" '
            f'height="{height}"/><path class="wb" d="M{start:.1f} {floor}H{stop:.1f}"/>'
        )
    for index in range(first, last + 1):
        level = y(index * step / 1000)
        tenths = index * step // 10
        parts.append(
            f'<path class="{"zr" if index == 0 else "gr"}" d="M{_LEFT} {level:.1f}H{right}"/>'
            f'<text x="{_LEFT - 6}" y="{level + 3.5:.1f}" text-anchor="end">'
            f"{f'{tenths:+d}' if tenths else '0'}</text>"
        )
    labelled = -100.0
    for position, point in zip(xs, points, strict=True):
        parts.append(f'<path class="tk" d="M{position:.1f} {floor}v4"/>')
        if position - labelled >= 30:
            labelled = position
            parts.append(
                f'<text x="{position:.1f}" y="{floor + 15}" text-anchor="middle">'
                f"{_num(point.strength)}</text>"
            )
    series = (
        ("t", -6.0 if single else 0.0, [point.inference.target for point in points]),
        ("r", 6.0 if single else 0.0, [point.inference.regression for point in points]),
    )
    for _kind, shift, estimates in series:
        if single:
            at, top, bottom = xs[0] + shift, y(estimates[0].ci_high), y(estimates[0].ci_low)
            parts.append(
                f'<path class="wk" d="M{at:.1f} {top:.1f}V{bottom:.1f}M{at - 3:.1f} {top:.1f}h6'
                f'M{at - 3:.1f} {bottom:.1f}h6"/>'
            )
            continue
        upper = [f"{at:.1f},{y(item.ci_high):.1f}" for at, item in zip(xs, estimates, strict=True)]
        lower = [f"{at:.1f},{y(item.ci_low):.1f}" for at, item in zip(xs, estimates, strict=True)]
        parts.append(f'<polygon class="bd" points="{" ".join(upper + lower[::-1])}"/>')
    if dial.nominated is not None:
        at = x(dial.nominated)
        label = min(max(at, 50.0), _W - 50.0)
        parts.append(
            f'<path class="nm" d="M{at:.1f} {_TOP - 3}V{floor}"/>'
            f'<path class="nt" d="M{at - 4:.1f} {_TOP - 9}h8l-4 6z"/>'
            f'<text x="{label:.1f}" y="{_TOP - 13}" text-anchor="middle">'
            f"nominated {_num(dial.nominated)}</text>"
        )
    for kind, _shift, estimates in series:
        if not single:
            line = " ".join(
                f"{at:.1f},{y(item.mean_difference):.1f}"
                for at, item in zip(xs, estimates, strict=True)
            )
            parts.append(f'<polyline class="l{kind}" points="{line}"/>')
    for kind, shift, estimates in reversed(series):
        for at, item in zip(xs, estimates, strict=True):
            centre, level = at + shift, y(item.mean_difference)
            parts.append(
                f'<circle class="mt" cx="{centre:.1f}" cy="{level:.1f}" r="4"/>'
                if kind == "t"
                else f'<rect class="mr" x="{centre - 3.5:.1f}" y="{level - 3.5:.1f}" '
                'width="7" height="7"/>'
            )
    ends = [y(estimates[-1].mean_difference) for _kind, _shift, estimates in series]
    if abs(ends[0] - ends[1]) >= 11:
        edge = xs[-1] + (15 if single else 9)
        parts.append(
            f'<text x="{edge:.1f}" y="{ends[0] + 3.5:.1f}">target</text>'
            f'<text x="{edge:.1f}" y="{ends[1] + 3.5:.1f}">regression</text>'
        )
    middle = _TOP + height / 2
    parts.append(
        f'<text x="{_LEFT + width / 2:.1f}" y="{_H - 4}" text-anchor="middle">strength</text>'
        f'<text x="11" y="{middle:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 11 {middle:.1f})">effect vs original, pp</text>'
    )
    for index, point in enumerate(points):
        start = _LEFT if index == 0 else (xs[index - 1] + xs[index]) / 2
        stop = right if index == len(points) - 1 else (xs[index] + xs[index + 1]) / 2
        parts.append(
            f'<rect class="ht" x="{start:.1f}" y="{_TOP}" width="{stop - start:.1f}" '
            f'height="{height}"><title>strength {_num(point.strength)}: target '
            f"{_effect(point.inference.target)} pp, regression "
            f"{_effect(point.inference.regression)} pp</title></rect>"
        )
    return (
        f'<svg class="plot" viewBox="0 0 {_W} {_H}" role="img" aria-label="Dial: target '
        "effect and regression effect against edit strength, with intervals. The table "
        'below lists the same numbers.">' + "".join(parts) + "</svg>"
    )


def _why(verdict: RefereeVerdict | None) -> str:
    """Why a verdict yields its status, in the sentence of the rule that derives the status.

    The page keeps no copy of that rule. A failed study's error code follows as a recorded
    fact; it explains nothing by itself.
    """
    reason = _e(status_reason(verdict))
    if verdict is not None and verdict.final_error_code is not None:
        code = _e(verdict.final_error_code, 80)
        reason += f" The study ended with error <code>{code}</code>."
    return reason


def _reference(name: str) -> str:
    if name == "baseline":
        return "original, unedited"
    if name == "no_op":
        return "no-op repeat of the original"
    for prefix in ("control", "comparator"):
        if name.startswith(prefix + ":"):
            return f"{prefix} {_e(name[len(prefix) + 1 :], 120)}"
    return _e(name, 120)


def _verdict_section(entry: LeverEntry) -> str:
    verdict = entry.verdict
    if verdict is None:
        return _section(
            "Verdict",
            "held out",
            f'<p class="say">{_why(verdict)} Nothing on this sheet is established.</p>',
        )
    assessment, status = verdict.assessment, entry.status
    if status == "inked":
        consequence = "This verdict inks the entry."
    elif status == "struck":
        consequence = (
            "This verdict strikes the entry, which stays on the map as a tested place that "
            "did not hold."
        )
    else:
        consequence = _why(verdict) + " The entry stays in pencil."
    controls = ", ".join(_e(name, 120) for name in verdict.controls) or "none registered"
    comparators = ", ".join(_e(name, 120) for name in verdict.comparators) or "none registered"
    body = [
        f'<p class="say"><b>{_LABEL[status]}.</b> {_OUTCOME[assessment.verdict]} '
        f"{_RECOMMENDATION[assessment.recommendation]} {consequence}</p>"
        "<p>The candidate had to beat the unedited original and every registered reference. "
        f"Controls: {controls}. Comparators: {comparators}.</p>"
    ]
    rule = verdict.inference_rule
    pairs = assessment.inference[:_MAX_REFERENCES]
    if pairs:
        level = _level(pairs[0].target.confidence)
        rows = []
        for pair in pairs:
            target = pair.target
            beat = "superior" if pair.target_superior else "not superior"
            spared = "non-inferior" if pair.regression_non_inferior else "not established"
            rows.append(
                "<tr>"
                + _td(_reference(pair.reference), "l w")
                + _td(_effect(target))
                + _td(_p(target.sign_test_p))
                + _td(f"{target.improved_cases} / {target.regressed_cases} / {target.tied_cases}")
                + _td("harmed" if pair.target_harmed else beat)
                + _td(_effect(pair.regression))
                + _td("inferior" if pair.regression_inferior else spared)
                + "</tr>"
            )
        body.append(
            _table(
                "Candidate minus each reference on held-out cases, in percentage points, "
                f"with {level} intervals.",
                (
                    "Reference",
                    f"Target effect, pp [{level}]",
                    "Sign test p",
                    "Target cases better / worse / tied",
                    "Targets",
                    f"Regression effect, pp [{level}]",
                    "Regressions",
                ),
                rows,
            )
        )
        if len(assessment.inference) > len(pairs):
            hidden = len(assessment.inference) - len(pairs)
            body.append(f"<p>{_plural(hidden, 'further reference')} not shown.</p>")
        base = next((pair for pair in pairs if pair.reference == "baseline"), pairs[0])
        if base.regression_non_inferior:
            result = "established"
        elif base.regression_inferior:
            result = "refuted, regressions got worse beyond the margin"
        else:
            result = "not established"
        margin = (
            "no registered margin"
            if rule is None
            else (f"registered margin {rule.regression_margin * 100:g} pp")
        )
        body.append(
            f"<p>Regression non-inferiority against {_reference(base.reference)}: <b>{result}"
            f"</b>. Effect {_effect(base.regression)} pp, sign test p "
            f"{_p(base.regression.sign_test_p)}, {margin}.</p>"
        )
    elif assessment.comparisons:
        rows = [
            "<tr>"
            + _td(_reference(item.reference), "l w")
            + _td(f"{item.improved_targets} / {item.regressed_targets}")
            + _td(f"{item.improved_regressions} / {item.regressed_regressions}")
            + "</tr>"
            for item in assessment.comparisons[:_MAX_REFERENCES]
        ]
        body.append(
            _table(
                "Paired counts only. No inference was recorded, so no effect is estimated.",
                ("Reference", "Targets better / worse", "Regressions better / worse"),
                rows,
            )
        )
    else:
        body.append("<p>No paired comparison was recorded.</p>")
    holdout = verdict.holdout
    design = (
        "none registered"
        if rule is None
        else (
            f"alpha {rule.alpha:g}, margin {rule.regression_margin * 100:g} pp, "
            f"{rule.resamples} resamples"
        )
    )
    truncated = "scored as failures" if verdict.truncation == "fail" else "left unscored"
    body.append(
        _figures(
            [
                (
                    "Held-out cases",
                    f"{holdout.count} <small>{holdout.targets} target, "
                    f"{holdout.regressions} regression</small>",
                ),
                ("Development cases", str(verdict.development_cases)),
                ("Seeds", ", ".join(str(seed) for seed in verdict.seeds)),
                ("Inference rule", design),
                ("Truncated outputs", truncated),
                (
                    "Study",
                    f"<code>{_short(verdict.study_id)}</code> "
                    f"<small>finished {_e(verdict.finished_at, 64)}</small>",
                ),
                ("Runtime", f"<code>{_short(verdict.runtime_fingerprint)}</code>"),
                ("Held-out commitment", f"<code>{_short(holdout.digest)}</code>"),
            ]
        )
    )
    if assessment.issues:
        shown = assessment.issues[:_MAX_ISSUES]
        more = len(assessment.issues) - len(shown)
        body.append(
            '<p>Issues recorded by the study:</p><ul class="plain">'
            + "".join(f"<li>{_e(issue, 200)}</li>" for issue in shown)
            + (f"<li>{_plural(more, 'further issue')} not shown.</li>" if more else "")
            + "</ul>"
        )
    return _section("Verdict", "held out", "".join(body))


def _replication_section(entry: LeverEntry) -> str:
    body = []
    verdict = entry.verdict
    if verdict is not None and entry.replications:
        supported, independent = _support(entry)
        rows = []
        for record in entry.replications:
            outcome = verdict_status(record)
            if outcome == "inked":
                result = "supported the edit"
            elif outcome == "struck":
                result = "did not support the edit"
            else:
                result = "does not count. " + _why(record)
            same = record.runtime_fingerprint == verdict.runtime_fingerprint
            rows.append(
                "<tr>"
                + _td(f"<code>{_short(record.study_id)}</code>")
                + _td(
                    f"<code>{_short(record.runtime_fingerprint)}</code> "
                    + ("same as first study" if same else "independent"),
                    "l",
                )
                + _td(
                    f"{record.holdout.count} ({record.holdout.targets} target, "
                    f"{record.holdout.regressions} regression)"
                )
                + _td(str(len(record.seeds)))
                + _td(_e(record.finished_at, 64))
                + _td(result, "l w")
                + "</tr>"
            )
        weighs = entry.status == "inked"
        body.append(
            f"<p>{_plural(len(entry.replications), 'replication')}. {supported} supported the "
            f"edit, on {_plural(independent, 'runtime')} independent of the first study."
            + (" Independent support sets the weight of the ink." if weighs else "")
            + "</p>"
            + _table(
                "",
                ("Study", "Runtime", "Held-out cases", "Seeds", "Finished", "Result"),
                rows,
                left=(0, 1, 5),
            )
        )
    else:
        body.append("<p>No replications recorded.</p>")
    if entry.contraindications:
        items = []
        for item in entry.contraindications:
            extra = []
            if item.base is not None:
                extra.append(
                    f"base <code>{_e(item.base.model_id)}</code> at "
                    f"<code>{_e(item.base.revision, 40)}</code>, weights "
                    f"<code>{_short(item.base.weights_sha256)}</code>"
                )
            if item.evidence_digest is not None:
                extra.append(f"evidence <code>{_short(item.evidence_digest)}</code>")
            items.append(
                f"<li><b>{_e(item.variant)}</b> {_e(item.note)}"
                + (f" <small>{'; '.join(extra)}</small>" if extra else "")
                + "</li>"
            )
        body.append(
            f"<p>Contraindications: {_plural(len(items), 'build')} of this model where the "
            'lever was tried and did not hold.</p><ul class="plain">' + "".join(items) + "</ul>"
        )
    else:
        body.append("<p>No contraindications recorded.</p>")
    return _section("Replications", "and where it failed", "".join(body))


def _legend_section(entry: LeverEntry) -> str:
    legend = entry.legend
    if legend is None:
        return _section(
            "Legend",
            "hypothesis",
            "<p>No legend: this direction has not been compared with an inherited feature "
            "dictionary.</p>",
            extra=" hyp",
        )
    blank = "&#8212;"
    rows = [
        "<tr>"
        + _td(str(item.index))
        + _td(f"{item.cosine:+.3f}")
        + _td("unlabelled" if item.label is None else _e(item.label), "l w")
        + _td(blank if item.label_source is None else _e(item.label_source), "l w")
        + _td(blank if item.url is None else _link(item.url), "l w")
        + "</tr>"
        for item in legend.entries
    ]
    source = legend.source
    return _section(
        "Legend",
        "hypothesis",
        "<p>Pencil. These are the inherited features nearest this lever's direction. Their "
        "labels are other people's names for other people's features, offered as hypotheses "
        "about this site and tested by nothing on this sheet.</p>"
        + _table(
            "",
            ("Feature", "Cosine", "Label, as attributed", "Label source", "Link"),
            rows,
            left=(0, 2, 3, 4),
        )
        + f"<p>The {_plural(len(rows), 'listed feature')} together span "
        f"{_pct(legend.explained)} of the direction's squared norm.</p>"
        f"<p>Dictionary: {_e(source.kind)} <code>{_e(source.repo_id)}</code> at revision "
        f"<code>{_e(source.revision)}</code>, licence {_e(source.license)}. File "
        f"<code>{_e(source.filename)}</code>, tensor <code>{_e(source.tensor)}</code> "
        f"(<code>{_short(source.tensor_sha256)}</code>), layer {source.layer}, writes "
        f"<code>{_e(source.writes)}</code>, {source.latents} latents of width {source.width}. "
        "Inherited, not trained here.</p>",
        extra=" hyp",
    )


def _recipe_section(entry: LeverEntry) -> str:
    derivation = entry.derivation
    if derivation is None:
        return _section(
            "Recipe",
            "re-runnable",
            "<p>No derivation recorded: this site was claimed, not found by a recorded recipe.</p>",
        )
    recipe = derivation.recipe
    rows = []
    for score in derivation.scores:
        chosen = score.site == derivation.chosen
        rows.append(
            ('<tr class="at">' if chosen else "<tr>")
            + _td(_e(score.site.parameter), "l w")
            + _td("not recorded" if score.site.layer is None else str(score.site.layer))
            + _td(f"{score.separation:+.3f}")
            + _td(f"{score.difference_norm:.4g}")
            + _td(str(score.fit_cases))
            + _td(str(score.selection_cases))
            + _td("chosen" if chosen else "not chosen")
            + "</tr>"
        )
    direction = derivation.direction
    shape = " &#215; ".join(str(size) for size in direction.shape)
    return _section(
        "Recipe",
        "re-runnable",
        _figures(
            [
                ("Kind", _e(recipe.kind.replace("_", " "))),
                ("Positive prompts", str(len(recipe.positive))),
                ("Negative prompts", str(len(recipe.negative))),
                ("Site selection", _e(recipe.selection.replace("_", " "))),
                ("Read position", _e(recipe.position.replace("_", " "))),
                ("Thinking", "enabled" if recipe.enable_thinking else "disabled"),
                ("Edit", _e(recipe.edit.replace("_", " "))),
                (
                    "Direction",
                    f"{_e(direction.dtype)}, {shape} <small>sha256 "
                    f"<code>{_short(direction.sha256)}</code></small>",
                ),
                ("Recipe digest", f"<code>{_short(derivation.recipe_digest)}</code>"),
                ("Runtime", f"<code>{_short(derivation.runtime_fingerprint)}</code>"),
            ]
        )
        + _table(
            f"Every candidate site, with the {_plural(len(rows) - 1, 'site')} not chosen. "
            "Separation is measured on prompts held out from the fit. Prompt text is not "
            "part of this page.",
            (
                "Candidate tensor",
                "Layer",
                "Separation",
                "Difference norm",
                "Fit prompts",
                "Selection prompts",
                "Choice",
            ),
            rows,
        ),
    )


def _attribution_section(entry: LeverEntry) -> str:
    if not entry.attribution:
        return ""
    return _section(
        "Attribution",
        "as claimed",
        '<ul class="plain">'
        + "".join(f"<li>{_link(line)}</li>" for line in entry.attribution)
        + "</ul>",
    )


# --- document -------------------------------------------------------------------------


def _document(title: str, body: str) -> str:
    """Wrap ``body``; ``title`` is already escaped. Refuses a page over the size limit."""
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">\n'
        '<meta name="referrer" content="no-referrer">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>{title}</title>\n<style>{_CSS}</style>\n</head>\n"
        f"<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
    )
    if len(page.encode("utf-8")) > MAX_REPORT_BYTES:
        raise _too_large()
    return page


_STRIKE = (
    '<svg class="strike" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">'
    '<path d="M0 100L100 0"/></svg>'
)
_KEY_TARGET = (
    '<svg viewBox="0 0 30 10" aria-hidden="true"><path class="lt" d="M2 5h26"/>'
    '<circle class="mt" cx="15" cy="5" r="4"/></svg>'
)
_KEY_REGRESSION = (
    '<svg viewBox="0 0 30 10" aria-hidden="true"><path class="lr" d="M2 5h26"/>'
    '<rect class="mr" x="11.5" y="1.5" width="7" height="7"/></svg>'
)
# Filtering needs no script: the radios drive :has() selectors, and the control only
# appears where the browser supports them. Every entry is on the page either way.
_FILTER = (
    '<div class="flt" role="group" aria-label="Show entries by status"><span>Show</span>'
    '<label><input type="radio" name="f" id="f-a" checked>all</label>'
    '<label><input type="radio" name="f" id="f-i">inked</label>'
    '<label><input type="radio" name="f" id="f-p">pencil</label>'
    '<label><input type="radio" name="f" id="f-s">struck</label></div>'
)
_FOOTER = (
    "<footer>Offline page; it loads nothing. Counts and aggregates only: prompts, expected "
    "answers, case content and weight values are not included. A dial is exploratory "
    "development evidence and selects a strength. Only a registered held-out verdict inks "
    "an entry, and a struck entry stays on the map.</footer>"
)

# One line of ASCII, so that no encoding or newline translation can change its hash.
_CSS = "".join(
    (
        ":root{--paper:#f2efe6;--sheet:#faf8f3;--ink:#17202c;--graphite:#555a62;",
        "--rule:#cfcabb;--accent:#9a4a12;--wash:rgb(154 74 18/.16);",
        '--serif:"Iowan Old Style","Palatino Linotype",Palatino,Charter,"Sitka Text",',
        "Cambria,Georgia,serif;",
        '--mono:ui-monospace,"SFMono-Regular","Cascadia Mono",Consolas,"Liberation Mono",',
        "Menlo,monospace}",
        "@media(prefers-color-scheme:dark){:root{--paper:#1b2129;--sheet:#232a33;",
        "--ink:#ece7da;--graphite:#aab0b9;--rule:#414a56;--accent:#e0a460;",
        "--wash:rgb(224 164 96/.2)}}",
        "@media print{:root{--paper:#fff;--sheet:#fff;--ink:#000;--graphite:#444;",
        "--rule:#bbb;--accent:#8a3d08;--wash:rgb(138 61 8/.16)}.flt{display:none!important}",
        ".sec{break-inside:avoid}}",
        "*,*::before,*::after{box-sizing:border-box}",
        "html{-webkit-text-size-adjust:100%;text-size-adjust:100%}",
        "@media(prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}",
        "body{margin:0;background:var(--paper);color:var(--ink);font:1rem/1.55 var(--serif)}",
        "main{max-width:70rem;margin:0 auto;padding:2rem 1rem 4rem}",
        "h1{font:600 clamp(1.7rem,5vw,2.5rem)/1.15 var(--serif);margin:0 0 .5rem;",
        "overflow-wrap:anywhere}",
        "h2{font:600 .75rem/1.3 var(--mono);letter-spacing:.14em;text-transform:uppercase;",
        "margin:2.5rem 0 .5rem;padding-top:.75rem;border-top:1px solid var(--ink)}",
        "p{margin:.5rem 0}.lede{max-width:44rem;margin-bottom:1rem}",
        "small{font-size:.85em;color:var(--graphite)}",
        "code{font:.85em var(--mono);overflow-wrap:anywhere}",
        "a{color:inherit;text-decoration-thickness:1px;text-underline-offset:2px}",
        "a:focus-visible,input:focus-visible{outline:2px solid var(--ink);outline-offset:2px}",
        # base
        ".base{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,13rem),1fr));",
        "gap:.5rem 1.5rem;margin:.75rem 0;padding:.75rem 0;border-top:1px solid var(--ink);",
        "border-bottom:1px solid var(--rule)}.base div{min-width:0}",
        "h2+.base{border-top:0;margin-top:0;padding-top:0}",
        ".base dt{font-size:.8rem;color:var(--graphite)}.base dd{margin:0;overflow-wrap:anywhere}",
        # status marks: dashed pencil, solid ink that thickens, struck with a slash
        ".mk{position:relative;display:inline-grid;place-items:center;width:24px;height:24px;",
        "vertical-align:middle}",
        '.mk::before{content:"";width:14px;height:14px;border:1.5px dashed var(--graphite)}',
        ".mk.inked::before{border:1.5px solid var(--ink)}.mk.w1::before{border-width:2.75px}",
        ".mk.w2::before{border-width:4px}.mk.w3::before{border-width:5.5px}",
        ".mk.struck::before{border-style:solid}",
        '.mk.struck::after{content:"";position:absolute;left:1px;right:1px;top:50%;',
        "margin-top:-.75px;border-top:1.5px solid var(--graphite);transform:rotate(-45deg)}",
        ".tally{list-style:none;margin:1rem 0 .5rem;padding:0;display:flex;flex-wrap:wrap;",
        "gap:.25rem 1.5rem}.tally.key{display:grid;gap:.15rem}",
        ".tally li{display:grid;grid-template-columns:24px auto minmax(0,1fr);gap:.5rem;",
        "align-items:center}.tally.key li{grid-template-columns:24px 2.25rem minmax(0,1fr)}",
        ".tally b{font:600 1.25rem/1.2 var(--mono);text-align:right}",
        ".bar{display:block;width:100%;max-width:36rem;height:auto;margin:.6rem 0}",
        ".bi{fill:var(--ink)}.bp,.bs{fill:none;stroke:var(--graphite);stroke-width:1.5}",
        ".bp{stroke-dasharray:4 3}",
        ".wts{font-size:.85rem;color:var(--graphite)}.wts .mk{margin-left:.35rem}",
        # filter
        ".flt{display:none;flex-wrap:wrap;gap:.25rem 1.25rem;align-items:center;",
        "margin:1.5rem 0 0;font-size:.9rem}.flt span{color:var(--graphite)}",
        ".flt label{display:inline-flex;align-items:center;gap:.4rem;min-height:24px}",
        ".flt input{margin:0;accent-color:var(--ink)}",
        "@supports selector(:has(*)){.flt{display:flex}",
        "body:has(#f-i:checked) .sheet:not(.inked),body:has(#f-p:checked) .sheet:not(.pencil),",
        "body:has(#f-s:checked) .sheet:not(.struck){display:none}",
        "body:has(#f-i:checked) .strip .mk:not(.inked),",
        "body:has(#f-p:checked) .strip .mk:not(.pencil),",
        "body:has(#f-s:checked) .strip .mk:not(.struck){opacity:.2;pointer-events:none}}",
        # terrain: solid cells hold a tested site, dotted cells are unsurveyed
        ".strip{list-style:none;margin:.75rem 0 0;padding:0;display:grid;gap:4px;",
        "grid-template-columns:repeat(auto-fill,minmax(3.5rem,1fr))}",
        ".strip li{min-height:3.6rem;padding:2px 3px;border:1px solid var(--ink)}",
        ".strip li.e{border:1px dotted var(--graphite)}.strip li.g{grid-column:span 2}",
        ".strip li.x{grid-column:1/-1}",
        ".strip b,.strip span{display:block;font:400 .68rem/1.5 var(--mono);",
        "color:var(--graphite)}",
        # datasheet
        ".sheet{position:relative;background:var(--sheet);border:1.5px dashed var(--graphite);",
        "padding:1.25rem 1rem;margin:1.5rem 0;scroll-margin-top:1rem}",
        ".sheet.inked{border:1.5px solid var(--ink)}.sheet.w1{border-width:2.5px}",
        ".sheet.w2{border-width:3.5px}.sheet.w3{border-width:4.5px}",
        ".sheet.struck{border-style:solid}",
        ".sheet:target{outline:1px solid var(--ink);outline-offset:5px}",
        ".strike{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}",
        ".strike path{fill:none;stroke:var(--graphite);stroke-width:1.25;",
        "vector-effect:non-scaling-stroke}",
        ".stamp{display:inline-block;margin:0;padding:.3rem .55rem;font:.72rem/1.3 var(--mono);",
        "border:1.5px dashed var(--graphite);color:var(--graphite)}",
        ".stamp b{letter-spacing:.14em;text-transform:uppercase;margin-right:.25rem}",
        ".inked .stamp{border:2px solid var(--ink);color:var(--ink)}",
        ".struck .stamp{border-style:solid}",
        ".sheet h3{font:400 1.05rem/1.35 var(--mono);margin:.7rem 0 .2rem;",
        "overflow-wrap:anywhere;color:var(--graphite)}",
        ".sheet.inked h3{font-weight:700;color:var(--ink)}",
        ".sheet.struck h3{text-decoration:line-through;text-decoration-thickness:1.5px}",
        ".sub{margin:0;font-size:.85rem;color:var(--graphite)}",
        ".sec{border-top:1px solid var(--rule);margin-top:1rem;padding-top:.9rem}",
        ".sec>div{min-width:0}",
        ".sec h4{font:600 .72rem/1.3 var(--mono);letter-spacing:.12em;text-transform:uppercase;",
        "margin:0 0 .5rem}",
        ".sec h4 small{display:block;font-size:1em;font-weight:400;letter-spacing:.02em;",
        "text-transform:none}",
        ".sec>div>p:first-child{margin-top:0}",
        "@media(min-width:60rem){.sheet{padding:1.5rem}",
        ".sec{display:grid;grid-template-columns:9rem minmax(0,1fr);gap:0 1.5rem}}",
        ".say{font-size:1.05rem}.inked .say{font-weight:600}",
        ".figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(8.5rem,1fr));",
        "gap:.6rem 1.25rem;margin:.6rem 0}.figs div{min-width:0}",
        ".figs dt{font-size:.8rem;line-height:1.3;color:var(--graphite)}",
        ".figs dt small{display:block;font-size:.9em}",
        ".figs dd{margin:0;font:.9rem/1.4 var(--mono);overflow-wrap:anywhere}",
        ".figs.big dd{font-size:1.15rem}",
        ".scroll{overflow-x:auto;margin:.6rem 0}",
        "table{border-collapse:collapse;width:100%;font:.78rem/1.4 var(--mono);",
        "font-variant-numeric:tabular-nums}",
        "figure{margin:.9rem 0 .6rem}figure .scroll{margin:.2rem 0 0}",
        "figcaption{font:italic .85rem/1.4 var(--serif);color:var(--graphite)}",
        ".pin th:first-child,.pin td:first-child{position:sticky;left:0;",
        "background:var(--sheet)}",
        "th,td{padding:.3rem .9rem .3rem 0;border-bottom:1px solid var(--rule);",
        "text-align:right;white-space:nowrap;vertical-align:top}",
        "th{font:600 .75rem/1.3 var(--serif);color:var(--graphite);vertical-align:bottom;",
        "white-space:normal}",
        "th.l,td.l,td:first-child{text-align:left}",
        "td.w{white-space:normal;overflow-wrap:anywhere;min-width:11rem}",
        "td.n{white-space:normal;min-width:5.5rem}",
        "tr.at td{font-weight:700;border-bottom-color:var(--graphite)}",
        ".plain{list-style:none;margin:.5rem 0;padding:0}",
        ".plain li{padding:.3rem 0;border-bottom:1px solid var(--rule);overflow-wrap:anywhere}",
        # the legend is always pencil
        ".hyp>div{border:1.5px dashed var(--graphite);padding:.7rem .8rem;",
        "color:var(--graphite)}.hyp td.w{font:italic .85rem/1.4 var(--serif)}",
        # dial: graphite throughout, the accent only for the working window
        ".dial{display:grid;gap:.5rem 2rem;align-items:start;",
        "grid-template-columns:repeat(auto-fit,minmax(min(100%,18rem),1fr))}",
        ".dial>div{min-width:0}",
        ".plot{display:block;width:100%;max-width:34rem;height:auto}",
        ".plot text{font:10px var(--mono);fill:var(--graphite)}",
        ".wn{fill:var(--wash)}.wb{fill:none;stroke:var(--accent);stroke-width:3}",
        ".gr{fill:none;stroke:var(--rule)}.zr,.tk{fill:none;stroke:var(--graphite)}",
        ".bd{fill:var(--graphite);fill-opacity:.14}",
        ".lt,.lr{fill:none;stroke:var(--graphite);stroke-width:2;stroke-linejoin:round}",
        ".lt{stroke-linecap:round}.lr{stroke-dasharray:5 3}",
        ".mt{fill:var(--graphite);stroke:var(--sheet);stroke-width:2}",
        ".mr{fill:var(--sheet);stroke:var(--graphite);stroke-width:1.75}",
        ".wk,.nm{fill:none;stroke:var(--graphite);stroke-width:1.25}.nt{fill:var(--graphite)}",
        ".ht{fill:none;pointer-events:all}",
        ".dkey{font-size:.8rem;color:var(--graphite)}",
        ".dkey span{display:inline-block;white-space:nowrap;margin-right:.9rem}",
        ".dkey svg{width:30px;height:10px;vertical-align:middle;margin-right:.35rem}",
        ".dkey i{display:inline-block;width:16px;height:11px;vertical-align:middle;",
        "margin-right:.35rem;background:var(--wash);border-bottom:3px solid var(--accent)}",
        "footer{max-width:44rem;margin-top:3rem;padding-top:1rem;font-size:.85rem;",
        "border-top:1px solid var(--ink);color:var(--graphite)}",
    )
)
_CSP = (
    "default-src 'none'; style-src 'sha256-"
    + base64.b64encode(hashlib.sha256(_CSS.encode("ascii")).digest()).decode("ascii")
    + "'; base-uri 'none'; form-action 'none'"
)
