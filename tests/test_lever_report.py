"""The lever map page: honest statuses, inert untrusted text, aggregates only, offline."""

from __future__ import annotations

import base64
import hashlib
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path

import pytest

from nrly.artifacts import TensorArtifactStore
from nrly.contracts import (
    CaseSpec,
    GenerationSpec,
    NrlyError,
    PortableBase,
    WeightOperationSpec,
    digest,
)
from nrly.lever_contracts import (
    AtlasSource,
    Contraindication,
    Derivation,
    Dial,
    DosePoint,
    Legend,
    LegendEntry,
    LeverEntry,
    LeverSite,
    RecipeSpec,
    RefereeVerdict,
    SiteScore,
    StrengthWindow,
    lever_name,
    operations_digest,
    seal,
    status_reason,
)
from nrly.lever_report import MAX_ENTRIES, MAX_REPORT_BYTES, render_lever, render_map
from nrly.study_contracts import (
    EffectEstimate,
    HoldoutCommitment,
    InferenceRule,
    PairedInference,
    StudyAssessment,
)

WEIGHTS = "ab12cd34ef56" + "0" * 52
BASE = PortableBase(
    backend="huggingface", model_id="org/model-4b", revision="c" * 40, dtype="bfloat16",
    weights_sha256=WEIGHTS,
)  # fmt: skip
RUNTIME, OTHER_RUNTIME, THIRD_RUNTIME = "1" * 64, "2" * 64, "3" * 64
PROMPT_SECRET = "SECRET-PROMPT-zebra-quartz"
ANSWER_SECRET = "SECRET-ANSWER-walrus-onyx"
PAYLOAD_SECRET = 0.123456789
ARROW = chr(0x2192)

VOID = {"meta", "br", "input"}
TAGS = {
    "html", "head", "meta", "title", "style", "body", "main", "header", "footer", "section",
    "article", "h1", "h2", "h3", "h4", "p", "dl", "dt", "dd", "div", "ul", "ol", "li", "table",
    "figure", "figcaption", "thead", "tbody", "tr", "th", "td", "code", "small", "b", "i", "a",
    "span", "br",
    "label", "input", "svg", "path", "rect", "circle", "polygon", "polyline", "text",
}  # fmt: skip
ATTRIBUTES = {
    "lang", "charset", "name", "content", "http-equiv", "class", "id", "href", "rel", "title",
    "type", "checked", "role", "aria-label", "aria-hidden", "scope", "viewbox",
    "preserveaspectratio", "d", "x", "y", "width", "height", "cx", "cy", "r", "points",
    "text-anchor", "transform",
}  # fmt: skip


class Page(HTMLParser):
    """Strict reading of a rendered page: every element must be closed in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.open: list[str] = []
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.words: list[str] = []
        self.rows: list[list[str]] = []
        self.cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        assert len(dict(attrs)) == len(attrs), f"duplicate attribute on <{tag}>"
        self.elements.append((tag, dict(attrs)))
        if tag not in VOID:
            self.open.append(tag)
        if tag == "tr":
            self.rows.append([])
        if tag in {"td", "th"}:
            self.rows[-1].append("")
            self.cell = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))

    def handle_endtag(self, tag: str) -> None:
        assert self.open and self.open[-1] == tag, f"unbalanced </{tag}> inside {self.open[-3:]}"
        self.open.pop()
        if tag in {"td", "th"}:
            self.cell = False

    def handle_data(self, data: str) -> None:
        self.words.append(data)
        if self.cell:
            self.rows[-1][-1] += data

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.words).split())

    def classed(self, name: str) -> list[tuple[str, dict[str, str | None]]]:
        return [
            (tag, attrs)
            for tag, attrs in self.elements
            if name in (attrs.get("class") or "").split()
        ]

    def tags(self, name: str) -> list[dict[str, str | None]]:
        return [attrs for tag, attrs in self.elements if tag == name]


def parse(page: str) -> Page:
    parser = Page()
    parser.feed(page)
    parser.close()
    assert not parser.open, f"unclosed elements: {parser.open}"
    return parser


def drawings(page: str) -> list[ElementTree.Element]:
    found = re.findall(r"<svg\b.*?</svg>", page, flags=re.DOTALL)
    assert page.count("<svg") == len(found)
    return [ElementTree.fromstring(item) for item in found]


def site(layer: int | None = 7, parameter: str | None = None) -> LeverSite:
    name = parameter or f"model.layers.{layer}.mlp.down_proj.weight"
    return LeverSite(parameter=name, module=f"model.layers.{layer}.mlp", layer=layer)


def operation(target: str) -> WeightOperationSpec:
    return WeightOperationSpec(
        kind="weight_low_rank", target=target, shape=[1, 2], rank=1, left=[1.0],
        right=[PAYLOAD_SECRET, 0.5],
    )  # fmt: skip


def estimate(mean: float, low: float, high: float, *, p: float = 0.0215) -> EffectEstimate:
    return EffectEstimate(
        cases=12, seeds=2, mean_difference=mean, ci_low=low, ci_high=high, confidence=0.95,
        improved_cases=9, regressed_cases=1, tied_cases=2, sign_test_p=p,
    )  # fmt: skip


def pair(reference: str, mean: float = 0.5, *, superior: bool = True) -> PairedInference:
    return PairedInference(
        reference=reference,
        target=estimate(mean, mean - 0.2, mean + 0.2, p=0.0215 if superior else 0.7539),
        regression=estimate(-0.01, -0.015, 0.02, p=1.0),
        target_superior=superior,
        target_harmed=False,
        regression_non_inferior=True,
        regression_inferior=False,
    )


def verdict(
    operations: Sequence[WeightOperationSpec],
    *,
    base: PortableBase = BASE,
    supported: bool = True,
    runtime: str = RUNTIME,
    scope: str = "real_model",
    controls: Sequence[str] = ("random-direction",),
    comparators: Sequence[str] = ("system-prompt",),
    study: int = 1,
    error: str | None = None,
    issues: Sequence[str] = (),
    finished: str = "2026-11-18T09:30:00Z",
) -> RefereeVerdict:
    references = [
        "baseline",
        "no_op",
        *("control:" + name for name in controls),
        *("comparator:" + name for name in comparators),
    ]
    return RefereeVerdict(
        base=base,
        study_id=f"{study:02x}" * 16,
        study_spec_digest="5" * 64,
        holdout=HoldoutCommitment(digest="6" * 64, count=48, targets=26, regressions=22),
        operations_digest=operations_digest(list(operations)),
        nominee_digest="7" * 64,
        runtime_fingerprint=runtime,
        inference_rule=InferenceRule(),
        seeds=[11, 12, 13],
        controls=list(controls),
        comparators=list(comparators),
        development_cases=40,
        assessment=StudyAssessment(
            scorer_version="nrly-scorers/3",
            verdict="observed_improvement" if supported else "no_change",
            recommendation="candidate_supported" if supported else "retain_original",
            scope=scope,
            inference=[pair(name, superior=supported) for name in references],
            issues=list(issues),
        ),
        final_error_code=error,
        finished_at=finished,
    )


def dial(strengths: Sequence[float], nominated: float | None, window: bool = True) -> Dial:
    points = [
        DosePoint(
            strength=strength,
            operations_digest="8" * 64,
            inference=PairedInference(
                reference="baseline",
                target=estimate(0.1 * index + 0.2, 0.1 * index, 0.1 * index + 0.4),
                regression=estimate(-0.05 * index, -0.05 * index - 0.1, 0.05),
                target_superior=True,
                target_harmed=False,
                regression_non_inferior=True,
                regression_inferior=False,
            ),
            target_rate=min(1.0, 0.45 + 0.1 * index),
            regression_rate=1.0 - 0.05 * index,
            finish_rate=0.9,
            accuracy_when_finished=0.875,
            selectivity=3.75 + index,
        )
        for index, strength in enumerate(strengths)
    ]
    return Dial(
        rule=InferenceRule(),
        generation=GenerationSpec(),
        seeds=[1, 2],
        cases=24,
        runtime_fingerprint=RUNTIME,
        baseline_target_rate=0.25,
        baseline_regression_rate=1.0,
        baseline_finish_rate=0.8,
        points=points,
        window=(
            StrengthWindow(low=strengths[0], high=strengths[-1], points=len(strengths))
            if window
            else None
        ),
        nominated=nominated,
    )


def derivation(
    root: Path, sites: Sequence[LeverSite], behavior: str = "refusal", base: PortableBase = BASE
) -> Derivation:
    def prompts(kind: str) -> list[CaseSpec]:
        return [
            CaseSpec(
                id=f"contrast-{kind}-{index}",
                input=f"{PROMPT_SECRET} {kind} {index}",
                expected=f"{ANSWER_SECRET} {index}",
            )
            for index in range(5)
        ]

    recipe = RecipeSpec(
        behavior=behavior, positive=prompts("pos"), negative=prompts("neg"), sites=list(sites)
    )
    return Derivation(
        recipe=recipe,
        recipe_digest=digest(recipe),
        base=base,
        runtime_fingerprint=RUNTIME,
        chosen=sites[0],
        direction=TensorArtifactStore(root).put([1.0, 0.0, 0.0]),
        scores=[
            SiteScore(
                site=candidate,
                separation=2.5 - index,
                difference_norm=1.25,
                fit_cases=5,
                selection_cases=5,
            )  # fmt: skip
            for index, candidate in enumerate(sites)
        ],
    )


def legend(entries: Sequence[LegendEntry]) -> Legend:
    return Legend(
        source=AtlasSource(
            kind="transcoder",
            repo_id="atlas-org/transcoders",
            revision="d" * 40,
            filename="layer_7/params.safetensors",
            tensor_sha256="9" * 64,
            license="CC-BY-4.0",
            layer=7,
            writes="model.layers.7.mlp",
            latents=16384,
            width=2560,
        ),  # fmt: skip
        entries=list(entries),
        explained=0.4375,
    )


def lever(
    *,
    sites: Sequence[LeverSite] | None = None,
    behavior: str = "refusal",
    base: PortableBase = BASE,
    strengths: Sequence[float] | None = (0.25, 0.5, 1.0),
    nominated: float | None = 0.5,
    **content: object,
) -> LeverEntry:
    chosen = list(sites) if sites is not None else [site()]
    operations = [operation(item.parameter) for item in chosen]
    data: dict[str, object] = {
        "name": lever_name(behavior, chosen),
        "behavior": behavior,
        "base": base,
        "sites": chosen,
        "operations": operations,
    }
    if strengths is not None:
        data["dial"] = dial(strengths, nominated)
        data["strength"] = nominated
    data.update(content)
    return seal(data)


def inked(layer: int = 7, **content: object) -> LeverEntry:
    chosen = [site(layer)]
    operations = [operation(chosen[0].parameter)]
    return lever(sites=chosen, verdict=verdict(operations), **content)


def struck(layer: int = 3) -> LeverEntry:
    chosen = [site(layer)]
    operations = [operation(chosen[0].parameter)]
    return lever(sites=chosen, verdict=verdict(operations, supported=False))


def test_three_statuses_have_distinct_labels_classes_and_counts() -> None:
    entries = [inked(7), lever(sites=[site(5)]), lever(sites=[site(9)]), struck(3)]
    assert [entry.status for entry in entries] == ["inked", "pencil", "pencil", "struck"]
    page = render_map(entries, layers=12)
    parsed = parse(page)

    sheets = [attrs["class"] for tag, attrs in parsed.classed("sheet") if tag == "article"]
    assert sheets == ["sheet struck", "sheet pencil", "sheet inked w0", "sheet pencil"]
    assert len({attrs["id"] for tag, attrs in parsed.classed("sheet")}) == 4
    stamps = re.findall(r'<p class="stamp"><b>(\w+)</b> ([^<]*)</p>', page)
    assert stamps == [
        ("Struck", "not supported on held-out cases"),
        ("Pencil", "not refereed"),
        ("Inked", "0 independent replications"),
        ("Pencil", "not refereed"),
    ]
    tally = re.findall(r'<li><i class="mk (\w+)"></i><b>(\d+)</b><span>(\w+)', page)
    assert tally == [("inked", "1", "inked"), ("pencil", "2", "pencil"), ("struck", "1", "struck")]
    # Only a struck sheet carries the strike, and it keeps every section beneath it.
    assert len(parsed.classed("strike")) == 1
    alone = parse(render_lever(entries[3]))
    assert len(alone.classed("strike")) == 1 and len(alone.classed("sec")) == 6
    assert "This verdict strikes the entry" in alone.text
    assert "+50.0 [+30.0, +70.0]" in alone.text


def test_header_states_the_base_and_groups_several_bases() -> None:
    other = BASE.model_copy(update={"model_id": "org/model-12b", "weights_sha256": "f" * 64})
    page = render_map([inked(7), lever(sites=[site(2)], base=other)], layers=12)
    parsed = parse(page)
    assert "org/model-4b" in parsed.text and "org/model-12b" in parsed.text
    assert "c" * 40 in parsed.text
    assert WEIGHTS[:12] in parsed.text and "f" * 12 in parsed.text
    assert WEIGHTS not in page and "f" * 13 not in page
    assert len(parsed.classed("base")) == 2 and len(parsed.classed("terrain")) == 2
    assert "2 exact checkpoints, grouped by base" in parsed.text

    single = parse(render_map([inked(7)], layers=12))
    assert len(single.classed("base")) == 1 and "one exact checkpoint" in single.text
    empty = parse(render_map([]))
    assert "No entries" in empty.text and not empty.classed("sheet")


def test_terrain_places_every_lever_and_leaves_unsurveyed_layers_empty() -> None:
    two_sites = lever(sites=[site(1), site(4)], behavior="spans")
    unplaced = lever(sites=[site(None, "embed.weight")], behavior="nowhere")
    entries = [inked(7), struck(3), two_sites, unplaced]
    parsed = parse(render_map(entries, layers=12))
    cells = [attrs.get("class") for attrs in parsed.tags("li") if attrs.get("class") != "mk"]
    strip = cells[3:]  # after the three tally rows
    assert strip == ["e", None, "e", None, None, "e", "e", None, "e", "e", "e", "e", "g"]
    marks = [(attrs["class"], attrs["href"]) for tag, attrs in parsed.classed("mk") if tag == "a"]
    assert sorted(marks) == sorted(
        [
            ("mk pencil", f"#l-{two_sites.id}"),
            ("mk struck", f"#l-{entries[1].id}"),
            ("mk pencil", f"#l-{two_sites.id}"),
            ("mk inked w0", f"#l-{entries[0].id}"),
            ("mk pencil", f"#l-{unplaced.id}"),
        ]
    )
    assert {href[1:] for _class, href in marks} == {
        attrs["id"] for _tag, attrs in parsed.classed("sheet")
    }
    assert "4 of 12 hold a tested site; 8 are unsurveyed" in parsed.text
    assert "records no layer, last cell: 1." in parsed.text

    unknown = parse(render_map(entries))
    assert "depth was not supplied" in unknown.text and "4 of 8 hold" in unknown.text
    beyond = parse(render_map(entries, layers=6))
    assert "Tested layers beyond the stated depth of 6: 1." in beyond.text

    # A crowded layer takes a row of its own rather than stretching its neighbours.
    crowd = [lever(sites=[site(6, f"model.layers.6.head{index}.weight")]) for index in range(9)]
    crowded = parse(render_map(crowd, layers=8))
    assert [attrs.get("class") for attrs in crowded.tags("li")][3:] == ["e"] * 6 + ["x", "e"]
    assert len([tag for tag, _attrs in crowded.classed("mk") if tag == "a"]) == 9

    deep = parse(render_map([inked(7), struck(300)], layers=400))
    assert "300 unsurveyed" not in deep.text
    assert "292 unsurveyed" in deep.text and "99 unsurveyed" in deep.text
    assert len(deep.tags("li")) < 40


def test_ink_weight_counts_only_independent_supporting_replications() -> None:
    chosen = [site(7)]
    operations = [operation(chosen[0].parameter)]
    replications = [
        verdict(operations, study=2),  # same runtime as the first study
        verdict(operations, study=3, runtime=OTHER_RUNTIME),
        verdict(operations, study=4, runtime=OTHER_RUNTIME),  # that runtime again
        verdict(operations, study=5, runtime=THIRD_RUNTIME, supported=False),
        verdict(operations, study=6, runtime="4" * 64, scope="engineering_fixture"),
    ]
    entry = lever(sites=chosen, verdict=verdict(operations), replications=replications)
    page = render_map([entry], layers=12)
    parsed = parse(page)
    assert parsed.classed("sheet")[0][1]["class"] == "sheet inked w1"
    assert [attrs["class"] for tag, attrs in parsed.classed("mk") if tag == "a"] == ["mk inked w1"]
    assert "5 replications. 3 supported the edit, on 1 runtime independent" in parsed.text
    assert "1 independent replication" in page and "1 independent replications" not in page
    studies = {row[0]: row for row in parsed.rows if len(row) == 6 and row[0] != "Study"}
    assert list(studies) == [f"{number:02x}" * 6 for number in range(2, 7)]
    assert "independent" in studies["03" * 6][1] and "same as first study" in studies["02" * 6][1]
    results = [row[-1] for row in studies.values()]
    assert results[:4] == [
        "supported the edit",
        "supported the edit",
        "supported the edit",
        "did not support the edit",
    ]
    assert results[4] == "does not count. " + status_reason(replications[4])

    heavy = lever(
        sites=chosen,
        verdict=verdict(operations),
        replications=[
            verdict(operations, study=10 + index, runtime=f"{index + 5:x}" * 64)
            for index in range(5)
        ],
    )
    assert 'class="sheet inked w3"' in render_lever(heavy)


def test_verdict_names_what_was_beaten_and_reports_the_held_out_numbers() -> None:
    parsed = parse(render_lever(inked(7)))
    text = parsed.text
    assert "Controls: random-direction. Comparators: system-prompt." in text
    rows = {row[0]: row for row in parsed.rows if len(row) == 7 and row[0] != "Reference"}
    assert list(rows) == [
        "original, unedited",
        "no-op repeat of the original",
        "control random-direction",
        "comparator system-prompt",
    ]
    assert rows["control random-direction"][1:] == [
        "+50.0 [+30.0, +70.0]", "0.0215", "9 / 1 / 2", "superior", "-1.0 [-1.5, +2.0]",
        "non-inferior",
    ]  # fmt: skip
    assert "Regression non-inferiority against original, unedited: established" in text
    assert "registered margin 2 pp" in text
    assert "48 26 target, 22 regression" in text
    assert "11, 12, 13" in text
    assert "This verdict inks the entry." in text
    assert "It supports the candidate." in text


def test_studies_that_cannot_ink_leave_pencil_and_say_why() -> None:
    chosen = [site(7)]
    operations = [operation(chosen[0].parameter)]
    fixture = lever(sites=chosen, verdict=verdict(operations, scope="engineering_fixture"))
    failed = lever(sites=chosen, verdict=verdict(operations, error="backend_closed"))
    no_control = lever(sites=chosen, verdict=verdict(operations, controls=()))
    unrefereed = lever(sites=chosen)
    entries = (fixture, failed, no_control, unrefereed)
    assert {entry.status for entry in entries} == {"pencil"}
    # Each explanation is the sentence of the rule that derived the status, not a copy of it.
    reasons = [status_reason(entry.verdict) for entry in entries]
    assert all(reasons) and len(set(reasons)) == len(entries)
    for entry, reason in zip(entries, reasons, strict=True):
        for page in (render_lever(entry), render_map([entry], layers=12)):
            text = parse(page).text
            assert [other for other in reasons if other in text] == [reason]
            assert 'class="sheet pencil"' in page

    for entry, reason in ((fixture, reasons[0]), (no_control, reasons[2])):
        page = render_lever(entry)
        text = parse(page).text
        # The study itself supported the candidate, and the page still says why that is not ink.
        assert f"It supports the candidate. {reason} The entry stays in pencil." in text
        assert "refereed, cannot be inked" in page
    assert "Controls: none registered" in parse(render_lever(no_control)).text
    page = render_lever(failed)
    assert "refereed, cannot be inked" in page
    # The recorded error code follows the rule's sentence as a fact.
    assert f"{reasons[1]} The study ended with error" in parse(page).text
    assert "ended with error <code>backend_closed</code>. The entry stays in pencil." in page
    page = render_lever(unrefereed)
    assert '<p class="stamp"><b>Pencil</b> not refereed</p>' in page
    assert f"{reasons[3]} Nothing on this sheet is established." in parse(page).text


def test_a_hand_altered_status_is_refused() -> None:
    forged = lever().model_copy(update={"status": "inked"})
    with pytest.raises(NrlyError) as error:
        render_map([forged])
    assert error.value.code == "invalid_request"
    with pytest.raises(NrlyError):
        render_lever(forged)


def test_hostile_text_is_escaped_and_never_becomes_markup(tmp_path: Path) -> None:
    hostile = [
        "<script>alert(1)</script>",
        '" onmouseover="alert(2)',
        "'><img src=x onerror=alert(3)>",
        "</td></tr></table><script>alert(4)</script>",
        "</style><style>.sheet{display:none}</style>",
        "</title><iframe src=//evil.example>",
        '<a href="javascript:alert(5)">meaning</a>',
        "&lt;already&gt; &amp; <b onclick=alert(6)>bold</b>",
    ]
    base = BASE.model_copy(update={"model_id": "org/" + hostile[0], "revision": hostile[1]})
    chosen = [
        LeverSite(parameter="enc." + hostile[0] + hostile[1], module=hostile[3], layer=2),
        LeverSite(parameter=hostile[2], module=hostile[4], layer=3),
    ]
    operations = [operation(item.parameter) for item in chosen]
    entry = lever(
        sites=chosen,
        base=base,
        derivation=derivation(tmp_path, chosen, base=base),
        legend=legend(
            [
                LegendEntry(index=1, cosine=0.9, label=hostile[3], label_source=hostile[1]),
                LegendEntry(index=2, cosine=-0.5, label=hostile[6], label_source=hostile[2]),
            ]
        ),
        verdict=verdict(
            operations,
            base=base,
            supported=False,
            controls=(hostile[0], hostile[7]),
            comparators=(hostile[1], hostile[2]),
            issues=(hostile[3], hostile[4]),
            finished=hostile[5],
        ),
        contraindications=[Contraindication(variant=hostile[2], note=hostile[4], base=base)],
        attribution=hostile,
    )
    failed = lever(
        sites=[site(8)],
        base=base,
        verdict=verdict(
            [operation(site(8).parameter)], base=base, error=hostile[0], controls=(hostile[5],)
        ),
    )
    pages = [
        render_map([entry, failed], title=hostile[5] + hostile[0], layers=12),
        render_lever(entry),
        render_lever(failed),
    ]
    for page in pages:
        parsed = parse(page)
        assert {tag for tag, _attrs in parsed.elements} <= TAGS
        assert {name for _tag, attrs in parsed.elements for name in attrs} <= ATTRIBUTES
        assert not parsed.tags("script") and len(parsed.tags("style")) == 1
        assert len(parsed.tags("title")) == page.count("<title>")
        for attrs in parsed.tags("a"):
            assert (attrs["href"] or "").startswith("#l-")
        assert "<script" not in page and "<img" not in page and "<iframe" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
        drawings(page)
    assert "&amp;lt;already&amp;gt; &amp;amp; &lt;b onclick=alert(6)&gt;" in pages[1]
    text = parse(pages[1]).text
    for value in hostile:
        assert value in text, value
    assert parse(pages[0]).tags("h1") and hostile[0] in parse(pages[0]).text
    # The name reaches a title attribute on the terrain mark as data, not as attributes.
    mark = next(attrs for tag, attrs in parse(pages[0]).classed("mk") if tag == "a")
    assert mark["title"] is not None and hostile[0] in mark["title"] and "(struck)" in mark["title"]


def test_control_and_direction_override_characters_are_neutralised() -> None:
    disguise = chr(0x202E) + "gninrael" + chr(0) + chr(0x1B) + "[31m" + chr(0x2028) + "tail"
    entry = lever(attribution=["line" + chr(10) + "break", disguise])
    page = render_lever(entry)
    for code in (0, 0x1B, 0x202E, 0x2028):
        assert chr(code) not in page
    assert chr(0xFFFD) + "gninrael" + chr(0xFFFD) * 2 + "[31m" + chr(0xFFFD) + "tail" in page
    assert "line break" in page
    page.encode("utf-8")


def test_only_plain_http_urls_become_links(tmp_path: Path) -> None:
    urls = [
        "https://atlas.example/feature/12?layer=7#top",
        "HTTP://ATLAS.EXAMPLE/feature/13",
        "javascript:alert(1)",
        "data:text/html,<script>alert(2)</script>",
        " https://atlas.example/leading-space",
        'https://atlas.example/"onclick="alert(3)',
        "https://atlas.example/a b",
        "//atlas.example/protocol-relative",
        "vbscript:msgbox(4)",
        "file:///etc/passwd",
        "JaVaScRiPt:alert(5)",
        "https://atlas.example/" + chr(9) + "tab",
    ]
    chosen = [site(7)]
    entry = lever(
        sites=chosen,
        derivation=derivation(tmp_path, chosen),
        legend=legend(
            [
                LegendEntry(index=index, cosine=1.0 - index / 100, url=url)
                for index, url in enumerate(urls)
            ]
        ),
        attribution=[*urls[:4], "See https://atlas.example/in-a-sentence for the source"],
    )
    page = render_lever(entry)
    parsed = parse(page)
    links = parsed.tags("a")
    assert [attrs["href"] for attrs in links] == [urls[0], urls[1]] * 2
    assert all(attrs["rel"] == "noopener noreferrer" for attrs in links)
    assert set().union(*links) == {"href", "rel"}
    for url in urls:
        assert " ".join(url.split()) in parsed.text  # still shown, as inert text
    assert "href" not in re.sub(r'<a href="https?://atlas[^"]*"', "", page, flags=re.IGNORECASE)


def test_recipe_prompts_answers_and_weight_values_never_appear(tmp_path: Path) -> None:
    chosen = [site(7), site(9), site(11)]
    entry = lever(
        sites=chosen[:1],
        derivation=derivation(tmp_path, chosen),
        legend=legend([LegendEntry(index=4, cosine=0.61, label="mentions of refusing")]),
        verdict=verdict([operation(chosen[0].parameter)]),
    )
    assert PROMPT_SECRET in entry.model_dump_json() and ANSWER_SECRET in entry.model_dump_json()
    for page in (render_map([entry], layers=12), render_lever(entry)):
        for secret in (PROMPT_SECRET, ANSWER_SECRET, "contrast-pos", "contrast-neg", "0.1234"):
            assert secret not in page
        parsed = parse(page)
        text = parsed.text
        assert "Positive prompts 5" in text and "Negative prompts 5" in text
        # Every candidate site is listed, including the two that were not chosen.
        choices = {row[0]: row[-1] for row in parsed.rows if row[-1] in {"chosen", "not chosen"}}
        assert choices == {
            chosen[0].parameter: "chosen",
            chosen[1].parameter: "not chosen",
            chosen[2].parameter: "not chosen",
        }
        assert "+2.500" in text and "+1.500" in text and "+0.500" in text
        # The legend is pencil, a hypothesis, and attributed to the dictionary it came from.
        assert len(parsed.classed("hyp")) == 1
        assert "Legend hypothesis" in text and "offered as hypotheses" in text
        assert "mentions of refusing" in text and "+0.610" in text and "43.8%" in text
        assert "atlas-org/transcoders" in text and "d" * 40 in text and "CC-BY-4.0" in text


def test_page_is_offline_and_its_policy_admits_only_its_own_stylesheet() -> None:
    entries = [inked(7), lever(sites=[site(5)]), struck(3)]
    for page in (render_map(entries, layers=12), render_lever(entries[0])):
        lowered = page.lower()
        for forbidden in (
            "http://", "https://", "src=", "<link", "@import", "url(", "<script", "<img",
            "<iframe", "<object", "<embed", "<form", "<base", "style=", "javascript:",
        ):  # fmt: skip
            assert forbidden not in lowered, forbidden
        assert not re.search(r"\son[a-z]+\s*=", lowered)
        parsed = parse(page)
        policies = [
            attrs["content"]
            for attrs in parsed.tags("meta")
            if attrs.get("http-equiv") == "Content-Security-Policy"
        ]
        assert len(policies) == 1 and policies[0] is not None
        policy = policies[0]
        assert policy.startswith("default-src 'none';")
        assert "base-uri 'none'" in policy and "form-action 'none'" in policy
        assert "unsafe" not in policy and "script-src" not in policy and "*" not in policy
        styles = re.findall(r"<style>(.*?)</style>", page, flags=re.DOTALL)
        assert len(styles) == 1 and styles[0].isascii() and chr(10) not in styles[0]
        fingerprint = base64.b64encode(hashlib.sha256(styles[0].encode()).digest()).decode()
        assert f"style-src 'sha256-{fingerprint}'" in policy
        assert "prefers-color-scheme:dark" in styles[0]
        assert "prefers-reduced-motion" in styles[0]
        assert "gradient" not in styles[0] and "shadow" not in styles[0]
        assert page.index("Content-Security-Policy") < page.index("<style>")


def test_text_and_line_colours_meet_aa_contrast_in_both_schemes() -> None:
    style = re.findall(r"<style>(.*?)</style>", render_map([]), flags=re.DOTALL)[0]
    light = re.search(r":root\{([^}]*)\}", style)
    dark = re.search(r"prefers-color-scheme:dark\)\{:root\{([^}]*)\}", style)
    assert light is not None and dark is not None

    def luminance(colour: str) -> float:
        channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def contrast(first: str, second: str) -> float:
        high, low = sorted((luminance(first), luminance(second)), reverse=True)
        return (high + 0.05) / (low + 0.05)

    for scheme in (light.group(1), dark.group(1)):
        colours = dict(re.findall(r"--(\w+):(#[0-9a-f]{6})", scheme))
        assert set(colours) == {"paper", "sheet", "ink", "graphite", "rule", "accent"}
        for ground in (colours["paper"], colours["sheet"]):
            assert contrast(colours["ink"], ground) >= 7
            assert contrast(colours["graphite"], ground) >= 4.5  # body text and pencil lines
            assert contrast(colours["accent"], ground) >= 3  # the working-window bar


def test_identical_input_gives_identical_bytes_in_any_order() -> None:
    entries = [inked(7), lever(sites=[site(5)]), struck(3), lever(sites=[site(5)], behavior="alt")]
    first = render_map(entries, title="Survey", layers=12)
    assert first == render_map(list(entries), title="Survey", layers=12)
    assert first == render_map(entries[::-1], title="Survey", layers=12)
    assert first == render_map([*entries, entries[0]], title="Survey", layers=12)
    assert render_lever(entries[0]) == render_lever(entries[0])
    assert first != render_map(entries, title="Survey", layers=13)
    assert first.encode("utf-8").isascii()


def test_dial_table_lists_every_strength_and_the_drawing_is_well_formed() -> None:
    strengths = (-0.5, 0.25, 0.5, 1.0, 2.0)
    entry = lever(strengths=strengths, nominated=0.5)
    page = render_lever(entry)
    parsed = parse(page)
    by_strength = {row[0]: row for row in parsed.rows if len(row) == 9}
    assert list(by_strength) == ["Strength", "original", "-0.5", "0.25", "0.5", "1", "2"]
    assert by_strength["original"][3:6] == ["25.0%", "100.0%", "80.0%"]
    assert by_strength["0.5"] == [
        "0.5", "+40.0 [+20.0, +60.0]", "-10.0 [-20.0, +5.0]", "65.0%", "90.0%", "90.0%",
        "87.5%", "5.75", "nominated, in window",
    ]  # fmt: skip
    assert by_strength["2"][-1] == "in window"
    text = parsed.text
    assert "Exploratory development evidence" in text and "does not establish anything" in text
    assert "At the nominated strength 0.5" in text
    assert "Selectivity" in text and f"Targets passing 25.0% {ARROW} 65.0%" in text
    assert f"Regressions passing 100.0% {ARROW} 90.0%" in text
    assert f"Finish rate 80.0% {ARROW} 90.0%" in text
    assert "Accuracy when finished, edited 87.5%" in text
    assert "working window -0.5 to 2" in text

    plots = [item for item in drawings(page) if item.get("class") == "plot"]
    assert len(plots) == 1
    plot = plots[0]
    classes = [item.get("class") for item in plot.iter() if item.get("class")]
    assert classes.count("wn") == 1 and classes.count("wb") == 1  # the working window
    assert classes.count("zr") == 1  # the zero line
    assert classes.count("nm") == 1  # the nominated strength
    assert classes.count("bd") == 2 and classes.count("lt") == 1 and classes.count("lr") == 1
    assert classes.count("mt") == len(strengths) and classes.count("mr") == len(strengths)
    titles = [item.text or "" for item in plot.iter("title")]
    assert len(titles) == len(strengths)
    assert titles[2] == (
        "strength 0.5: target +40.0 [+20.0, +60.0] pp, regression -10.0 [-20.0, +5.0] pp"
    )
    labels = [item.text for item in plot.iter("text")]
    assert "nominated 0.5" in labels and "0" in labels and "strength" in labels
    width, height = (float(value) for value in (plot.get("viewBox") or "").split()[2:])
    for item in plot.iter():
        for name in ("x", "cx"):
            if item.get(name) is not None:
                assert 0 <= float(item.get(name, "0")) <= width
        for name in ("y", "cy"):
            if item.get(name) is not None:
                assert 0 <= float(item.get(name, "0")) <= height


def test_one_point_dial_and_a_dial_without_a_window_still_draw() -> None:
    single = render_lever(lever(strengths=(1.0,), nominated=1.0))
    plot = next(item for item in drawings(single) if item.get("class") == "plot")
    classes = [item.get("class") for item in plot.iter()]
    assert classes.count("wk") == 2 and "lt" not in classes and "bd" not in classes
    assert [row[0] for row in parse(single).rows if len(row) == 9][1:] == ["original", "1"]

    chosen = [site(7)]
    unnominated = seal(
        {
            "name": lever_name("refusal", chosen),
            "behavior": "refusal",
            "base": BASE,
            "sites": chosen,
            "dial": dial((0.5, 1.0), None, window=False),
        }
    )
    page = render_lever(unnominated)
    plot = next(item for item in drawings(page) if item.get("class") == "plot")
    classes = [item.get("class") for item in plot.iter()]
    assert "wn" not in classes and "nm" not in classes
    text = parse(page).text
    assert "No strength was nominated from this dial." in text
    assert "No working window was found." in text


def test_bare_entry_renders_and_says_it_has_not_been_refereed() -> None:
    chosen = [site(None, "embed.weight")]
    name = lever_name("claimed", chosen)
    bare = seal({"name": name, "behavior": "claimed", "base": BASE, "sites": chosen})
    assert bare.dial is None and bare.legend is None and bare.verdict is None
    for page in (render_lever(bare), render_map([bare])):
        parsed = parse(page)
        text = parsed.text
        assert status_reason(None) + " Nothing on this sheet is established." in text
        assert "No dial recorded" in text and "No legend" in text
        assert "No derivation recorded" in text and "No replications recorded." in text
        assert "No contraindications recorded." in text
        assert [attrs["class"] for _tag, attrs in parsed.classed("sheet")] == ["sheet pencil"]
        assert not [item for item in drawings(page) if item.get("class") == "plot"]
    assert "No site here records a layer" in parse(render_map([bare])).text


def test_entry_and_size_limits_raise() -> None:
    entry = lever()
    with pytest.raises(NrlyError) as error:
        render_map([entry] * (MAX_ENTRIES + 1))
    assert error.value.code == "lever_report_too_large"

    note = "x" * 500
    large = [
        lever(
            sites=[site(index % 12, f"model.layers.{index}.weight")],
            contraindications=[
                Contraindication(variant=f"build-{item}", note=note) for item in range(64)
            ],
        )
        for index in range(70)
    ]
    assert len(render_map(large[:30], layers=12).encode("utf-8")) <= MAX_REPORT_BYTES
    with pytest.raises(NrlyError) as error:
        render_map(large, layers=12)
    assert error.value.code == "lever_report_too_large"

    for layers in (0, -1, 5000, True):
        with pytest.raises(NrlyError) as error:
            render_map([entry], layers=layers)
        assert error.value.code == "invalid_request"
    with pytest.raises(NrlyError):
        render_lever(entry.model_dump())  # type: ignore[arg-type]
