"""End-to-end lever workflow on the deterministic fixture: dial, referee, status.

The fixture halves every sum. Adding ``strength * [[0.5, 0.5]]`` to its output weights
repairs it at exactly strength 1, so the dial has a known one-point window. Because
this is an engineering fixture, even a supported verdict must leave the lever in pencil.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from nrly import cli
from nrly.backends.tiny import TinyBackend
from nrly.contracts import (
    BudgetSpec,
    CaseSpec,
    ModelSpec,
    NrlyError,
    PortableBase,
    WeightOperationSpec,
    canonical_json,
    digest,
)
from nrly.lever_app import (
    DialRequest,
    LegendRequest,
    LeverPlan,
    RefereeRequest,
    dial_lever,
    lever_map,
    referee_lever,
    scale_operations,
)
from nrly.lever_contracts import (
    Derivation,
    LeverEntry,
    LeverSite,
    RecipeSpec,
    SiteScore,
    operations_digest,
)
from nrly.levers import pencil
from nrly.study_contracts import InferenceRule

SITE = LeverSite(parameter="output.weight", module="output", layer=0)
RULE = InferenceRule(resamples=1000)
BUDGET = BudgetSpec(wall_seconds=60, max_cases=12, max_output_tokens=64)
ZEROS = [
    "[0,0]", "[0.0,0]", "[0,0.0]", "[0.0,0.0]", "[0e0,0]", "[0,0e0]",
    "[0e0,0e0]", "[0.00,0]", "[0,0.00]", "[0.00,0.00]", "[0e1,0]", "[0,0e1]",
]  # fmt: skip


def _operation(right: list[float]) -> WeightOperationSpec:
    return WeightOperationSpec(
        kind="weight_low_rank", target="output.weight", shape=[1, 2], rank=1,
        left=[1.0], right=right,
    )  # fmt: skip


def _cases(pairs: list[tuple[int, int]], zeros: list[str], split: str) -> list[CaseSpec]:
    targets = [
        CaseSpec(
            id=f"{split}-sum-{a}-{b}", input=f"[{a},{b}]", expected=str(a + b),
            scorer="numeric", split=split,
        )  # fmt: skip
        for a, b in pairs
    ]
    regressions = [
        CaseSpec(
            id=f"{split}-zero-{index}", input=text, expected="0", scorer="numeric",
            split=split, role="regression",
        )  # fmt: skip
        for index, text in enumerate(zeros)
    ]
    return targets + regressions


DEVELOPMENT = _cases([(2, 4), (1, 7), (3, 3), (5, 6), (8, 1), (4, 9)], ZEROS[:6], "discovery")
HOLDOUT = _cases([(12, 5), (7, 7), (9, 2), (6, 11), (3, 14), (10, 8)], ZEROS[6:], "test")


def _pencilled(root: Path, *, weights: str | None = None) -> LeverEntry:
    """A hand-built derivation and plan; recipes themselves need a language model."""
    levers = lever_map(root)
    with TinyBackend(ModelSpec()) as backend:
        identity = backend.identity
    base = PortableBase.from_identity(identity)
    if weights is not None:
        base = base.model_copy(update={"weights_sha256": weights})
    prompt = [CaseSpec(id=f"prompt-{index}", input="[1,1]") for index in range(8)]
    recipe = RecipeSpec(behavior="tiny-sum", positive=prompt[:4], negative=prompt[4:], sites=[SITE])
    derivation = Derivation(
        recipe=recipe, recipe_digest=digest(recipe), base=base,
        runtime_fingerprint=identity.fingerprint, chosen=SITE,
        direction=levers.tensors.put([1.0]),
        scores=[
            SiteScore(
                site=SITE, separation=1.0, difference_norm=1.0, fit_cases=2, selection_cases=2
            )
        ],
    )  # fmt: skip
    entry = levers.add(pencil(derivation))
    plan = LeverPlan(
        entry_id=entry.id, base=base, unit_operations=[_operation([0.5, 0.5])],
        # Equally large, pointed elsewhere: the output then ignores the second addend.
        controls={"wrong-direction": [_operation([0.5, -0.5])]},
    )  # fmt: skip
    plans = root / "levers" / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    (plans / f"{entry.id}.json").write_text(canonical_json(plan), encoding="utf-8")
    return entry


def _dial_request(strengths: list[float]) -> DialRequest:
    return DialRequest(
        model=ModelSpec(), cases=DEVELOPMENT, strengths=strengths, seeds=[0], rule=RULE,
        budget=BUDGET,
    )  # fmt: skip


def test_scaling_is_linear_and_refuses_nonlinear_operations() -> None:
    doubled = scale_operations([_operation([0.5, 0.5])], 2.0)
    assert doubled[0].factor == 2.0 and doubled[0].right == [0.5, 0.5]
    scale = WeightOperationSpec(kind="weight_scale", target="output.weight", factor=2)
    with pytest.raises(NrlyError) as error:
        scale_operations([scale], 0.5)
    assert error.value.code == "unsupported_operation"


def test_dial_then_referee_runs_real_arms_and_a_fixture_never_inks(tmp_path: Path) -> None:
    entry = _pencilled(tmp_path)
    with pytest.raises(NrlyError) as error:
        referee_lever(
            tmp_path, entry.id, RefereeRequest(model=ModelSpec(), development_cases=DEVELOPMENT),
            HOLDOUT,
        )  # fmt: skip
    assert error.value.code == "lever_unnominated"

    dialled = dial_lever(tmp_path, entry.id, _dial_request([0.5, 1.0, 2.0]))
    assert dialled["evidence"] == "exploratory_development"
    assert dialled["failed"] == []
    assert dialled["nominated"] == 1.0
    assert dialled["window"] == {"low": 1.0, "high": 1.0, "points": 1}
    by_strength = {point["strength"]: point for point in dialled["points"]}
    assert by_strength[1.0]["target_rate"] == 1.0
    assert by_strength[0.5]["target_rate"] == by_strength[2.0]["target_rate"] == 0.0
    assert all(point["regression_rate"] == 1.0 for point in dialled["points"])

    levers = lever_map(tmp_path)
    revised = levers.get(dialled["entry_id"])
    assert revised.parent_id == entry.id and revised.status == "pencil"
    assert revised.strength == 1.0
    assert revised.operations == scale_operations([_operation([0.5, 0.5])], 1.0)
    assert levers.get(entry.id).dial is None  # the original pencil entry is untouched

    request = RefereeRequest(
        model=ModelSpec(), development_cases=DEVELOPMENT, inference=RULE, run_budget=BUDGET
    )
    with pytest.raises(NrlyError) as error:
        referee_lever(tmp_path, revised.id, request, HOLDOUT, replication=True)
    assert error.value.code == "lever_unrefereed"

    result = referee_lever(tmp_path, revised.id, request, HOLDOUT)
    assert result["refereed"] and result["final_error_code"] is None
    assert result["verdict"] == "observed_improvement"
    assert [pair["reference"] for pair in result["inference"]] == [
        "baseline", "no_op", "control:wrong-direction",
    ]  # fmt: skip
    assert all(pair["target_superior"] for pair in result["inference"])

    refereed = levers.get(result["entry_id"])
    assert refereed.verdict is not None
    assert refereed.verdict.controls == ["wrong-direction"]
    assert refereed.verdict.holdout.count == len(HOLDOUT)
    assert refereed.verdict.operations_digest == operations_digest(refereed.operations)
    assert refereed.verdict.assessment.scope == "engineering_fixture"
    # A supported verdict on an engineering fixture never becomes ink.
    assert refereed.status == result["status"] == "pencil"
    assert [item.id for item in levers.history(refereed.id)] == [entry.id, revised.id, refereed.id]

    # The refereed edit is frozen: no second first-verdict and no new dial.
    with pytest.raises(NrlyError) as error:
        referee_lever(tmp_path, refereed.id, request, HOLDOUT)
    assert error.value.code == "lever_refereed"
    with pytest.raises(NrlyError) as error:
        dial_lever(tmp_path, refereed.id, _dial_request([1.0]))
    assert error.value.code == "lever_refereed"


def test_a_dial_with_no_working_strength_nominates_nothing(tmp_path: Path) -> None:
    entry = _pencilled(tmp_path)
    dialled = dial_lever(tmp_path, entry.id, _dial_request([0.25, 3.0]))
    assert dialled["nominated"] is None and dialled["window"] is None
    revised = lever_map(tmp_path).get(dialled["entry_id"])
    assert revised.dial is not None and revised.operations == [] and revised.strength is None


def _cli(capsys: pytest.CaptureFixture[str], root: Path, *arguments: str) -> tuple[int, Any]:
    code = cli.execute(cli.parser().parse_args(["--root", str(root), *arguments]))
    return code, json.loads(capsys.readouterr().out)


def test_lever_commands_share_the_application_operations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "store"
    entry = _pencilled(root)
    code, listed = _cli(capsys, root, "lever", "list")
    assert code == 0 and [item["id"] for item in listed] == [entry.id]
    assert listed[0]["status"] == "pencil" and listed[0]["strength"] is None
    assert _cli(capsys, root, "lever", "list", "--status", "inked") == (0, [])

    request = tmp_path / "dial.json"
    request.write_text(_dial_request([0.25, 3.0]).model_dump_json(), encoding="utf-8")
    code, dialled = _cli(capsys, root, "lever", "dial", entry.id, str(request))
    assert code == 0 and dialled["nominated"] is None

    # No working window: the operator may still nominate a tested strength to referee.
    with pytest.raises(NrlyError) as error:
        _cli(capsys, root, "lever", "nominate", dialled["entry_id"], "9.0")
    assert error.value.code == "lever_unmeasured"
    code, nominated = _cli(capsys, root, "lever", "nominate", dialled["entry_id"], "3.0")
    assert code == 0 and nominated["nominated"] == 3.0 and nominated["inside_window"] is False
    head = lever_map(root).get(nominated["entry_id"])
    assert head.strength == 3.0 and head.dial is not None and head.dial.window is None
    assert head.operations == scale_operations([_operation([0.5, 0.5])], 3.0)

    code, history = _cli(capsys, root, "lever", "history", head.id)
    assert [item["id"] for item in history] == [entry.id, dialled["entry_id"], head.id]
    code, shown = _cli(capsys, root, "lever", "show", head.id)
    assert shown["id"] == head.id and shown["status"] == "pencil"

    # A legend is read from an inherited dictionary file and is always pencil.
    header = json.dumps({"W_dec": {"dtype": "F32", "shape": [3, 1], "data_offsets": [0, 12]}})
    dictionary = tmp_path / "layer_0.safetensors"
    dictionary.write_bytes(
        struct.pack("<Q", len(header)) + header.encode() + struct.pack("<3f", 1.0, -1.0, 0.5)
    )
    legend_request = tmp_path / "legend.json"
    legend_request.write_text(
        LegendRequest(
            kind="transcoder", repo_id="example/dictionary", revision="a" * 40,
            filename="layer_{layer}.safetensors", license="mit", writes="mlp.hook_out", top_k=2,
        ).model_dump_json(),
        encoding="utf-8",
    )  # fmt: skip
    code, named = _cli(
        capsys, root, "lever", "legend", head.id, str(legend_request),
        "--local-file", str(dictionary), "--cache", str(tmp_path / "atlas"),
    )  # fmt: skip
    assert code == 0 and named["legend"]["status"] == "pencil"
    assert [item["index"] for item in named["legend"]["entries"]] == [0, 1]
    assert named["legend"]["source"]["tensor_sha256"]
    legend_head = lever_map(root).get(named["entry_id"])
    assert legend_head.status == "pencil"
    assert any("example/dictionary" in line and "mit" in line for line in legend_head.attribution)

    page, chart = tmp_path / "lever.html", tmp_path / "map.html"
    assert _cli(capsys, root, "lever", "page", legend_head.id, str(page))[0] == 0
    assert _cli(capsys, root, "lever", "map", str(chart), "--layers", "4")[0] == 0
    for document in (page.read_text(encoding="utf-8"), chart.read_text(encoding="utf-8")):
        assert "tiny-sum@output.weight" in document
        # Shareable pages carry aggregates only, never the recipe's prompts.
        assert "prompt-0" not in document and "[1,1]" not in document
    with pytest.raises(NrlyError) as error:
        _cli(capsys, root, "lever", "page", legend_head.id, str(page))
    assert error.value.code == "destination_exists"

    bundle = tmp_path / "bundle"
    assert _cli(capsys, root, "lever", "export", legend_head.id, str(bundle))[0] == 0
    code, imported = _cli(capsys, tmp_path / "elsewhere", "lever", "import", str(bundle))
    assert code == 0 and imported["id"] == legend_head.id

    for kind in ("lever", "recipe", "lever-dial-request", "portable-patch"):
        code, schema = _cli(capsys, root, "schema", kind)
        assert code == 0 and schema["type"] == "object"


def test_a_lever_for_other_weights_is_refused_before_any_edit_runs(tmp_path: Path) -> None:
    entry = _pencilled(tmp_path, weights="0" * 64)
    with pytest.raises(NrlyError) as error:
        dial_lever(tmp_path, entry.id, _dial_request([1.0]))
    assert error.value.code == "base_mismatch"
