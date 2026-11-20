from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nrly.artifacts import TensorArtifactStore
from nrly.contracts import (
    CaseSpec,
    GenerationSpec,
    ModelIdentity,
    ModelSpec,
    NrlyError,
    PortableBase,
    ProgramSpec,
    WeightOperationSpec,
    canonical_json,
    digest,
)
from nrly.evaluation.reports import CaseEvaluation, EvaluationReport, GroupSummary
from nrly.evaluation.statistics import paired_inference
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
)
from nrly.levers import (
    BUNDLE_ENTRY,
    BUNDLE_TENSORS,
    LeverMap,
    build_dial,
    pencil,
    verdict_from_study,
)
from nrly.study_contracts import (
    CandidateSpec,
    ComparatorSpec,
    EffectEstimate,
    HoldoutCommitment,
    InferenceRule,
    PairedInference,
    StudyAssessment,
    StudyRecord,
    StudySpec,
    StudyUsage,
)

WEIGHTS = "1" * 64
OTHER_WEIGHTS = "9" * 64
RUNTIME = "2" * 64
REFEREE_RUNTIME = "3" * 64
SITE = LeverSite(
    parameter="model.layers.3.mlp.down_proj.weight", module="model.layers.3.mlp", layer=3
)
SPARE = LeverSite(
    parameter="model.layers.4.mlp.down_proj.weight", module="model.layers.4.mlp", layer=4
)
RULE = InferenceRule(resamples=1000)
IDENTITY_FIELDS = ("base", "behavior", "sites", "name", "derivation")


def _code(error: pytest.ExceptionInfo[NrlyError]) -> str:
    return error.value.code


def _base(weights: str = WEIGHTS, dtype: str = "torch.float32") -> PortableBase:
    return PortableBase(
        backend="huggingface",
        model_id="org/model",
        revision="a" * 40,
        dtype=dtype,
        weights_sha256=weights,
    )


def _derivation(
    levers: LeverMap, *, behavior: str = "refusal", weights: str = WEIGHTS
) -> Derivation:
    recipe = RecipeSpec(
        behavior=behavior,
        positive=[CaseSpec(id=f"p{index}", input=f"harmful request {index}") for index in range(4)],
        negative=[CaseSpec(id=f"n{index}", input=f"benign request {index}") for index in range(4)],
        sites=[SITE, SPARE],
    )
    return Derivation(
        recipe=recipe,
        recipe_digest=digest(recipe),
        base=_base(weights),
        runtime_fingerprint=RUNTIME,
        chosen=SITE,
        direction=levers.tensors.put([0.6, 0.8, 0.0, 0.0]),
        scores=[
            SiteScore(
                site=SITE, separation=3.5, difference_norm=1.25, fit_cases=4, selection_cases=4
            ),
            SiteScore(
                site=SPARE, separation=0.4, difference_norm=0.5, fit_cases=4, selection_cases=4
            ),
        ],
    )


def _ops(strength: float = 1.0, target: str = SITE.parameter) -> list[WeightOperationSpec]:
    """The edit a strength compiles to in these tests: scale the site by that factor."""
    return [WeightOperationSpec(kind="weight_scale", target=target, factor=strength)]


def _effect(mean: float) -> EffectEstimate:
    return EffectEstimate(
        cases=8,
        seeds=1,
        mean_difference=mean,
        ci_low=mean,
        ci_high=mean,
        confidence=0.95,
        improved_cases=8 if mean > 0 else 0,
        regressed_cases=0,
        tied_cases=0 if mean > 0 else 8,
        sign_test_p=2 / 256 if mean > 0 else 1.0,
    )


def _pair(helps: bool = True, reference: str = "baseline") -> PairedInference:
    return PairedInference(
        reference=reference,
        target=_effect(1.0 if helps else 0.0),
        regression=_effect(0.0),
        target_superior=helps,
        target_harmed=False,
        regression_non_inferior=True,
        regression_inferior=False,
    )


def _dial(
    nominated: float | None = 1.0,
    strengths: Sequence[float] = (0.5, 1.0),
    target: str = SITE.parameter,
) -> Dial:
    return Dial(
        rule=RULE,
        generation=GenerationSpec(),
        seeds=[0],
        cases=16,
        runtime_fingerprint=RUNTIME,
        baseline_target_rate=0.0,
        baseline_regression_rate=1.0,
        points=[
            DosePoint(
                strength=strength,
                operations_digest=operations_digest(_ops(strength, target)),
                inference=_pair(strength == nominated),
                target_rate=float(strength == nominated),
                regression_rate=1.0,
                selectivity=50.0 if strength == nominated else 0.0,
            )
            for strength in strengths
        ],
        window=None
        if nominated is None
        else StrengthWindow(low=nominated, high=nominated, points=1),
        nominated=nominated,
    )


def _verdict(
    operations: list[WeightOperationSpec],
    *,
    controls: Sequence[str] = ("random-direction",),
    verdict: str = "observed_improvement",
    recommendation: str = "candidate_supported",
    scope: str = "real_model",
    error: str | None = None,
    runtime: str = REFEREE_RUNTIME,
    rule: InferenceRule | None = RULE,
    base: PortableBase | None = None,
) -> RefereeVerdict:
    """A verdict refereed on the weights ``_derivation`` pins unless ``base`` says otherwise."""
    return RefereeVerdict(
        base=base or _base(),
        study_id="ab" * 16,
        study_spec_digest="6" * 64,
        holdout=HoldoutCommitment(digest="7" * 64, count=16, targets=8, regressions=8),
        operations_digest=operations_digest(operations),
        nominee_digest="8" * 64,
        runtime_fingerprint=runtime,
        inference_rule=rule,
        seeds=[11, 12],
        controls=list(controls),
        comparators=["prompt:be-brief"],
        development_cases=12,
        assessment=StudyAssessment.model_validate(
            {
                "scorer_version": "nrly-scorers/2",
                "verdict": verdict,
                "recommendation": recommendation,
                "scope": scope,
                "inference": [_pair(verdict == "observed_improvement").model_dump()],
            }
        ),
        final_error_code=error,
        finished_at="2026-11-20T01:00:00Z",
    )


def _struck(operations: list[WeightOperationSpec], **changes: Any) -> RefereeVerdict:
    return _verdict(operations, verdict="no_change", recommendation="retain_original", **changes)


def _legend(repo_id: str = "atlas-lab/transcoders") -> Legend:
    return Legend(
        source=AtlasSource(
            kind="transcoder",
            repo_id=repo_id,
            revision="b" * 40,
            filename="layer_3/width_16k/params.safetensors",
            tensor_sha256="4" * 64,
            license="CC-BY-4.0",
            layer=3,
            writes="mlp_out",
            latents=16384,
            width=4,
        ),
        entries=[
            LegendEntry(index=7, cosine=0.9, label="declines"),
            LegendEntry(index=2, cosine=-0.4),
        ],
        explained=0.85,
    )


def _root(levers: LeverMap, **derivation: str) -> LeverEntry:
    return levers.add(pencil(_derivation(levers, **derivation)))


def _dialled(levers: LeverMap, **derivation: str) -> LeverEntry:
    return levers.attach_dial(_root(levers, **derivation).id, _dial(), _ops())


def _entry_file(levers: LeverMap, entry: LeverEntry) -> Path:
    return levers.entries / f"{entry.id}.json"


def _hand_sealed(content: dict[str, Any]) -> str:
    """An entry file whose id matches its content, written without any contract check."""
    return canonical_json({**content, "id": digest(content)})


def _assert_revises(child: LeverEntry, parent: LeverEntry) -> None:
    assert child.parent_id == parent.id and child.id != parent.id
    for field in IDENTITY_FIELDS:
        assert getattr(child, field) == getattr(parent, field)


# --- entries -------------------------------------------------------------------------


def test_pencil_entry_is_named_by_test_and_site_and_round_trips(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    derivation = _derivation(levers)
    content = pencil(derivation)
    assert content["name"] == "refusal@model.layers.3.mlp.down_proj.weight"
    assert content["strength"] is None and content["operations"] == []
    assert content["status"] == "pencil" and content["parent_id"] is None

    entry = levers.add(content)
    assert entry.status == "pencil" and entry.verdict is None and entry.dial is None
    assert entry.sites == [SITE] and entry.derivation == derivation
    assert entry.base.weights_sha256 == WEIGHTS
    assert levers.get(entry.id) == entry
    assert _entry_file(levers, entry).read_bytes() == canonical_json(entry).encode("utf-8")
    # Content addressing makes a repeated add the same entry, not a second one.
    assert levers.add(content).id == entry.id
    assert [item.id for item in levers.heads()] == [entry.id]


def test_a_status_or_a_meaning_cannot_be_chosen(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    content = pencil(_derivation(levers))
    claimed = levers.add({**content, "status": "inked"})
    assert claimed.status == "pencil" and claimed.id == levers.add(content).id
    with pytest.raises(NrlyError, match="named by its behavior test") as error:
        levers.add({**content, "name": "the-deception-feature"})
    assert _code(error) == "invalid_lever"
    with pytest.raises(NrlyError) as error:
        levers.add({**content, "interpretation": "detects harmful intent"})
    assert _code(error) == "invalid_lever"
    assert len(levers.heads()) == 1


def test_get_refuses_unknown_and_malformed_handles(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    with pytest.raises(NrlyError) as error:
        levers.get("0" * 64)
    assert _code(error) == "unknown_lever"
    with pytest.raises(NrlyError) as error:
        levers.get("../entries/" + "0" * 64)
    assert _code(error) == "invalid_artifact"


def test_add_requires_the_direction_in_the_maps_own_tensor_store(tmp_path: Path) -> None:
    elsewhere = LeverMap(tmp_path / "elsewhere")
    levers = LeverMap(tmp_path / "map")
    with pytest.raises(NrlyError, match="direction artifact") as error:
        levers.add(pencil(_derivation(elsewhere)))
    assert _code(error) == "invalid_artifact"
    assert levers.heads() == []


def test_altered_entries_are_refused_on_read(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    entry = _root(levers)
    stored = _entry_file(levers, entry)
    data = json.loads(stored.read_text(encoding="utf-8"))
    data["derivation"]["scores"][0]["separation"] = 99.0
    stored.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(NrlyError, match="altered") as error:
        levers.get(entry.id)
    assert _code(error) == "invalid_artifact"
    with pytest.raises(NrlyError):
        levers.heads()
    with pytest.raises(NrlyError):
        levers.attach_dial(entry.id, _dial(), _ops())

    # Rehashing the altered content yields a valid entry, but not under this handle.
    forged = seal({key: value for key, value in data.items() if key != "id"})
    assert forged.id != entry.id
    stored.write_text(canonical_json(forged), encoding="utf-8")
    with pytest.raises(NrlyError, match="handle") as error:
        levers.get(entry.id)
    assert _code(error) == "artifact_mismatch"


# --- revisions -----------------------------------------------------------------------


def test_attach_dial_records_strength_and_operations_but_never_a_status(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    root = _root(levers)
    dialled = levers.attach_dial(root.id, _dial(), _ops())
    _assert_revises(dialled, root)
    assert dialled.strength == 1.0 and dialled.operations == _ops()
    assert dialled.dial == _dial() and dialled.dial.window is not None
    assert dialled.status == "pencil"
    # The revised entry is untouched and still readable.
    assert levers.get(root.id) == root and root.dial is None

    # A dial may be redone before any verdict; the earlier one stays in the parent.
    redone = levers.attach_dial(dialled.id, _dial(0.5), _ops(0.5))
    assert redone.strength == 0.5 and redone.operations == _ops(0.5)
    assert levers.get(dialled.id).strength == 1.0


def test_attach_dial_ties_operations_to_the_nominated_strength(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    root = _root(levers)
    null = levers.attach_dial(root.id, _dial(None), [])
    assert null.strength is None and null.operations == [] and null.dial is not None
    assert null.dial.nominated is None and null.status == "pencil"

    refusals = [
        (_dial(None), _ops()),  # no nominated strength, yet an edit is supplied
        (_dial(), []),  # a nominated strength without the edit it compiles to
        (_dial(), _ops(0.5)),  # the edit of another tested strength
    ]
    for dial, operations in refusals:
        with pytest.raises(NrlyError) as error:
            levers.attach_dial(root.id, dial, operations)
        assert _code(error) == "invalid_lever"
    # An edit must stay on the lever's own sites even when the dial tested it.
    with pytest.raises(NrlyError, match="one of the lever's sites") as error:
        levers.attach_dial(root.id, _dial(target=SPARE.parameter), _ops(target=SPARE.parameter))
    assert _code(error) == "invalid_lever"
    assert {entry.id for entry in levers.heads()} == {null.id}


def test_the_refereed_edit_cannot_be_redialled(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    inked = levers.attach_verdict(_dialled(levers).id, _verdict(_ops()))
    with pytest.raises(NrlyError, match="refereed edit") as error:
        levers.attach_dial(inked.id, _dial(0.5), _ops(0.5))
    assert _code(error) == "lever_refereed"
    with pytest.raises(NrlyError) as error:
        levers.attach_dial(inked.id, _dial(), _ops())
    assert _code(error) == "lever_refereed"


def test_attach_legend_stays_pencil_and_credits_the_inherited_source(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    inked = levers.attach_verdict(_dialled(levers).id, _verdict(_ops()))
    line = "Feature labels: atlas-lab/transcoders @ bbbbbbb, licensed CC-BY-4.0"
    legended = levers.attach_legend(inked.id, _legend(), f"  {line}  ")
    _assert_revises(legended, inked)
    assert legended.legend == _legend() and legended.legend.status == "pencil"
    assert legended.attribution == [line]
    # A legend names hypotheses; it neither earns nor costs ink.
    assert legended.status == inked.status == "inked"

    again = levers.attach_legend(legended.id, _legend(), line)
    assert again.attribution == [line]
    other = "Also atlas-lab/saes (cc-by-4.0)"
    second = levers.attach_legend(again.id, _legend("atlas-lab/saes"), other)
    assert second.attribution == [line, other]
    assert second.legend is not None and second.legend.source.repo_id == "atlas-lab/saes"


def test_attach_legend_refusals(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    root = _root(levers)
    for line in (
        "atlas-lab/transcoders",  # the licence is missing
        "some atlas, CC-BY-4.0",  # the source is missing
        "",
    ):
        with pytest.raises(NrlyError, match="source and its licence") as error:
            levers.attach_legend(root.id, _legend(), line)
        assert _code(error) == "invalid_lever"
    with pytest.raises(NrlyError, match="attribution") as error:
        levers.attach_legend(root.id, _legend(), "atlas-lab/transcoders CC-BY-4.0 " + "x" * 300)
    assert _code(error) == "invalid_lever"

    # A lever claimed without a derivation has no direction for a legend to describe.
    bare = levers.add(
        {
            "name": lever_name("refusal", [SPARE]),
            "behavior": "refusal",
            "base": root.base.model_dump(),
            "sites": [SPARE.model_dump()],
        }
    )
    assert bare.status == "pencil"
    with pytest.raises(NrlyError, match="derived direction") as error:
        levers.attach_legend(bare.id, _legend(), "atlas-lab/transcoders CC-BY-4.0")
    assert _code(error) == "invalid_lever"


@pytest.mark.parametrize(
    ("changes", "status"),
    [
        pytest.param({}, "inked", id="supported-real-model-with-control"),
        pytest.param({"controls": ()}, "pencil", id="supported-without-a-control"),
        pytest.param(
            {"verdict": "no_change", "recommendation": "retain_original"},
            "struck",
            id="completed-and-unsupported",
        ),
        pytest.param(
            {"verdict": "regressed", "recommendation": "retain_original"},
            "struck",
            id="completed-and-regressed",
        ),
        pytest.param({"scope": "engineering_fixture"}, "pencil", id="engineering-fixture"),
        pytest.param({"error": "study_execution_failed"}, "pencil", id="failed-execution"),
        pytest.param({"rule": None}, "pencil", id="no-registered-inference-rule"),
    ],
)
def test_status_is_derived_from_the_attached_verdict(
    tmp_path: Path, changes: dict[str, Any], status: str
) -> None:
    levers = LeverMap(tmp_path)
    dialled = _dialled(levers)
    verdict = _verdict(_ops(), **changes)
    entry = levers.attach_verdict(dialled.id, verdict)
    _assert_revises(entry, dialled)
    assert entry.status == status
    assert entry.verdict == verdict
    assert levers.get(entry.id).status == status
    assert [item.id for item in levers.lookup(WEIGHTS, status=status)] == [entry.id]  # type: ignore[arg-type]
    # A stored status cannot be upgraded by editing the file either.
    if status != "inked":
        stored = _entry_file(levers, entry)
        data = json.loads(stored.read_text(encoding="utf-8"))
        data["status"] = "inked"
        stored.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(NrlyError) as error:
            levers.get(entry.id)
        assert _code(error) == "invalid_artifact"


def test_a_verdict_cannot_be_attached_to_an_entry_with_no_operations(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    root = _root(levers)
    null = levers.attach_dial(root.id, _dial(None), [])
    assert root.operations == [] and null.operations == []
    # ``_verdict([])`` carries the digest of no operations, so only the missing edit refuses it.
    for entry in (root, null):
        for verdict in (_verdict([]), _verdict(_ops()), _struck([])):
            with pytest.raises(NrlyError, match="referees operations") as error:
                levers.attach_verdict(entry.id, verdict)
            assert _code(error) == "invalid_lever"
    # The contract refuses the same content when it arrives through ``add``.
    for entry in (root, null):
        content = entry.model_dump(mode="json", exclude={"id"})
        with pytest.raises(NrlyError, match="needs the edit it refereed") as error:
            levers.add({**content, "verdict": _verdict([]).model_dump(mode="json")})
        assert _code(error) == "invalid_lever"
    assert [entry.id for entry in levers.heads()] == [null.id]
    assert levers.get(null.id).status == "pencil" and levers.get(null.id).verdict is None


def test_attach_verdict_refusals(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    dialled = _dialled(levers)
    with pytest.raises(NrlyError, match="exactly this lever's operations") as error:
        levers.attach_verdict(dialled.id, _verdict(_ops(0.5)))
    assert _code(error) == "verdict_mismatch"
    # The right edit refereed on other bytes, or on these bytes loaded at another dtype.
    for base in (_base(OTHER_WEIGHTS), _base(dtype="torch.float16")):
        for verdict in (_verdict(_ops(), base=base), _struck(_ops(), base=base)):
            with pytest.raises(NrlyError, match="weights this lever pins") as error:
                levers.attach_verdict(dialled.id, verdict)
            assert _code(error) == "verdict_mismatch"
    assert [entry.id for entry in levers.heads()] == [dialled.id]
    assert levers.get(dialled.id).status == "pencil" and levers.get(dialled.id).verdict is None

    struck = levers.attach_verdict(dialled.id, _struck(_ops()))
    assert struck.status == "struck"
    with pytest.raises(NrlyError, match="replication") as error:
        levers.attach_verdict(struck.id, _verdict(_ops(), runtime="5" * 64))
    assert _code(error) == "verdict_exists"
    assert levers.get(struck.id).status == "struck"


def test_replications_need_a_verdict_and_a_new_runtime(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    dialled = _dialled(levers)
    with pytest.raises(NrlyError, match="existing verdict") as error:
        levers.add_replication(dialled.id, _verdict(_ops()))
    assert _code(error) == "verdict_missing"

    struck = levers.attach_verdict(dialled.id, _struck(_ops()))
    with pytest.raises(NrlyError) as error:
        levers.add_replication(struck.id, _verdict(_ops(), runtime=REFEREE_RUNTIME))
    assert _code(error) == "replication_exists"
    with pytest.raises(NrlyError) as error:
        levers.add_replication(struck.id, _verdict(_ops(0.5), runtime="5" * 64))
    assert _code(error) == "verdict_mismatch"
    # A new runtime that loaded other weights replicates nothing about this lever.
    for base in (_base(OTHER_WEIGHTS), _base(dtype="torch.float16")):
        with pytest.raises(NrlyError, match="weights this lever pins") as error:
            levers.add_replication(struck.id, _verdict(_ops(), runtime="5" * 64, base=base))
        assert _code(error) == "verdict_mismatch"
    assert [entry.id for entry in levers.heads()] == [struck.id]

    supportive = _verdict(_ops(), runtime="5" * 64)
    replicated = levers.add_replication(struck.id, supportive)
    _assert_revises(replicated, struck)
    assert replicated.replications == [supportive]
    # A refutation is not overturned by a later supportive run: the first verdict rules.
    assert replicated.status == "struck" and replicated.verdict == struck.verdict
    with pytest.raises(NrlyError) as error:
        levers.add_replication(replicated.id, _struck(_ops(), runtime="5" * 64))
    assert _code(error) == "replication_exists"
    third = levers.add_replication(replicated.id, _struck(_ops(), runtime="a" * 64))
    assert [item.runtime_fingerprint for item in third.replications] == ["5" * 64, "a" * 64]
    assert third.status == "struck"


def test_contraindications_accumulate_in_order(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    root = _root(levers)
    quantized = Contraindication(variant="q4_k_m gguf", note="no target gain at any strength")
    abliterated = Contraindication(
        variant="abliterated fine-tune", note="direction absent", evidence_digest="c" * 64
    )
    first = levers.add_contraindication(root.id, quantized)
    second = levers.add_contraindication(first.id, abliterated)
    _assert_revises(first, root)
    _assert_revises(second, first)
    assert second.contraindications == [quantized, abliterated]
    assert levers.get(root.id).contraindications == []
    assert second.status == "pencil"


def test_a_revision_cannot_rewrite_identity_or_drop_recorded_evidence(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    struck = levers.attach_verdict(_dialled(levers).id, _struck(_ops()))
    noted = levers.add_contraindication(
        struck.id, Contraindication(variant="int8", note="did not hold")
    )
    content = noted.model_dump(mode="json", exclude={"id"})

    with pytest.raises(NrlyError, match="not on this map") as error:
        levers.add({**content, "parent_id": "0" * 64})
    assert _code(error) == "unknown_lever"

    elsewhere = pencil(_derivation(levers, weights=OTHER_WEIGHTS))
    rewrites: list[dict[str, Any]] = [
        # the same chain claimed for other weights, its verdict re-pinned to match
        {
            **content,
            "base": elsewhere["base"],
            "derivation": elsewhere["derivation"],
            "verdict": _struck(_ops(), base=_base(OTHER_WEIGHTS)).model_dump(mode="json"),
        },
        # a struck entry quietly returned to pencil
        {**content, "verdict": None},
        # the refutation swapped for a supportive verdict
        {**content, "verdict": _verdict(_ops()).model_dump(mode="json")},
        # the refereed edit changed under its verdict
        {**content, "dial": None, "strength": None},
        # a recorded contraindication removed
        {**content, "contraindications": []},
    ]
    for rewrite in rewrites:
        with pytest.raises(NrlyError) as error:
            levers.add({**rewrite, "parent_id": noted.id})
        assert _code(error) == "lineage_mismatch"
    # Moving the chain without its verdict does not even seal.
    with pytest.raises(NrlyError, match="weights this lever pins") as error:
        levers.add(
            {
                **content,
                "base": elsewhere["base"],
                "derivation": elsewhere["derivation"],
                "parent_id": noted.id,
            }
        )
    assert _code(error) == "invalid_lever"
    assert [entry.id for entry in levers.heads()] == [noted.id]
    assert levers.heads()[0].status == "struck"

    # Adding to the record through ``add`` is still a legitimate revision.
    extended = levers.add(
        {
            **content,
            "parent_id": noted.id,
            "contraindications": [
                *content["contraindications"],
                {"variant": "int4", "note": "did not hold"},
            ],
        }
    )
    assert extended.parent_id == noted.id and extended.status == "struck"


def test_add_and_import_cannot_sidestep_the_revision_rules(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path / "map")
    root = _root(levers)
    dialled = levers.attach_dial(root.id, _dial(), _ops())
    fresh = root.model_dump(mode="json", exclude={"id"})
    tested = dialled.model_dump(mode="json", exclude={"id", "parent_id"})
    supportive = _verdict(_ops()).model_dump(mode="json")
    elsewhere = _verdict(_ops(), runtime="5" * 64, base=_base(OTHER_WEIGHTS)).model_dump(
        mode="json"
    )
    redialled = {**tested, "operations": [op.model_dump() for op in _ops(0.5)]}
    sidesteps: list[tuple[dict[str, Any], str]] = [
        # a supportive verdict over no edit at all, whose digest of nothing agrees
        ({**fresh, "verdict": _verdict([]).model_dump(mode="json")}, "needs the edit it refereed"),
        # the right edit, refereed on other weights
        ({**tested, "verdict": elsewhere}, "weights this lever pins"),
        # an honest verdict, padded with a replication from other weights
        ({**tested, "verdict": supportive, "replications": [elsewhere]}, "weights this lever pins"),
        # a strength tested by the dial, recorded with a different edit
        (redialled, "nominated strength"),
        # a dial that found no window, yet an edit is recorded
        (
            {
                **fresh,
                "dial": _dial(None).model_dump(mode="json"),
                "operations": [op.model_dump() for op in _ops()],
            },
            "nominates no strength",
        ),
    ]
    for content, reason in sidesteps:
        with pytest.raises(NrlyError, match=reason) as error:
            levers.add(content)
        assert _code(error) == "invalid_lever"
    assert [entry.id for entry in levers.heads()] == [dialled.id]

    # An entry the contract seals but the map refuses is refused as a bundle too.
    forged = seal(redialled)
    assert forged.strength == 1.0 and forged.operations == _ops(0.5)
    bundle = levers.export(root.id, tmp_path / "bundle")
    (bundle / BUNDLE_ENTRY).write_text(canonical_json(forged), encoding="utf-8")
    target = LeverMap(tmp_path / "target")
    with pytest.raises(NrlyError, match="nominated strength") as error:
        target.import_bundle(bundle)
    assert _code(error) == "invalid_lever"
    assert target.heads() == []


def test_import_refuses_hand_sealed_verdicts_the_contract_forbids(tmp_path: Path) -> None:
    source = LeverMap(tmp_path / "source")
    root = _root(source)
    dialled = source.attach_dial(root.id, _dial(), _ops())
    fresh = root.model_dump(mode="json", exclude={"id"})
    tested = dialled.model_dump(mode="json", exclude={"id"})
    elsewhere = _verdict(_ops(), runtime="5" * 64, base=_base(OTHER_WEIGHTS)).model_dump(
        mode="json"
    )
    inked = {**tested, "verdict": _verdict(_ops()).model_dump(mode="json"), "status": "inked"}

    def shipped(name: str, content: dict[str, Any]) -> tuple[Path, LeverMap]:
        bundle = source.export(dialled.id, tmp_path / name)
        (bundle / BUNDLE_ENTRY).write_text(_hand_sealed(content), encoding="utf-8")
        return bundle, LeverMap(tmp_path / f"{name}-map")

    # Hand sealing reproduces a real identity, so the refusals below are the contract's own.
    bundle, target = shipped("honest", inked)
    assert target.import_bundle(bundle) == seal(inked)
    assert [entry.status for entry in target.lookup(WEIGHTS)] == ["inked"]

    forgeries: dict[str, dict[str, Any]] = {
        "no-edit": {**fresh, "verdict": _verdict([]).model_dump(mode="json"), "status": "inked"},
        "other-weights": {**inked, "verdict": elsewhere},
        "replicated-on-other-weights": {**inked, "replications": [elsewhere]},
    }
    for name, content in forgeries.items():
        bundle, target = shipped(name, content)
        with pytest.raises(NrlyError, match="altered") as error:
            target.import_bundle(bundle)
        assert _code(error) == "invalid_artifact", name
        assert target.heads() == [] and not target.tensors.root.exists(), name


# --- reading the map -----------------------------------------------------------------


def test_history_heads_and_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    levers = LeverMap(tmp_path)
    assert levers.heads() == [] and levers.lookup(WEIGHTS) == []

    root = _root(levers)
    dialled = levers.attach_dial(root.id, _dial(), _ops())
    inked = levers.attach_verdict(dialled.id, _verdict(_ops()))
    # A second dial of the same root is a branch: both tips are current.
    branch = levers.attach_dial(root.id, _dial(0.5), _ops(0.5))
    other_test = _root(levers, behavior="sycophancy")
    other_weights = levers.attach_verdict(
        _dialled(levers, weights=OTHER_WEIGHTS).id, _struck(_ops(), base=_base(OTHER_WEIGHTS))
    )

    assert [entry.id for entry in levers.history(inked.id)] == [root.id, dialled.id, inked.id]
    assert [entry.id for entry in levers.history(branch.id)] == [root.id, branch.id]
    assert levers.history(root.id) == [root]

    def ids(entries: list[LeverEntry]) -> set[str]:
        return {entry.id for entry in entries}

    assert ids(levers.heads()) == {inked.id, branch.id, other_test.id, other_weights.id}
    assert ids(levers.heads(WEIGHTS)) == {inked.id, branch.id, other_test.id}
    assert ids(levers.heads(OTHER_WEIGHTS)) == {other_weights.id}
    assert levers.heads("f" * 64) == []
    # Deterministic order: by lever name, then id.
    names = [(entry.name, entry.id) for entry in levers.heads()]
    assert names == sorted(names)

    assert ids(levers.lookup(WEIGHTS)) == {inked.id, branch.id, other_test.id}
    assert ids(levers.lookup(WEIGHTS, behavior="refusal")) == {inked.id, branch.id}
    assert ids(levers.lookup(WEIGHTS, status="inked")) == {inked.id}
    assert ids(levers.lookup(WEIGHTS, status="pencil")) == {branch.id, other_test.id}
    assert ids(levers.lookup(WEIGHTS, behavior="sycophancy", status="pencil")) == {other_test.id}
    assert levers.lookup(WEIGHTS, behavior="sycophancy", status="inked") == []
    # The refuted lever is found, not hidden.
    assert ids(levers.lookup(OTHER_WEIGHTS, status="struck")) == {other_weights.id}
    with pytest.raises(NrlyError) as error:
        levers.heads("not-a-digest")
    assert _code(error) == "invalid_artifact"

    # Files that are not content-addressed entries are not entries.
    (levers.entries / "notes.txt").write_text("scratch", encoding="utf-8")
    assert len(levers.heads()) == 4

    monkeypatch.setattr("nrly.levers.MAX_MAP_ENTRIES", 2)
    with pytest.raises(NrlyError, match="at most 2 entries") as error:
        levers.heads()
    assert _code(error) == "budget_exceeded"
    with pytest.raises(NrlyError) as error:
        levers.lookup(WEIGHTS)
    assert _code(error) == "budget_exceeded"
    with pytest.raises(NrlyError, match="revision bound") as error:
        levers.history(inked.id)
    assert _code(error) == "lineage_mismatch"


def test_history_refuses_a_stored_chain_that_rewrites_its_parent(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    struck = levers.attach_verdict(_dialled(levers).id, _struck(_ops()))
    # Written straight into the store, bypassing ``add``: validly sealed, but it erases
    # the refutation it claims to revise.
    content = struck.model_dump(mode="json", exclude={"id"})
    forged = seal({**content, "verdict": None, "parent_id": struck.id})
    assert forged.status == "pencil"
    (levers.entries / f"{forged.id}.json").write_text(canonical_json(forged), encoding="utf-8")
    # Otherwise the forged tip would be the chain's head and the lever would read as pencil.
    reads = [
        lambda: levers.get(forged.id),
        lambda: levers.history(forged.id),
        levers.heads,
        lambda: levers.lookup(WEIGHTS, status="pencil"),
        lambda: levers.export(forged.id, tmp_path / "out"),
    ]
    for read in reads:
        with pytest.raises(NrlyError) as error:
            read()
        assert _code(error) == "lineage_mismatch"
    assert levers.get(struck.id).status == "struck"
    assert not (tmp_path / "out").exists()


# --- bundles -------------------------------------------------------------------------


def test_export_import_round_trip_carries_the_entry_and_its_direction(tmp_path: Path) -> None:
    source = LeverMap(tmp_path / "source")
    dialled = _dialled(source)
    inked = source.attach_verdict(dialled.id, _verdict(_ops()))
    assert inked.derivation is not None
    direction = inked.derivation.direction

    bundle = source.export(inked.id, tmp_path / "bundle")
    assert bundle == (tmp_path / "bundle").resolve()
    assert sorted(path.name for path in bundle.iterdir()) == [BUNDLE_ENTRY, BUNDLE_TENSORS]
    assert sorted(path.name for path in (bundle / BUNDLE_TENSORS).iterdir()) == [
        f"{direction.id}.bin",
        f"{direction.id}.json",
    ]
    assert (bundle / BUNDLE_ENTRY).read_bytes() == canonical_json(inked).encode("utf-8")

    target = LeverMap(tmp_path / "target")
    imported = target.import_bundle(bundle)
    assert imported == inked and target.get(inked.id) == inked
    assert imported.status == "inked" and imported.base.weights_sha256 == WEIGHTS
    np.testing.assert_array_equal(
        target.tensors.get(direction), np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
    )
    assert (target.tensors.root / f"{direction.id}.bin").read_bytes() == (
        source.tensors.root / f"{direction.id}.bin"
    ).read_bytes()
    assert target.tensors.inspect(direction.id) == direction

    # The parent stayed behind: the imported entry is a chain of one and a head.
    assert imported.parent_id == dialled.id
    assert target.history(inked.id) == [inked]
    assert target.lookup(WEIGHTS, status="inked") == [inked]
    assert target.import_bundle(bundle) == inked
    # Importing the parent later joins the chain.
    target.import_bundle(source.export(dialled.id, tmp_path / "parent"))
    assert [entry.id for entry in target.history(inked.id)] == [dialled.id, inked.id]
    assert target.heads() == [inked]
    # An imported entry is revised like any other.
    noted = target.add_contraindication(
        inked.id, Contraindication(variant="int8", note="did not hold")
    )
    assert noted.parent_id == inked.id and noted.status == "inked"


def test_export_refusals_leave_no_bundle(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path / "map")
    root = _root(levers)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(NrlyError, match="absent or empty") as error:
        levers.export(root.id, occupied)
    assert _code(error) == "destination_exists"
    assert [path.name for path in occupied.iterdir()] == ["keep.txt"]
    with pytest.raises(NrlyError) as error:
        levers.export("0" * 64, tmp_path / "unknown")
    assert _code(error) == "unknown_lever"

    assert root.derivation is not None
    blob = levers.tensors.root / f"{root.derivation.direction.id}.bin"
    blob.write_bytes(np.array([0.0, 1.0, 0.0, 0.0], dtype="<f4").tobytes())
    with pytest.raises(NrlyError) as error:
        levers.export(root.id, tmp_path / "altered")
    assert _code(error) == "artifact_mismatch"
    assert not (tmp_path / "unknown").exists() and not (tmp_path / "altered").exists()
    # An existing empty directory is an acceptable destination once the store is intact.
    blob.write_bytes(np.array([0.6, 0.8, 0.0, 0.0], dtype="<f4").tobytes())
    empty = tmp_path / "empty"
    empty.mkdir()
    assert (levers.export(root.id, empty) / BUNDLE_ENTRY).is_file()


def test_tampered_bundles_are_refused_and_add_nothing(tmp_path: Path) -> None:
    source = LeverMap(tmp_path / "source")
    dialled = _dialled(source)
    assert dialled.derivation is not None
    direction = dialled.derivation.direction

    def fresh(name: str) -> tuple[Path, LeverMap]:
        return source.export(dialled.id, tmp_path / name), LeverMap(tmp_path / f"{name}-map")

    bundle, target = fresh("flipped-direction")
    blob = bundle / BUNDLE_TENSORS / f"{direction.id}.bin"
    blob.write_bytes(np.array([-0.6, -0.8, 0.0, 0.0], dtype="<f4").tobytes())
    with pytest.raises(NrlyError) as error:
        target.import_bundle(bundle)
    assert _code(error) == "artifact_mismatch"
    assert target.heads() == [] and not target.tensors.root.exists()

    bundle, target = fresh("missing-direction")
    (bundle / BUNDLE_TENSORS / f"{direction.id}.bin").unlink()
    with pytest.raises(NrlyError) as error:
        target.import_bundle(bundle)
    assert _code(error) == "invalid_artifact"
    assert target.heads() == []

    bundle, target = fresh("edited-entry")
    manifest = bundle / BUNDLE_ENTRY
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["strength"] = 0.5
    data["dial"]["nominated"] = 0.5
    data["dial"]["window"] = {"low": 0.5, "high": 0.5, "points": 1}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(NrlyError, match="altered") as error:
        target.import_bundle(bundle)
    assert _code(error) == "invalid_artifact"
    assert target.heads() == []

    bundle, target = fresh("claimed-status")
    manifest = bundle / BUNDLE_ENTRY
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["status"] = "inked"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(NrlyError) as error:
        target.import_bundle(bundle)
    assert _code(error) == "invalid_artifact"

    with pytest.raises(NrlyError) as error:
        LeverMap(tmp_path / "nothing-map").import_bundle(tmp_path / "no-such-bundle")
    assert _code(error) == "invalid_artifact"


def test_import_refuses_a_bundle_that_rewrites_a_held_parent(tmp_path: Path) -> None:
    source = LeverMap(tmp_path / "source")
    struck = source.attach_verdict(_dialled(source).id, _struck(_ops()))
    content = struck.model_dump(mode="json", exclude={"id"})
    # A map that never saw the refutation will hold this laundering revision...
    lenient = LeverMap(tmp_path / "lenient")
    lenient.tensors.put([0.6, 0.8, 0.0, 0.0])
    laundered = seal({**content, "verdict": None, "parent_id": struck.id})
    (lenient.entries).mkdir(parents=True)
    (lenient.entries / f"{laundered.id}.json").write_text(
        canonical_json(laundered), encoding="utf-8"
    )
    bundle = lenient.export(laundered.id, tmp_path / "bundle")
    # ...but a map that holds the struck parent refuses it.
    with pytest.raises(NrlyError) as error:
        source.import_bundle(bundle)
    assert _code(error) == "lineage_mismatch"
    assert [entry.id for entry in source.heads()] == [struck.id]


# --- dials ---------------------------------------------------------------------------


def _report(
    targets: Sequence[int],
    regressions: Sequence[int],
    *,
    finished: Sequence[bool] | None = None,
    complete: bool = True,
    scope: str = "real_model",
    prefix: str = "",
) -> EvaluationReport:
    """A scored report; ``finished`` switches on truncation="fail" bookkeeping."""
    roles = ["target"] * len(targets) + ["regression"] * len(regressions)
    flags = [None] * len(roles) if finished is None else list(finished)
    cases = [
        CaseEvaluation.model_validate(
            {
                "case_id": f"{prefix}{role}-{index}",
                "split": "validation",
                "role": role,
                "outcome": "pass" if passed else "fail",
                "output": "out",
                "expected": "out",
                "scorer": "exact",
                "score": float(passed),
                "reason": "exact match" if passed else "exact mismatch",
                "finished": flag,
            }
        )
        for index, (role, passed, flag) in enumerate(
            zip(roles, [*targets, *regressions], flags, strict=True)
        )
    ]

    def summary(selected: list[CaseEvaluation]) -> GroupSummary:
        passed = sum(case.outcome == "pass" for case in selected)
        done = [case for case in selected if case.finished]
        return GroupSummary(
            total=len(selected),
            scored=len(selected),
            passed=passed,
            failed=len(selected) - passed,
            unscored=0,
            accuracy=passed / len(selected) if selected else None,
            finished=None if finished is None else len(done),
            finished_passed=None
            if finished is None
            else sum(case.outcome == "pass" for case in done),
        )

    return EvaluationReport.model_validate(
        {
            "scorer_version": "nrly-scorers/2" if finished is None else "nrly-scorers/3",
            "truncation": "unscored" if finished is None else "fail",
            "run_id": "run",
            "scope": scope,
            "complete": complete,
            "issues": [] if complete else ["run status is failed"],
            "summary": summary(cases),
            "targets": summary([case for case in cases if case.role == "target"]),
            "regressions": summary([case for case in cases if case.role == "regression"]),
            "cases": cases,
        }
    )


# Eight target cases the unedited model fails and eight regression cases it passes.
BASELINE = _report([0] * 8, [1] * 8)
HELPS = _report([1] * 8, [1] * 8)  # selectivity 1.0 / 0.02
HELPS_LESS = _report([1] * 7 + [0], [1] * 8)  # selectivity 0.875 / 0.02
FLAT = _report([0] * 8, [1] * 8)
BREAKS = _report([1] * 8, [0] * 4 + [1] * 4)  # fixes targets, loses half the regressions


def _build(
    points: dict[float, Sequence[EvaluationReport] | EvaluationReport],
    *,
    baseline: Sequence[EvaluationReport] = (BASELINE,),
    seeds: Sequence[int] = (0,),
    rule: InferenceRule = RULE,
    truncation: str = "unscored",
) -> Dial:
    return build_dial(
        rule=rule,
        truncation=truncation,  # type: ignore[arg-type]
        generation=GenerationSpec(max_new_tokens=64),
        seeds=seeds,
        runtime_fingerprint=RUNTIME,
        baseline=baseline,
        points={
            strength: (
                operations_digest(_ops(strength)),
                [reports] if isinstance(reports, EvaluationReport) else reports,
            )
            for strength, reports in points.items()
        },
    )


def test_dial_window_selectivity_and_nomination(tmp_path: Path) -> None:
    # Supplied out of order: a dial is always reported by increasing strength.
    dial = _build({2.0: BREAKS, 0.5: FLAT, 1.5: HELPS, 1.0: HELPS_LESS})
    assert [point.strength for point in dial.points] == [0.5, 1.0, 1.5, 2.0]
    assert [point.operations_digest for point in dial.points] == [
        operations_digest(_ops(strength)) for strength in (0.5, 1.0, 1.5, 2.0)
    ]
    assert dial.cases == 16 and dial.seeds == [0] and dial.rule == RULE
    assert dial.generation.max_new_tokens == 64 and dial.runtime_fingerprint == RUNTIME
    assert (dial.baseline_target_rate, dial.baseline_regression_rate) == (0.0, 1.0)
    assert [point.target_rate for point in dial.points] == [0.0, 0.875, 1.0, 1.0]
    assert [point.regression_rate for point in dial.points] == [1.0, 1.0, 1.0, 0.5]

    flat = [0] * 8
    expected = {
        0.5: (flat, flat),
        1.0: ([1] * 7 + [0], flat),
        1.5: ([1] * 8, flat),
        2.0: ([1] * 8, [-1] * 4 + [0] * 4),
    }
    for point in dial.points:
        target, regression = expected[point.strength]
        assert point.inference == paired_inference("baseline", [target], [regression], RULE)
    helps = [
        point.inference.target_superior and point.inference.regression_non_inferior
        for point in dial.points
    ]
    assert helps == [False, True, True, False]
    assert dial.points[3].inference.target_superior
    assert dial.points[3].inference.regression_inferior

    # Target gain over the larger of the registered margin (0.02) and the observed cost.
    assert [point.selectivity for point in dial.points] == pytest.approx([0.0, 43.75, 50.0, 2.0])
    assert dial.window == StrengthWindow(low=1.0, high=1.5, points=2)
    assert dial.nominated == 1.5

    # Unscored truncation reports no finish metrics.
    assert dial.truncation == "unscored" and dial.baseline_finish_rate is None
    assert all(
        point.finish_rate is None and point.accuracy_when_finished is None for point in dial.points
    )

    # However strong, a dial is development evidence: the entry stays in pencil.
    assert dial.evidence == "exploratory_development"
    levers = LeverMap(tmp_path)
    entry = levers.attach_dial(_root(levers).id, dial, _ops(1.5))
    assert entry.strength == 1.5 and entry.status == "pencil" and entry.verdict is None
    with pytest.raises(NrlyError) as error:
        levers.attach_dial(_root(levers).id, dial, _ops(1.0))
    assert _code(error) == "invalid_lever"


def test_dial_without_a_window_nominates_nothing(tmp_path: Path) -> None:
    dial = _build({0.5: FLAT, 1.0: BREAKS})
    assert dial.window is None and dial.nominated is None
    assert [point.selectivity for point in dial.points] == pytest.approx([0.0, 2.0])
    # The null result is still recorded on the map, with no edit.
    levers = LeverMap(tmp_path)
    entry = levers.attach_dial(_root(levers).id, dial, [])
    assert entry.dial == dial and entry.strength is None and entry.operations == []


def test_dial_window_needs_adjacent_strengths() -> None:
    # Helps at 0.5 and 1.5 but not 1.0: two separate runs, never one window across the gap.
    broken = _build({0.5: HELPS_LESS, 1.0: FLAT, 1.5: HELPS})
    assert broken.window == StrengthWindow(low=1.5, high=1.5, points=1)
    assert broken.nominated == 1.5
    mirrored = _build({0.5: HELPS, 1.0: FLAT, 1.5: HELPS_LESS})
    assert mirrored.window == StrengthWindow(low=0.5, high=0.5, points=1)
    assert mirrored.nominated == 0.5
    # Equal runs with equal selectivity resolve to the lower strengths.
    tied = _build({0.5: HELPS, 1.0: FLAT, 1.5: HELPS})
    assert tied.window == StrengthWindow(low=0.5, high=0.5, points=1)
    assert tied.nominated == 0.5
    # Length comes first: a longer run beats a shorter one holding the best selectivity.
    longer = _build({0.5: HELPS, 1.0: FLAT, 1.5: HELPS_LESS, 2.0: HELPS_LESS})
    assert longer.window == StrengthWindow(low=1.5, high=2.0, points=2)
    assert longer.nominated == 1.5
    # A run broken by regression harm, not only by a missing target effect.
    harmed = _build({0.5: HELPS, 1.0: BREAKS, 1.5: HELPS, 2.0: HELPS})
    assert harmed.window == StrengthWindow(low=1.5, high=2.0, points=2)


def test_dial_nomination_ties_prefer_the_smallest_absolute_strength() -> None:
    dial = _build({-1.0: HELPS, 0.5: HELPS, 2.0: HELPS})
    assert dial.window == StrengthWindow(low=-1.0, high=2.0, points=3)
    assert dial.nominated == 0.5
    best_wins = _build({-1.0: HELPS_LESS, 0.5: HELPS_LESS, 2.0: HELPS})
    assert best_wins.window == StrengthWindow(low=-1.0, high=2.0, points=3)
    assert best_wins.nominated == 2.0


def test_dial_selectivity_is_undefined_without_a_margin_or_a_cost() -> None:
    strict = InferenceRule(regression_margin=0, resamples=1000)
    dial = _build({1.0: HELPS, 2.0: HELPS, 3.0: BREAKS}, rule=strict)
    assert [point.selectivity for point in dial.points] == [None, None, pytest.approx(2.0)]
    assert dial.window == StrengthWindow(low=1.0, high=2.0, points=2)
    assert dial.nominated == 1.0


def test_dial_pairs_each_seed_with_its_own_baseline_and_averages_rates() -> None:
    already = _report([1] + [0] * 7, [1] * 8)  # this seed's baseline passes one target
    dial = _build({1.0: [HELPS, HELPS_LESS]}, baseline=[BASELINE, already], seeds=[3, 4])
    point = dial.points[0]
    assert dial.seeds == [3, 4] and dial.cases == 16
    assert point.inference == paired_inference(
        "baseline", [[1] * 8, [0] + [1] * 6 + [0]], [[0] * 8, [0] * 8], RULE
    )
    assert (point.inference.target.cases, point.inference.target.seeds) == (8, 2)
    assert point.inference.target.mean_difference == pytest.approx(0.875)
    assert point.target_rate == pytest.approx((1.0 + 0.875) / 2)
    assert dial.baseline_target_rate == pytest.approx((0.0 + 0.125) / 2)
    assert point.regression_rate == dial.baseline_regression_rate == 1.0
    assert point.selectivity == pytest.approx(0.875 / 0.02)


def test_dial_reports_finish_metrics_when_truncation_fails() -> None:
    done = [True] * 16
    baseline = [
        _report([0] * 8, [1] * 8, finished=done),
        # Four regression outputs hit the token limit and are scored as failures.
        _report([0] * 8, [1] * 4 + [0] * 4, finished=[True] * 12 + [False] * 4),
    ]
    helped = [
        _report([1] * 8, [1] * 8, finished=done),
        # Fourteen finish; one of those is wrong, two targets are cut off.
        _report([1] * 5 + [0] * 3, [1] * 8, finished=[True] * 6 + [False] * 2 + [True] * 8),
    ]
    cut_off = [_report([0] * 8, [0] * 8, finished=[False] * 16)] * 2
    dial = _build({1.0: helped, 2.0: cut_off}, baseline=baseline, seeds=[3, 4], truncation="fail")
    assert dial.truncation == "fail"
    assert dial.baseline_finish_rate == pytest.approx((1.0 + 0.75) / 2)
    first, second = dial.points
    assert first.finish_rate == pytest.approx((1.0 + 14 / 16) / 2)
    # Pooled over seeds (29 of 30), not the mean of per-seed accuracies.
    assert first.accuracy_when_finished == pytest.approx(29 / 30)
    assert second.finish_rate == 0.0 and second.accuracy_when_finished is None
    assert second.target_rate == 0.0 and second.regression_rate == 0.0

    with pytest.raises(NrlyError, match="truncation") as error:
        _build({1.0: [HELPS, HELPS]}, baseline=baseline, seeds=[3, 4], truncation="fail")
    assert _code(error) == "dial_incomplete"
    with pytest.raises(NrlyError, match="truncation") as error:
        _build({1.0: helped}, baseline=baseline, seeds=[3, 4], truncation="unscored")
    assert _code(error) == "dial_incomplete"
    uncounted = helped[1].model_copy(
        update={"summary": helped[1].summary.model_copy(update={"finished": None})}
    )
    with pytest.raises(NrlyError, match="finish counts") as error:
        _build({1.0: [helped[0], uncounted]}, baseline=baseline, seeds=[3, 4], truncation="fail")
    assert _code(error) == "dial_incomplete"


def test_dial_refuses_incomplete_or_misaligned_reports() -> None:
    unscored_case = HELPS.model_copy(
        update={
            "cases": [
                HELPS.cases[0].model_copy(update={"outcome": "unscored", "score": None}),
                *HELPS.cases[1:],
            ]
        }
    )
    reordered = HELPS.model_copy(update={"cases": [*HELPS.cases[1:], HELPS.cases[0]]})
    refusals: list[dict[str, Any]] = [
        {"points": {1.0: _report([1] * 8, [1] * 8, complete=False)}},
        {"points": {1.0: HELPS}, "baseline": [_report([0] * 8, [1] * 8, complete=False)]},
        {"points": {1.0: unscored_case}},
        {"points": {1.0: reordered}},
        {"points": {1.0: _report([1] * 8, [1] * 8, prefix="other-")}},
        {"points": {1.0: _report([1] * 7, [1] * 8)}},
        {"points": {0.5: HELPS, 1.0: _report([1] * 9, [1] * 7)}},  # roles moved between cases
        {"points": {1.0: _report([1] * 8, [1] * 8, scope="engineering_fixture")}},
        {"points": {1.0: [HELPS, HELPS]}},  # two reports for one seed
        {"points": {1.0: HELPS}, "seeds": [0, 1]},  # a seed without reports
        {"points": {1.0: [HELPS, HELPS]}, "baseline": [BASELINE, BASELINE], "seeds": [7, 7]},
        {"points": {}},
        {"points": {1.0: HELPS}, "baseline": [], "seeds": []},
        {
            "points": {1.0: _report([1] * 8, [])},  # no regression cases to spare
            "baseline": [_report([0] * 8, [])],
        },
    ]
    for arguments in refusals:
        with pytest.raises(NrlyError) as error:
            _build(**arguments)
        assert _code(error) == "dial_incomplete", arguments


def test_dial_refuses_untestable_strengths_and_malformed_digests() -> None:
    with pytest.raises(NrlyError) as error:
        _build({32.0: HELPS})
    assert _code(error) == "invalid_dial"
    with pytest.raises(NrlyError) as error:
        build_dial(
            rule=RULE,
            truncation="unscored",
            generation=GenerationSpec(),
            seeds=[0],
            runtime_fingerprint=RUNTIME,
            baseline=[BASELINE],
            points={1.0: ("not-a-digest", [HELPS])},
        )
    assert _code(error) == "invalid_dial"


# --- verdicts ------------------------------------------------------------------------

CANDIDATE = CandidateSpec(
    name="refusal-lever",
    program=ProgramSpec(kind="weight_scale", target=SITE.parameter, factor=1.0),
)


def _identity(
    fingerprint: str = REFEREE_RUNTIME,
    *,
    weights: str | None = WEIGHTS,
    dtype: str | None = "torch.float32",
) -> ModelIdentity:
    """One runtime's identity; ``None`` leaves that detail out, as an unpinned backend would."""
    details = {"dtype": dtype, "weights_sha256": weights}
    return ModelIdentity(
        backend="huggingface",
        model_id="org/model",
        revision="a" * 40,
        fingerprint=fingerprint,
        details={key: value for key, value in details.items() if value is not None},
    )


def _study(candidate: CandidateSpec = CANDIDATE, **changes: Any) -> StudyRecord:
    cases = [
        CaseSpec(id=f"{role}-{index}", input=f"{role} prompt {index}", expected="yes", role=role)
        for role in ("target", "regression")
        for index in range(6)
    ]
    spec = StudySpec(
        schema_version="0.3",
        name="refusal lever referee",
        model=ModelSpec(backend="huggingface", model_id="org/model", revision="a" * 40),
        development_cases=cases,
        controls=[
            CandidateSpec(
                name="random-direction",
                program=ProgramSpec(kind="weight_scale", target=SITE.parameter, factor=0.9),
            )
        ],
        comparators=[
            ComparatorSpec(name="be-brief", kind="prompt", prefix="Be brief. "),
            ComparatorSpec(
                name="thinking-on", kind="generation", generation=GenerationSpec(top_k=20)
            ),
        ],
        seeds=[11, 12],
        truncation="fail",
        inference=RULE,
    )
    record: dict[str, Any] = {
        "id": "ab" * 16,
        "status": "completed",
        "spec": spec,
        "spec_digest": "6" * 64,
        "holdout": HoldoutCommitment(digest="7" * 64, count=16, targets=8, regressions=8),
        "created_at": "2026-11-20T00:00:00Z",
        "usage": StudyUsage(),
        "model_identity": _identity(),
        "nominee_attempt_id": "cd" * 16,
        "nominee_digest": digest(candidate),
        "finished_at": "2026-11-20T01:00:00Z",
        "final_result": StudyAssessment(
            scorer_version="nrly-scorers/3",
            verdict="observed_improvement",
            recommendation="candidate_supported",
            scope="real_model",
            inference=[_pair(), _pair(reference="control:random-direction")],
        ),
        "final_error_code": None,
    }
    return StudyRecord(**{**record, **changes})


def test_verdict_from_study_records_the_registered_design(tmp_path: Path) -> None:
    study = _study()
    verdict = verdict_from_study(study, CANDIDATE, _ops())
    assert verdict.study_id == study.id and verdict.study_spec_digest == study.spec_digest
    assert verdict.holdout == study.holdout
    assert verdict.operations_digest == operations_digest(_ops())
    assert verdict.nominee_digest == digest(CANDIDATE)
    # The weights the study loaded, taken from its pinned identity and not from its spec.
    assert study.model_identity is not None
    assert verdict.base == PortableBase.from_identity(study.model_identity) == _base()
    assert verdict.runtime_fingerprint == REFEREE_RUNTIME
    assert verdict.inference_rule == RULE and verdict.truncation == "fail"
    assert verdict.seeds == [11, 12]
    assert verdict.controls == ["random-direction"]
    assert verdict.comparators == ["prompt:be-brief", "generation:thinking-on"]
    assert verdict.development_cases == 12
    assert verdict.assessment == study.final_result
    assert verdict.final_error_code is None and verdict.finished_at == study.finished_at

    # The whole path: derive, dial, referee, ink.
    levers = LeverMap(tmp_path)
    inked = levers.attach_verdict(_dialled(levers).id, verdict)
    assert inked.status == "inked" and inked.verdict == verdict


def test_a_study_verdict_only_counts_for_the_weights_it_loaded(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    dialled = _dialled(levers)
    # The same repository, revision, nominee and edit, but the study loaded other bytes,
    # or these bytes at another dtype.
    strangers = [
        (_identity(weights=OTHER_WEIGHTS), _base(OTHER_WEIGHTS)),
        (_identity(dtype="torch.float16"), _base(dtype="torch.float16")),
    ]
    for identity, base in strangers:
        foreign = verdict_from_study(_study(model_identity=identity), CANDIDATE, _ops())
        assert foreign.base == base != dialled.base
        assert foreign.operations_digest == operations_digest(dialled.operations)
        with pytest.raises(NrlyError, match="weights this lever pins") as error:
            levers.attach_verdict(dialled.id, foreign)
        assert _code(error) == "verdict_mismatch"
    assert levers.get(dialled.id).status == "pencil"

    inked = levers.attach_verdict(dialled.id, verdict_from_study(_study(), CANDIDATE, _ops()))
    assert inked.status == "inked" and inked.verdict is not None
    assert inked.verdict.base == inked.base
    # Another machine that loaded the lever's own weights replicates it...
    again = verdict_from_study(_study(model_identity=_identity("5" * 64)), CANDIDATE, _ops())
    assert again.base == inked.base and again.runtime_fingerprint == "5" * 64
    replicated = levers.add_replication(inked.id, again)
    assert replicated.replications == [again]
    # ...and one that loaded other weights does not.
    stranger = _identity("a" * 64, weights=OTHER_WEIGHTS)
    foreign = verdict_from_study(_study(model_identity=stranger), CANDIDATE, _ops())
    with pytest.raises(NrlyError, match="weights this lever pins") as error:
        levers.add_replication(replicated.id, foreign)
    assert _code(error) == "verdict_mismatch"
    assert [entry.id for entry in levers.heads()] == [replicated.id]


def test_verdict_from_study_accepts_a_pinned_pipeline_and_default_seed() -> None:
    pipeline = CandidateSpec(
        name="refusal-lever",
        program=ProgramSpec(kind="pipeline", expected_base=REFEREE_RUNTIME, operations=_ops()),
    )
    unseeded = _study(pipeline).spec.model_copy(update={"seeds": None})
    verdict = verdict_from_study(_study(pipeline, spec=unseeded), pipeline, _ops())
    assert verdict.operations_digest == operations_digest(_ops())
    assert verdict.seeds == [0]


def test_a_failed_final_becomes_a_visible_pencil_verdict(tmp_path: Path) -> None:
    failed = _study(
        final_result=StudyAssessment(
            verdict="inconclusive", scope="real_model", issues=["final evaluation cancelled"]
        ),
        final_error_code="study_cancelled",
    )
    verdict = verdict_from_study(failed, CANDIDATE, _ops())
    assert verdict.final_error_code == "study_cancelled"
    levers = LeverMap(tmp_path)
    entry = levers.attach_verdict(_dialled(levers).id, verdict)
    # Neither a pass nor a refutation: the failure is on the record and the lever is pencil.
    assert entry.status == "pencil" and entry.verdict is not None
    assert entry.verdict.assessment.issues == ["final evaluation cancelled"]


def test_verdict_from_study_refusals() -> None:
    other = CandidateSpec(
        name="refusal-lever",
        program=ProgramSpec(kind="weight_scale", target=SITE.parameter, factor=0.5),
    )
    two_step = CandidateSpec(
        name="refusal-lever",
        program=ProgramSpec(
            kind="pipeline", expected_base=REFEREE_RUNTIME, operations=[*_ops(), *_ops(0.5)]
        ),
    )
    refusals: list[tuple[StudyRecord, CandidateSpec, list[WeightOperationSpec]]] = [
        (_study(status="open", final_result=None, finished_at=None), CANDIDATE, _ops()),
        (_study(status="finalizing"), CANDIDATE, _ops()),
        (_study(status="interrupted"), CANDIDATE, _ops()),
        (_study(final_result=None), CANDIDATE, _ops()),
        (_study(finished_at=None), CANDIDATE, _ops()),
        (_study(nominee_digest=None), CANDIDATE, _ops()),
        # A candidate the study never froze, even with operations that match it.
        (_study(), other, _ops(0.5)),
        # The frozen nominee, but a different edit than the one it compiles to.
        (_study(), CANDIDATE, _ops(0.5)),
        (_study(), CANDIDATE, []),
        (_study(), CANDIDATE, [*_ops(), *_ops()]),
        (_study(two_step), two_step, [*_ops(0.5), *_ops()]),  # same operations, reordered
        (_study(model_identity=None), CANDIDATE, _ops()),
        # An identity that does not say which bytes were loaded, or at which dtype.
        (_study(model_identity=_identity(weights=None)), CANDIDATE, _ops()),
        (_study(model_identity=_identity(dtype=None)), CANDIDATE, _ops()),
        (_study(model_identity=_identity(weights=None, dtype=None)), CANDIDATE, _ops()),
        (_study(model_identity=_identity(weights="not-a-digest")), CANDIDATE, _ops()),
        # Exact weights, but no runtime fingerprint to record the evidence under.
        (_study(model_identity=_identity("")), CANDIDATE, _ops()),
        (_study(id="not-a-study-id"), CANDIDATE, _ops()),
    ]
    for study, candidate, operations in refusals:
        with pytest.raises(NrlyError) as error:
            verdict_from_study(study, candidate, operations)
        assert _code(error) == "verdict_unavailable", (study.status, candidate.program.kind)
    assert verdict_from_study(_study(two_step), two_step, [*_ops(), *_ops(0.5)]).seeds == [11, 12]


def test_tensor_store_is_a_plain_artifact_store(tmp_path: Path) -> None:
    levers = LeverMap(tmp_path)
    assert isinstance(levers.tensors, TensorArtifactStore)
    assert levers.tensors.root == (tmp_path / "tensors").resolve()
    assert levers.entries == (tmp_path / "entries").resolve()
