from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from nrly.artifacts import TensorArtifactStore
from nrly.authoring import WeightProgram, unit_direction
from nrly.backends.tiny import TinyBackend
from nrly.contracts import (
    CaptureSpec,
    CaseSpec,
    GenerationSpec,
    ModelIdentity,
    ModelSpec,
    NrlyError,
    PortableBase,
    ProgramSpec,
    WeightOperationSpec,
    digest,
)
from nrly.kernel import capture_activations, intervene, tensor_digest
from nrly.lever_contracts import Derivation, LeverSite, RecipeSpec
from nrly.recipes import backend_activations, compile_removal, derive

WIDTH = 16
TRUTH = unit_direction(np.cos(np.arange(WIDTH) * 1.7) + 0.3)
E0, E1 = np.eye(WIDTH)[0], np.eye(WIDTH)[1]
# A synthetic identity: these tests check arithmetic, never a model.
IDENTITY = ModelIdentity(
    backend="synthetic", model_id="fixture/activations", revision="1", fingerprint="1" * 64,
    details={"dtype": "float32", "weights_sha256": "2" * 64},
)
Vector = Callable[[bool, int, str], Sequence[float]]


class Source:
    """Activations as a known function of (class, position within the class, module)."""

    def __init__(self, vector: Vector) -> None:
        self.vector = vector
        self.calls: list[tuple[list[str], list[str]]] = []

    def __call__(
        self, cases: Sequence[CaseSpec], modules: Sequence[str],
    ) -> list[dict[str, Any]]:
        self.calls.append(([case.id for case in cases], list(modules)))
        rows = []
        for case in cases:
            label, index = case.id.split("-")
            rows.append({
                module: np.asarray(self.vector(label == "pos", int(index), module), float).tolist()
                for module in modules
            })
        return rows

    def matrix(self, module: str, positive: bool, count: int) -> np.ndarray:
        return np.array([self.vector(positive, index, module) for index in range(count)], float)


def _noise(positive: bool, index: int, module: str, sigma: float) -> np.ndarray:
    seed = zlib.crc32(f"{module}|{positive}|{index}".encode())
    return np.random.default_rng(seed).normal(0.0, sigma, WIDTH)


def _site(module: str) -> LeverSite:
    return LeverSite(parameter=f"{module}.out.weight", module=module)


def _recipe(
    sites: list[LeverSite], positive: int = 8, negative: int = 8, thinking: bool = False,
) -> RecipeSpec:
    return RecipeSpec(
        behavior="contrast-fixture",
        positive=[CaseSpec(id=f"pos-{i}", input=f"positive prompt {i}") for i in range(positive)],
        negative=[CaseSpec(id=f"neg-{i}", input=f"negative prompt {i}") for i in range(negative)],
        sites=sites, enable_thinking=thinking,
    )


def _separation(
    fit_positive: np.ndarray, fit_negative: np.ndarray,
    score_positive: np.ndarray, score_negative: np.ndarray,
) -> tuple[float, float]:
    """Independent statement of the score: standardized gap along the fitted direction."""
    difference = fit_positive.mean(axis=0) - fit_negative.mean(axis=0)
    unit = difference / np.linalg.norm(difference)
    high, low = score_positive @ unit, score_negative @ unit
    pooled = np.sqrt(
        ((len(high) - 1) * high.var(ddof=1) + (len(low) - 1) * low.var(ddof=1))
        / (len(high) + len(low) - 2)
    )
    return float((high.mean() - low.mean()) / pooled), float(np.linalg.norm(difference))


def _held_out(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    return _separation(positive[0::2], negative[0::2], positive[1::2], negative[1::2])


def _in_sample(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    return _separation(positive[0::2], negative[0::2], positive[0::2], negative[0::2])


def _separating(positive: bool, index: int, module: str) -> np.ndarray:
    """blocks.1.mlp separates the classes along TRUTH; every other module is noise."""
    if module == "blocks.1.mlp":
        sign = 1.5 if positive else -1.5
        return 3.0 + sign * TRUTH + _noise(positive, index, module, 0.1)
    return 3.0 + _noise(positive, index, module, 1.0)


def _degenerate(positive: bool, index: int, module: str) -> np.ndarray:
    sign = 1.0 if positive else -1.0
    if module == "constant":  # identical class means: no direction to fit
        return np.full(WIDTH, 0.375)
    if module == "flat":  # distinct class means without spread: nothing to score against
        return sign * TRUTH
    # The held-out half contradicts the fitted half, so the real score is negative.
    return sign * (2.0 if index % 2 == 0 else -1.0) * E0 + _noise(positive, index, module, 0.2)


def _apply(weights: np.ndarray, operation: WeightOperationSpec) -> np.ndarray:
    rows, columns = operation.shape
    left = np.array(operation.left).reshape(rows, operation.rank)
    right = np.array(operation.right).reshape(operation.rank, columns)
    return weights + operation.factor * left @ right


def test_derive_scores_every_site_and_stores_the_separating_direction(tmp_path: Path) -> None:
    null, lever = _site("blocks.0.mlp"), _site("blocks.1.mlp")
    recipe = _recipe([null, lever], positive=9, negative=8)
    source, store = Source(_separating), TensorArtifactStore(tmp_path)
    derivation = derive(recipe, source, IDENTITY, store)

    prompts = [case.id for case in (*recipe.positive, *recipe.negative)]
    assert source.calls == [(prompts, ["blocks.0.mlp", "blocks.1.mlp"])]
    assert derivation.chosen == lever
    assert [score.site for score in derivation.scores] == [null, lever]
    for score in derivation.scores:
        positive = source.matrix(score.site.module, True, 9)
        negative = source.matrix(score.site.module, False, 8)
        separation, norm = _held_out(positive, negative)
        assert score.separation == pytest.approx(separation, rel=1e-9)
        assert score.difference_norm == pytest.approx(norm, rel=1e-9)
        assert (score.fit_cases, score.selection_cases) == (9, 8)
    assert derivation.scores[1].separation > 10 > abs(derivation.scores[0].separation)

    stored = store.get(derivation.direction)
    assert stored.dtype == np.float32 and stored.shape == (WIDTH,)
    assert float(np.linalg.norm(stored)) == pytest.approx(1.0, abs=1e-6)
    assert float(stored @ TRUTH) > 0.99
    assert store.inspect(derivation.direction.id) == derivation.direction

    assert derivation.recipe == recipe and derivation.recipe_digest == digest(recipe)
    assert derivation.base == PortableBase.from_identity(IDENTITY)
    assert derivation.runtime_fingerprint == IDENTITY.fingerprint
    assert Derivation.model_validate_json(derivation.model_dump_json()) == derivation
    assert derive(recipe, source, IDENTITY, store) == derivation


def test_site_is_chosen_on_held_out_prompts_and_direction_is_refitted_on_all(
    tmp_path: Path,
) -> None:
    def vector(positive: bool, index: int, module: str) -> np.ndarray:
        sign = 1.0 if positive else -1.0
        if module == "memorized":
            # A huge class gap that exists only in the prompts used for fitting.
            if index % 2 == 0:
                return sign * 4.0 * E0 + _noise(positive, index, module, 0.01)
            return _noise(positive, index, module, 1.0)
        return sign * E1 + _noise(positive, index, module, 0.25)

    memorized, honest = _site("memorized"), _site("honest")
    recipe = _recipe([memorized, honest])
    source, store = Source(vector), TensorArtifactStore(tmp_path)
    derivation = derive(recipe, source, IDENTITY, store)

    data = {
        site.module: (source.matrix(site.module, True, 8), source.matrix(site.module, False, 8))
        for site in recipe.sites
    }
    # Scoring on the fitting half would have chosen the memorized site.
    assert _in_sample(*data["memorized"])[0] > 10 * _in_sample(*data["honest"])[0] > 0
    assert derivation.chosen == honest
    for score in derivation.scores:
        separation, norm = _held_out(*data[score.site.module])
        assert score.separation == pytest.approx(separation, rel=1e-9)
        assert score.difference_norm == pytest.approx(norm, rel=1e-9)
    assert derivation.scores[1].separation > 3 > abs(derivation.scores[0].separation)
    # The fit used the even positions: only they carry the memorized site's gap of 8, while
    # the odd positions differ by unit noise averaged over four prompts (about 2.8).
    odd_norm = np.linalg.norm(
        data["memorized"][0][1::2].mean(axis=0) - data["memorized"][1][1::2].mean(axis=0)
    )
    assert derivation.scores[0].difference_norm > 7 > 4 > odd_norm

    positive, negative = data["honest"]
    refit = unit_direction(positive.mean(axis=0) - negative.mean(axis=0))
    half = unit_direction(positive[0::2].mean(axis=0) - negative[0::2].mean(axis=0))
    stored = store.get(derivation.direction)
    np.testing.assert_allclose(stored, refit, atol=1e-6)
    assert np.abs(half - refit).max() > 1e-3


def test_sites_sharing_a_module_are_observed_once_and_ties_keep_recipe_order(
    tmp_path: Path,
) -> None:
    first = LeverSite(parameter="blocks.1.mlp.up.weight", module="blocks.1.mlp")
    second = LeverSite(parameter="blocks.1.mlp.down.weight", module="blocks.1.mlp")
    source = Source(_separating)
    derivation = derive(_recipe([first, second]), source, IDENTITY, TensorArtifactStore(tmp_path))
    assert [modules for _, modules in source.calls] == [["blocks.1.mlp"]]
    assert derivation.scores[0].separation == derivation.scores[1].separation > 10
    assert derivation.chosen == first


def test_degenerate_sites_score_zero_and_never_beat_a_site_that_was_scored(
    tmp_path: Path,
) -> None:
    constant, flat, reversed_site = _site("constant"), _site("flat"), _site("reversed")
    derivation = derive(
        _recipe([constant, flat, reversed_site]), Source(_degenerate), IDENTITY,
        TensorArtifactStore(tmp_path),
    )
    nothing_to_fit, nothing_to_score, real = derivation.scores
    assert nothing_to_fit.separation == 0.0 and nothing_to_fit.difference_norm == 0.0
    assert nothing_to_score.separation == 0.0
    assert nothing_to_score.difference_norm == pytest.approx(2.0)
    # Zero outranks a negative score numerically, yet the scored site is the one chosen.
    assert real.separation < -3
    assert derivation.chosen == reversed_site


def test_differences_at_rounding_scale_are_not_mistaken_for_separation(tmp_path: Path) -> None:
    ulp = float(np.finfo(np.float64).eps)

    def vector(positive: bool, index: int, module: str) -> np.ndarray:
        if module != "jitter":
            return _separating(positive, index, module)
        # A class gap of 129 ulp with one ulp of spread, on activations of norm 4.
        row = np.ones(WIDTH)
        row[0] += (1.0 if positive else -1.0) * (64 + (index // 2) % 2) * ulp
        return row

    jitter, lever = _site("jitter"), _site("blocks.1.mlp")
    source = Source(vector)
    naive = {
        site.module: _held_out(source.matrix(site.module, True, 8),
                               source.matrix(site.module, False, 8))[0]
        for site in (jitter, lever)
    }
    # Dividing rounding by rounding would score the jitter far above the real lever.
    assert naive["jitter"] > 5 * naive["blocks.1.mlp"] > 50
    derivation = derive(_recipe([jitter, lever]), source, IDENTITY, TensorArtifactStore(tmp_path))
    assert derivation.scores[0].separation == 0.0
    assert 0 < derivation.scores[0].difference_norm < 1e-13
    assert derivation.scores[1].separation == pytest.approx(naive["blocks.1.mlp"], rel=1e-9)
    assert derivation.chosen == lever


@pytest.mark.parametrize("module", ["constant", "flat"])
def test_derivation_fails_when_no_site_can_be_scored(tmp_path: Path, module: str) -> None:
    with pytest.raises(NrlyError, match="held-out") as error:
        derive(_recipe([_site(module)]), Source(_degenerate), IDENTITY,
               TensorArtifactStore(tmp_path))
    assert error.value.code == "derivation_failed"
    assert not list(tmp_path.iterdir())


def test_derivation_fails_when_the_refit_over_all_prompts_has_no_direction(
    tmp_path: Path,
) -> None:
    # The even half fits [3, 0] and the odd half scores it at -6, a real (negative) score,
    # but over all eight prompts both class means are exactly the origin.
    table = {
        True: [[3, 1], [-3, 1], [3, -1], [-3, -1]],
        False: [[1, 0], [-0.5, 0.5], [-1, 0], [0.5, -0.5]],
    }
    positive, negative = np.array(table[True], float), np.array(table[False], float)
    assert _held_out(positive, negative)[0] == pytest.approx(-6.0)
    source = Source(lambda positive, index, module: table[positive][index])
    recipe = _recipe([_site("cancel")], positive=4, negative=4)
    with pytest.raises(NrlyError, match="all prompts") as error:
        derive(recipe, source, IDENTITY, TensorArtifactStore(tmp_path))
    assert error.value.code == "derivation_failed"
    assert not list(tmp_path.iterdir())


def _drop_module(rows: list[Any]) -> None:
    del rows[3]["blocks.1.mlp"]


def _ragged(rows: list[Any]) -> None:
    rows[3]["blocks.1.mlp"] = rows[3]["blocks.1.mlp"][:-1]


def _nan(rows: list[Any]) -> None:
    rows[0]["blocks.0.mlp"][2] = float("nan")


def _infinite(rows: list[Any]) -> None:
    rows[-1]["blocks.1.mlp"][0] = float("-inf")


def _beyond_float32(rows: list[Any]) -> None:
    rows[1]["blocks.1.mlp"][0] = 1e39


def _text(rows: list[Any]) -> None:
    rows[2]["blocks.0.mlp"] = ["a"] * WIDTH


def _scalars(rows: list[Any]) -> None:
    for row in rows:
        row["blocks.0.mlp"] = 1.0


def _nested(rows: list[Any]) -> None:
    for row in rows:
        row["blocks.0.mlp"] = [row["blocks.0.mlp"]]


def _empty(rows: list[Any]) -> None:
    for row in rows:
        row["blocks.1.mlp"] = []


def _missing_prompt(rows: list[Any]) -> None:
    rows.pop()


def _extra_prompt(rows: list[Any]) -> None:
    rows.append(dict(rows[0]))


def _not_a_mapping(rows: list[Any]) -> None:
    rows[0] = list(rows[0].values())


@pytest.mark.parametrize(
    "corrupt",
    [
        _drop_module, _ragged, _nan, _infinite, _beyond_float32, _text, _scalars, _nested,
        _empty, _missing_prompt, _extra_prompt, _not_a_mapping,
    ],
)
def test_malformed_activations_are_refused(
    tmp_path: Path, corrupt: Callable[[list[Any]], None],
) -> None:
    def source(cases: Sequence[CaseSpec], modules: Sequence[str]) -> list[dict[str, Any]]:
        rows = Source(_separating)(cases, modules)
        corrupt(rows)
        return rows

    recipe = _recipe([_site("blocks.0.mlp"), _site("blocks.1.mlp")])
    with pytest.raises(NrlyError) as error:
        derive(recipe, source, IDENTITY, TensorArtifactStore(tmp_path))
    assert error.value.code == "invalid_activations"
    assert not list(tmp_path.iterdir())


def test_a_source_that_returns_no_observations_is_refused(tmp_path: Path) -> None:
    recipe = _recipe([_site("blocks.1.mlp")])
    with pytest.raises(NrlyError) as error:
        derive(recipe, lambda cases, modules: None, IDENTITY,  # type: ignore[arg-type,return-value]
               TensorArtifactStore(tmp_path))
    assert error.value.code == "invalid_activations"


def test_unpinnable_identity_and_tampered_recipe_fail_before_any_observation(
    tmp_path: Path,
) -> None:
    source, store = Source(_separating), TensorArtifactStore(tmp_path)
    recipe = _recipe([_site("blocks.1.mlp")])
    for update in ({"details": {"dtype": "float32"}}, {"fingerprint": "not-a-digest"}):
        with pytest.raises(NrlyError) as error:
            derive(recipe, source, IDENTITY.model_copy(update=update), store)
        assert error.value.code == "invalid_identity"
    del recipe.negative[1:]  # the contract is frozen, but its lists are not
    with pytest.raises(ValidationError):
        derive(recipe, source, IDENTITY, store)
    assert not source.calls


class RecordingBackend:
    def __init__(self) -> None:
        self.source = Source(_separating)
        self.calls: list[tuple[list[CaseSpec], GenerationSpec, list[str]]] = []

    def last_token_activations(
        self, cases: Sequence[CaseSpec], generation: GenerationSpec, targets: Sequence[str],
    ) -> list[dict[str, Any]]:
        self.calls.append((list(cases), generation, list(targets)))
        return self.source(cases, targets)


@pytest.mark.parametrize("thinking", [False, True])
def test_backend_activations_observe_one_token_in_the_recipes_thinking_mode(
    tmp_path: Path, thinking: bool,
) -> None:
    null, lever = _site("blocks.0.mlp"), _site("blocks.1.mlp")
    recipe = _recipe([null, lever], thinking=thinking)
    backend = RecordingBackend()
    derivation = derive(
        recipe, backend_activations(backend, recipe),  # type: ignore[arg-type]
        IDENTITY, TensorArtifactStore(tmp_path),
    )
    [(cases, generation, targets)] = backend.calls
    assert generation == GenerationSpec(max_new_tokens=1, enable_thinking=thinking)
    assert cases == [*recipe.positive, *recipe.negative]
    assert targets == ["blocks.0.mlp", "blocks.1.mlp"]
    assert derivation.chosen == lever


def test_a_backend_that_cannot_observe_prompts_reports_its_own_refusal() -> None:
    recipe = _recipe([LeverSite(parameter="hidden.weight", module="hidden")])
    with TinyBackend(ModelSpec()) as backend:
        observe = backend_activations(backend, recipe)
        with pytest.raises(NrlyError) as error:
            observe(recipe.positive, ["hidden"])
    assert error.value.code == "unsupported_operation"


@pytest.mark.parametrize("strength", [1.0, 0.5, -2.0, 1.5])
def test_compile_removal_is_one_rank_one_operation_equal_to_the_projection(
    strength: float,
) -> None:
    rng = np.random.default_rng(11)
    weights, raw, probe = rng.normal(size=(5, 7)), rng.normal(size=5), rng.normal(size=5)
    unit = raw / np.linalg.norm(raw)
    site = _site("blocks.1.mlp")
    [operation] = compile_removal(site, weights, 3 * raw, strength)
    assert WeightOperationSpec.model_validate(operation.model_dump()) == operation
    assert (operation.kind, operation.target) == ("weight_low_rank", "blocks.1.mlp.out.weight")
    assert (operation.shape, operation.rank, operation.factor) == ([5, 7], 1, -strength)
    # Twelve numbers for a 35-element tensor: the dense delta is never materialized.
    assert (len(operation.left), len(operation.right), operation.values) == (5, 7, [])

    updated = _apply(weights, operation)
    np.testing.assert_allclose(
        updated, weights - strength * np.outer(unit, unit) @ weights, atol=1e-12,
    )
    np.testing.assert_allclose(unit @ updated, (1 - strength) * (unit @ weights), atol=1e-12)
    orthogonal = probe - (probe @ unit) * unit
    np.testing.assert_allclose(orthogonal @ updated, orthogonal @ weights, atol=1e-12)


def test_full_strength_removal_leaves_nothing_written_along_the_direction() -> None:
    rng = np.random.default_rng(3)
    weights, direction = rng.normal(size=(64, 48)), rng.normal(size=64)
    unit = direction / np.linalg.norm(direction)
    assert np.abs(unit @ weights).max() > 0.1
    [operation] = compile_removal(_site("blocks.1.mlp"), weights, direction, 1)
    updated = _apply(weights, operation)
    assert np.abs(unit @ updated).max() < 1e-12
    # Any input now produces an output with no component along the direction.
    assert np.abs(unit @ (updated @ rng.normal(size=(48, 10)))).max() < 1e-12


def test_remove_output_direction_mirrors_the_input_side_helper() -> None:
    weights = np.array([[2.0, 3.0], [4.0, 5.0]])
    builder = WeightProgram("base")
    assert builder.remove_output_direction("weight", weights, [2, 0]) is builder
    program = builder.build()
    [operation] = program.operations
    assert program.expected_base == "base" and operation.kind == "weight_low_rank"
    assert len(operation.left) + len(operation.right) == 4
    updated = _apply(weights, operation)
    np.testing.assert_allclose(updated, [[0, 0], [4, 5]])
    # The same direction on the input side removes a column instead of a row ...
    mirrored = WeightProgram("base").remove_input_direction("weight", weights, [2, 0]).build()
    np.testing.assert_allclose(_apply(weights, mirrored.operations[0]), [[0, 3], [0, 5]])
    # ... and the two helpers are transposes of each other.
    transposed = WeightProgram("base").remove_input_direction("weight", weights.T, [2, 0])
    np.testing.assert_allclose(_apply(weights.T, transposed.build().operations[0]).T, updated)
    half = WeightProgram("base").remove_output_direction("weight", weights, [1, 1], strength=0.5)
    np.testing.assert_allclose(
        _apply(weights, half.build().operations[0]), [[0.5, 1.0], [2.5, 3.0]],
    )


@pytest.mark.parametrize(
    ("weights", "direction"),
    [
        (np.ones((5, 7)), np.ones(7)),
        (np.ones(5), np.ones(5)),
        (np.ones((5, 7, 2)), np.ones(5)),
        (np.ones((5, 7)), np.zeros(5)),
        (np.ones((5, 7)), np.ones((5, 1))),
        (np.full((5, 7), np.nan), np.ones(5)),
    ],
)
def test_output_removal_refuses_shapes_and_values_it_cannot_compile(
    weights: np.ndarray, direction: np.ndarray,
) -> None:
    builder = WeightProgram("base")
    with pytest.raises(NrlyError) as error:
        builder.remove_output_direction("weight", weights, direction)
    assert error.value.code == "invalid_authoring"
    assert builder.build().kind == "identity"
    with pytest.raises(NrlyError) as error:
        compile_removal(_site("blocks.1.mlp"), weights, direction, 1.0)
    assert error.value.code == "invalid_authoring"


@pytest.mark.parametrize("strength", [0, 0.0, float("nan"), float("inf"), float("-inf")])
def test_compile_removal_refuses_zero_and_nonfinite_strengths(strength: float) -> None:
    with pytest.raises(NrlyError, match="strength") as error:
        compile_removal(_site("blocks.1.mlp"), np.ones((5, 7)), np.ones(5), strength)
    assert error.value.code == "invalid_authoring"


def test_compile_removal_refuses_input_side_sites_and_oversized_factors() -> None:
    site = LeverSite(parameter="blocks.1.mlp.in.weight", module="blocks.1.mlp", side="input")
    with pytest.raises(NrlyError, match="output side") as error:
        compile_removal(site, np.ones((5, 7)), np.ones(5), 1.0)
    assert error.value.code == "invalid_authoring"
    with pytest.raises(NrlyError) as error:
        compile_removal(_site("blocks.1.mlp"), np.ones((2, 65535)), np.ones(2), 1.0)
    assert error.value.code == "budget_exceeded"


def test_removal_executes_through_the_kernel_and_restores_the_tiny_model() -> None:
    case = CaseSpec(id="sum", input="[2,4]", expected="6", scorer="numeric")
    site = LeverSite(parameter="output.weight", module="output")
    with TinyBackend(ModelSpec()) as backend:
        before = tensor_digest(backend.model)
        weights = backend.read_parameter("output.weight")
        np.testing.assert_array_equal(weights, [[0.5, 0.5]])
        outputs = {}
        for strength in (0.5, 1.0, -1.0):
            program = ProgramSpec(
                kind="pipeline", expected_base=backend.identity.fingerprint,
                operations=compile_removal(site, weights, [1.0], strength),
            )
            result = backend.run_case(case, program, [], GenerationSpec())
            outputs[strength] = result.output
            [change] = result.weight_changes
            assert (change.target, change.operation_kind) == ("output.weight", "weight_low_rank")
            assert change.changed_elements == 2
            assert tensor_digest(backend.model) == before
        # output.weight has one row, so its whole output is the direction: 3 * (1 - strength).
        assert outputs == {0.5: "1.5", 1.0: "0", -1.0: "6"}
        assert backend.run_case(case, ProgramSpec(), [], GenerationSpec()).output == "3"


def test_derived_direction_compiles_to_an_edit_that_silences_it_in_the_real_module(
    tmp_path: Path,
) -> None:
    truth = np.array([0.6, 0.8])

    def vector(positive: bool, index: int, module: str) -> np.ndarray:
        noise = _noise(positive, index, module, 0.05)[:2]
        return 1.0 + (1.0 if positive else -1.0) * truth + noise

    site = LeverSite(parameter="hidden.weight", module="hidden")
    store = TensorArtifactStore(tmp_path)
    with TinyBackend(ModelSpec()) as backend:
        before = tensor_digest(backend.model)
        derivation = derive(_recipe([site]), Source(vector), backend.identity, store)
        assert derivation.base.matches(backend.identity)
        direction = store.get(derivation.direction).astype(np.float64)
        assert float(direction @ truth) > 0.99
        program = ProgramSpec(
            kind="pipeline", expected_base=backend.identity.fingerprint,
            operations=compile_removal(
                derivation.chosen, backend.read_parameter(derivation.chosen.parameter),
                direction, 1.0,
            ),
        )
        captures = [CaptureSpec(target="hidden", max_elements=16)]
        value = torch.tensor([[2.0, 4.0]])

        def hidden_output() -> np.ndarray:
            with capture_activations(backend.model, captures) as recorded:
                backend.model(value)
            return np.array(recorded["hidden"].values)

        assert abs(hidden_output() @ direction) > 4
        with intervene(backend.model, program, backend.identity.fingerprint) as changes:
            assert changes[0].operation_kind == "weight_low_rank"
            edited = backend.read_parameter("hidden.weight")
            unit = direction / np.linalg.norm(direction)
            np.testing.assert_allclose(edited, np.eye(2) - np.outer(unit, unit), atol=1e-6)
            assert abs(hidden_output() @ direction) < 1e-5
        assert tensor_digest(backend.model) == before
