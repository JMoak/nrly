"""Re-runnable derivations of lever directions, and the edits they compile to.

A recipe is trusted local authoring: it observes activations, does its arithmetic in
NumPy, and leaves behind a bounded direction artifact and declarative operations. No
code or path enters a program, and nothing here establishes that a lever works; that
is the referee's job.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from nrly.artifacts import TensorArtifactStore
from nrly.authoring import MAX_AUTHORING_ELEMENTS, FloatArray, WeightProgram, unit_direction
from nrly.contracts import (
    CaseSpec,
    GenerationSpec,
    ModelIdentity,
    NrlyError,
    PortableBase,
    WeightOperationSpec,
    digest,
)
from nrly.lever_contracts import Derivation, LeverSite, RecipeSpec, SiteScore

if TYPE_CHECKING:
    from nrly.backends.torch_backend import TorchBackend

# Prompts and module paths in; one mapping per prompt, in prompt order, from each module
# path to that module's output at the final prompt token.
ActivationSource = Callable[[Sequence[CaseSpec], Sequence[str]], list[dict[str, Sequence[float]]]]

# The kernel captures activations as float32. Within that range none of the float64 sums
# and squares below can overflow, so every recorded statistic is finite.
_MAX_MAGNITUDE = float(np.finfo(np.float32).max)
# A mean difference or a held-out spread below this fraction of the site's largest
# activation norm is rounding, not signal. Dividing by it would rank noise above real sites.
_RELATIVE_FLOOR = 1e-12


def backend_activations(backend: TorchBackend, recipe: RecipeSpec) -> ActivationSource:
    """Observe a loaded model as the recipe specifies: final prompt token, nothing generated."""
    generation = GenerationSpec(max_new_tokens=1, enable_thinking=recipe.enable_thinking)

    def observe(
        cases: Sequence[CaseSpec], modules: Sequence[str],
    ) -> list[dict[str, Sequence[float]]]:
        return [dict(row) for row in backend.last_token_activations(cases, generation, modules)]

    return observe


def _site_matrix(observed: Sequence[Mapping[str, Sequence[float]]], module: str) -> FloatArray:
    rows = []
    for row in observed:
        if not isinstance(row, Mapping) or module not in row:
            raise NrlyError("invalid_activations", f"an observation lacks module {module}")
        rows.append(row[module])
    try:
        matrix = np.asarray(rows)
    except (ValueError, TypeError) as error:
        raise NrlyError(
            "invalid_activations", f"activations at {module} are ragged or not numeric",
        ) from error
    if (
        matrix.dtype.kind not in "fiu" or matrix.ndim != 2
        or not 1 <= matrix.size <= MAX_AUTHORING_ELEMENTS
    ):
        raise NrlyError(
            "invalid_activations",
            f"activations at {module} must be bounded real vectors of one shared width",
        )
    values = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(values).all() or float(np.abs(values).max()) > _MAX_MAGNITUDE:
        raise NrlyError(
            "invalid_activations", f"activations at {module} must be finite float32 values",
        )
    return values


def _mean_difference(positive: FloatArray, negative: FloatArray) -> tuple[FloatArray, float]:
    difference = positive.mean(axis=0) - negative.mean(axis=0)
    return difference, float(np.linalg.norm(difference))


def _score(
    site: LeverSite, positive: FloatArray, negative: FloatArray, floor: float,
) -> tuple[SiteScore, bool]:
    """Fit on even positions, score on odd ones; False marks a site that cannot be scored."""
    fit_positive, fit_negative = positive[0::2], negative[0::2]
    held_positive, held_negative = positive[1::2], negative[1::2]
    difference, norm = _mean_difference(fit_positive, fit_negative)
    separation, scoreable = 0.0, False
    if norm > floor:
        unit = unit_direction(difference)
        high, low = held_positive @ unit, held_negative @ unit
        pooled = math.sqrt(
            float(np.sum((high - high.mean()) ** 2) + np.sum((low - low.mean()) ** 2))
            / (high.size + low.size - 2)
        )
        if pooled > floor:
            separation, scoreable = float((high.mean() - low.mean()) / pooled), True
    score = SiteScore(
        site=site, separation=separation, difference_norm=norm,
        fit_cases=len(fit_positive) + len(fit_negative),
        selection_cases=len(held_positive) + len(held_negative),
    )
    return score, scoreable


def derive(
    recipe: RecipeSpec,
    activations: ActivationSource,
    identity: ModelIdentity,
    store: TensorArtifactStore,
) -> Derivation:
    """Run a difference-of-means recipe once on one exact checkpoint.

    Every prompt is observed once at every candidate module. At each site the unit
    difference of class means is fitted on the even-indexed prompts of each class and
    scored on the odd-indexed ones, so no site is chosen on the data that produced its
    direction. Every site is recorded. A site with no difference to fit, or no held-out
    spread to measure it against, records a separation of zero and is never chosen over
    a site that could be scored. The stored direction is refitted on every prompt at the
    chosen site. A low or negative separation is a retained null, not an error.
    """
    recipe = RecipeSpec.model_validate(recipe.model_dump())
    # Anything the finished record would reject is refused before the model is observed.
    base = PortableBase.from_identity(identity)
    if not re.fullmatch(r"[0-9a-f]{64}", identity.fingerprint):
        raise NrlyError("invalid_identity", "model identity lacks a runtime fingerprint digest")
    cases = [*recipe.positive, *recipe.negative]
    modules = list(dict.fromkeys(site.module for site in recipe.sites))
    observed = activations(cases, modules)
    if not isinstance(observed, Sequence) or len(observed) != len(cases):
        raise NrlyError(
            "invalid_activations", "expected one observation per contrast prompt, in order",
        )
    matrices = {module: _site_matrix(observed, module) for module in modules}
    floors = {
        module: _RELATIVE_FLOOR * float(np.linalg.norm(matrix, axis=1).max())
        for module, matrix in matrices.items()
    }
    split = len(recipe.positive)
    scored = []
    for site in recipe.sites:
        matrix = matrices[site.module]
        scored.append(_score(site, matrix[:split], matrix[split:], floors[site.module]))
    best: int | None = None
    for index, (score, scoreable) in enumerate(scored):
        if scoreable and (best is None or score.separation > scored[best][0].separation):
            best = index
    if best is None:
        raise NrlyError(
            "derivation_failed",
            "no candidate site has a direction that can be scored on the held-out prompts",
        )
    chosen = recipe.sites[best]
    matrix = matrices[chosen.module]
    difference, norm = _mean_difference(matrix[:split], matrix[split:])
    if norm <= floors[chosen.module]:
        raise NrlyError(
            "derivation_failed",
            f"the class means coincide over all prompts at {chosen.module}",
        )
    return Derivation(
        recipe=recipe, recipe_digest=digest(recipe), base=base,
        runtime_fingerprint=identity.fingerprint, chosen=chosen,
        direction=store.put(unit_direction(difference), dtype="float32"),
        scores=[score for score, _ in scored],
    )


def compile_removal(
    site: LeverSite, weights: ArrayLike, direction: ArrayLike, strength: float,
) -> list[WeightOperationSpec]:
    """Portable operations removing ``strength`` times a direction from what a tensor writes.

    ``weights`` are the current values of ``site.parameter`` in the [outputs, inputs]
    layout of ``nn.Linear``, so the direction's length is the row count. The result
    carries no base pin; whoever applies it binds it to a verified runtime.
    """
    if not math.isfinite(strength) or strength == 0:
        raise NrlyError("invalid_authoring", "a removal needs a finite, nonzero strength")
    if site.side != "output":
        raise NrlyError("invalid_authoring", "a removal edits the output side of a site")
    # The builder validates against a base; only the unpinned operations leave this function.
    program = WeightProgram("unpinned").remove_output_direction(
        site.parameter, weights, direction, strength=strength,
    ).build()
    return list(program.operations)
