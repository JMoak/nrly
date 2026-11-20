"""Paired inference for controlled studies.

Cases are the independent unit. Repeated seeds are nested inside a case and are
averaged before any test, so seed repeats never inflate the sample size. Results
are deterministic for the pinned NumPy: the bootstrap uses PCG64 with a registered
seed and a chunk size that depends only on the number of cases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Literal

import numpy as np

from nrly.contracts import NrlyError
from nrly.study_contracts import EffectEstimate, InferenceRule, PairedInference

_CHUNK_ELEMENTS = 4_000_000
Verdict = Literal["observed_improvement", "no_change", "regressed", "inconclusive"]


def case_means(blocks: Sequence[Sequence[float]]) -> list[float]:
    """Mean paired difference per case across aligned seed blocks.

    ``blocks[seed][case]`` is candidate score minus reference score for one execution.
    Every block must list the same cases in the same order.
    """
    if not blocks or not blocks[0]:
        raise NrlyError("invalid_statistics", "paired inference requires at least one case")
    width = len(blocks[0])
    if any(len(block) != width for block in blocks):
        raise NrlyError("invalid_statistics", "seed blocks must cover the same ordered cases")
    values = [math.fsum(block[index] for block in blocks) / len(blocks) for index in range(width)]
    if any(not math.isfinite(value) or abs(value) > 1 for value in values):
        raise NrlyError("invalid_statistics", "paired differences must lie between -1 and 1")
    return values


def sign_test(differences: Sequence[float]) -> float:
    """Exact two-sided binomial p-value on the signs of the nonzero differences."""
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    trials = positive + negative
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, index) for index in range(min(positive, negative) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2**trials)))


def bootstrap_interval(
    values: Sequence[float], *, alpha: float, resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile interval for the mean, resampling whole cases with replacement."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise NrlyError("invalid_statistics", "bootstrap requires finite case-level values")
    if not 0 < alpha < 1 or resamples < 1:
        raise NrlyError("invalid_statistics", "invalid bootstrap level or resample count")
    generator = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(resamples, dtype=np.float64)
    step = max(1, _CHUNK_ELEMENTS // data.size)
    for start in range(0, resamples, step):
        count = min(step, resamples - start)
        indices = generator.integers(0, data.size, size=(count, data.size))
        means[start : start + count] = data[indices].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return max(-1.0, float(low)), min(1.0, float(high))


def paired_effect(blocks: Sequence[Sequence[float]], rule: InferenceRule) -> EffectEstimate:
    """Estimate candidate-minus-reference on one role's cases under a registered rule."""
    values = case_means(blocks)
    low, high = bootstrap_interval(
        values, alpha=rule.alpha, resamples=rule.resamples, seed=rule.bootstrap_seed
    )
    mean = math.fsum(values) / len(values)
    return EffectEstimate(
        cases=len(values),
        seeds=len(blocks),
        mean_difference=max(-1.0, min(1.0, mean)),
        ci_low=low,
        ci_high=high,
        confidence=1 - rule.alpha,
        improved_cases=sum(value > 0 for value in values),
        regressed_cases=sum(value < 0 for value in values),
        tied_cases=sum(value == 0 for value in values),
        sign_test_p=sign_test(values),
    )


def paired_inference(
    reference: str,
    target_blocks: Sequence[Sequence[float]],
    regression_blocks: Sequence[Sequence[float]],
    rule: InferenceRule,
) -> PairedInference:
    """Compare the candidate with one reference arm on target and regression cases.

    Superiority needs both an interval above zero and a significant sign test. The
    two-sided interval's lower bound is used for non-inferiority, which is conservative
    for the one-sided claim.
    """
    target = paired_effect(target_blocks, rule)
    regression = paired_effect(regression_blocks, rule)
    significant = target.sign_test_p < rule.alpha
    return PairedInference(
        reference=reference,
        target=target,
        regression=regression,
        target_superior=target.ci_low > 0 and significant,
        target_harmed=target.ci_high < 0 and significant,
        regression_non_inferior=regression.ci_low >= -rule.regression_margin,
        regression_inferior=regression.ci_high < -rule.regression_margin,
    )


def inferential_verdict(pairs: Sequence[PairedInference]) -> tuple[Verdict, str | None]:
    """Registered verdict rule; ``pairs[0]`` must be the unedited original.

    Requiring superiority over every reference is an intersection-union test, so the
    joint claim keeps the level of each individual comparison. ``no_change`` means no
    supported target effect with regressions preserved; it is not an equivalence claim.
    """
    if not pairs:
        raise NrlyError("invalid_statistics", "a verdict requires the original reference arm")
    base = pairs[0]
    if base.target_harmed or base.regression_inferior:
        return "regressed", None
    if not base.regression_non_inferior:
        return (
            "inconclusive",
            "regression non-inferiority against the original was not established",
        )
    if not base.target_superior:
        return "no_change", None
    if all(pair.target_superior for pair in pairs):
        return "observed_improvement", None
    return "inconclusive", "candidate did not improve targets beyond every registered control"
