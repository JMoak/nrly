from __future__ import annotations

import pytest

from nrly.contracts import NrlyError
from nrly.evaluation.statistics import (
    bootstrap_interval,
    case_means,
    inferential_verdict,
    paired_effect,
    paired_inference,
    sign_test,
)
from nrly.study_contracts import InferenceRule, PairedInference

RULE = InferenceRule(resamples=2000)


def test_sign_test_matches_exact_binomial_tails() -> None:
    assert sign_test([1, 1, 1, 1, 1, 1]) == pytest.approx(2 / 64)
    assert sign_test([1, 1, 1, 1, 1, -1]) == pytest.approx(14 / 64)
    assert sign_test([-1] * 6) == pytest.approx(2 / 64)
    # Ties carry no direction, and a balanced split is capped at one.
    assert sign_test([0, 0, 0]) == 1.0
    assert sign_test([1, -1, 0, 0]) == 1.0
    assert sign_test([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0]) == pytest.approx(2 / 64)


def test_five_cases_can_never_be_significant_at_five_percent() -> None:
    assert sign_test([1] * 5) == pytest.approx(2 / 32)
    assert sign_test([1] * 5) > 0.05


def test_seed_repeats_are_averaged_inside_a_case_not_counted_as_cases() -> None:
    blocks = [[1, 0, -1], [1, 0, 1], [1, 0, 0]]
    assert case_means(blocks) == pytest.approx([1.0, 0.0, 0.0])
    effect = paired_effect([[1] * 6 for _ in range(8)], RULE)
    assert (effect.cases, effect.seeds) == (6, 8)
    # Eight seeds of six cases have the p-value of six cases, never of forty-eight.
    assert effect.sign_test_p == pytest.approx(2 / 64)


def test_misaligned_or_empty_blocks_are_refused() -> None:
    for blocks in ([], [[]], [[1, 0], [1]]):
        with pytest.raises(NrlyError) as error:
            case_means(blocks)
        assert error.value.code == "invalid_statistics"
    with pytest.raises(NrlyError):
        case_means([[2.0]])


def test_bootstrap_is_deterministic_bounded_and_degenerate_on_constants() -> None:
    values = [1, 1, 0, 1, -1, 1, 0, 1, 1, 0, 1, 1]
    first = bootstrap_interval(values, alpha=0.05, resamples=2000, seed=7)
    assert first == bootstrap_interval(values, alpha=0.05, resamples=2000, seed=7)
    assert -1 <= first[0] < sum(values) / len(values) < first[1] <= 1
    # Seed-averaged differences are fractional, so the registered seed is visible.
    fractional = [index / 37 for index in range(-12, 25)]
    assert bootstrap_interval(
        fractional, alpha=0.05, resamples=2000, seed=7
    ) != bootstrap_interval(fractional, alpha=0.05, resamples=2000, seed=8)
    assert bootstrap_interval([0, 0, 0, 0], alpha=0.05, resamples=1000, seed=0) == (0.0, 0.0)
    assert bootstrap_interval([1] * 9, alpha=0.05, resamples=1000, seed=0) == (1.0, 1.0)
    with pytest.raises(NrlyError):
        bootstrap_interval([], alpha=0.05, resamples=1000, seed=0)
    with pytest.raises(NrlyError):
        bootstrap_interval([float("nan")], alpha=0.05, resamples=1000, seed=0)


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    values = [1, 0, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, 1, 0, 1, 0]
    narrow = bootstrap_interval(values, alpha=0.2, resamples=4000, seed=3)
    wide = bootstrap_interval(values, alpha=0.01, resamples=4000, seed=3)
    assert wide[0] <= narrow[0] <= narrow[1] <= wide[1]
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


def _pair(
    reference: str, target: list[float], regression: list[float], rule: InferenceRule = RULE
) -> PairedInference:
    return paired_inference(reference, [target], [regression], rule)


def test_inference_flags_superiority_harm_and_non_inferiority() -> None:
    helped = _pair("baseline", [1] * 10 + [0] * 2, [0] * 12)
    assert helped.target_superior and not helped.target_harmed
    assert helped.regression_non_inferior and not helped.regression_inferior
    assert helped.target.improved_cases == 10 and helped.target.tied_cases == 2

    harmed = _pair("baseline", [-1] * 10 + [0] * 2, [0] * 12)
    assert harmed.target_harmed and not harmed.target_superior

    # A single flipped case is noise, in either direction: no claim is made.
    noise = _pair("baseline", [1] + [0] * 11, [-1] + [0] * 11)
    assert not noise.target_superior and not noise.target_harmed
    assert not noise.regression_inferior
    assert not noise.regression_non_inferior  # the interval reaches below the 2-point margin

    broken = _pair("baseline", [1] * 12, [-1] * 10 + [0] * 2)
    assert broken.regression_inferior and not broken.regression_non_inferior


def test_zero_margin_still_accepts_an_unchanged_regression_set() -> None:
    rule = InferenceRule(regression_margin=0, resamples=1000)
    assert _pair("baseline", [1] * 8, [0] * 8, rule).regression_non_inferior


def test_verdict_requires_beating_every_reference_and_sparing_regressions() -> None:
    strong, flat = [1] * 10 + [0] * 2, [0] * 12
    base = _pair("baseline", strong, flat)
    assert inferential_verdict([base, _pair("control:random", strong, flat)]) == (
        "observed_improvement",
        None,
    )
    # The candidate helps, but a registered alternative helps just as much.
    verdict, issue = inferential_verdict([base, _pair("comparator:thinking-off", flat, flat)])
    assert verdict == "inconclusive"
    assert issue == "candidate did not improve targets beyond every registered control"

    assert inferential_verdict([_pair("baseline", flat, flat)]) == ("no_change", None)
    assert inferential_verdict([_pair("baseline", [-1] * 10 + [0] * 2, flat)])[0] == "regressed"
    assert inferential_verdict([_pair("baseline", strong, [-1] * 10 + [0] * 2)])[0] == "regressed"

    verdict, issue = inferential_verdict([_pair("baseline", strong, [-1] + [0] * 11)])
    assert verdict == "inconclusive"
    assert issue == "regression non-inferiority against the original was not established"
    with pytest.raises(NrlyError):
        inferential_verdict([])


def test_one_flipped_case_among_hundreds_is_not_a_regression() -> None:
    # The count-based rule called this "regressed"; under inference it is a supported gain.
    target = [1] * 60 + [0] * 139 + [-1]
    regression = [0] * 199 + [-1]
    pair = _pair("baseline", target, regression)
    assert pair.target_superior and pair.regression_non_inferior
    assert inferential_verdict([pair]) == ("observed_improvement", None)
