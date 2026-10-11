"""Exact PMF, inclusive-tail, cropping, and display-label regressions."""

from dataclasses import replace
from fractions import Fraction
from math import ceil, comb, floor, fsum, isfinite, log

import numpy as np
import pytest
from scipy.stats import binom

from court_assignment_audit.binomial_view import build_binomial_view
from court_assignment_audit.statistics import exact_target_test


def result_for(n, k, p=0.4):
    return exact_target_test([k, n - k], [p, 1 - p])


@pytest.mark.parametrize(
    "n,k,p",
    [
        (1, 0, Fraction(2, 5)),
        (1, 1, Fraction(2, 5)),
        (10, 0, Fraction(1, 5)),
        (10, 4, Fraction(2, 5)),
        (10, 10, Fraction(2, 5)),
        (7, 3, Fraction(1, 3)),
    ],
)
def test_pmf_and_inclusive_shading_match_independent_combinatorial_probabilities(n, k, p):
    view = build_binomial_view(result_for(n, k, float(p)))
    probabilities = [Fraction(comb(n, j)) * p**j * (1 - p) ** (n - j) for j in range(n + 1)]
    for point in view.points:
        expected = float(probabilities[point.count])
        assert point.pmf == pytest.approx(expected, rel=1e-12, abs=1e-16)
        assert point.log_pmf == pytest.approx(log(expected), abs=1e-12)
        assert point.in_tail is (point.count >= k)
    assert next(point for point in view.points if point.count == k).in_tail
    exact_tail = float(sum(probabilities[k:], Fraction(0)))
    assert view.full_tail_probability == pytest.approx(exact_tail, rel=1e-12)
    assert view.shown_tail_mass + view.omitted_right_mass == pytest.approx(exact_tail, rel=1e-12)
    assert fsum(point.pmf for point in view.points) + view.omitted_mass == pytest.approx(1)


@pytest.mark.parametrize(
    "n,k,p",
    [
        (1, 0, 0.4),
        (1, 1, 0.4),
        (5000, 0, 0.4),
        (5000, 2000, 0.4),
        (5000, 2900, 0.4),
        (5000, 3600, 0.4),
        (5000, 5000, 0.4),
        (100, 1, 0.001),
        (100, 100, 0.999),
    ],
)
def test_range_contains_observed_expected_and_preserves_total_and_relative_tail_mass(n, k, p):
    result = result_for(n, k, p)
    view = build_binomial_view(result)
    assert 0 <= view.lower_count <= min(k, floor(n * p))
    assert max(k, ceil(n * p)) <= view.upper_count <= n
    assert [point.count for point in view.points] == list(
        range(view.lower_count, view.upper_count + 1)
    )
    assert view.omitted_mass <= 1e-6
    assert view.omitted_tail_fraction <= 1e-6
    assert view.omitted_left_mass == binom.cdf(view.lower_count - 1, n, p)
    assert view.omitted_right_mass == binom.sf(view.upper_count, n, p)
    assert view.full_tail_probability == result.p_value
    assert view.full_log_tail_probability == result.log_p_value
    assert np.logaddexp(view.log_shown_tail_mass, view.log_omitted_right_mass) == pytest.approx(
        result.log_p_value, abs=1e-8, rel=0
    )
    counts = np.array([point.count for point in view.points])
    np.testing.assert_array_equal([point.pmf for point in view.points], binom.pmf(counts, n, p))
    np.testing.assert_array_equal(
        [point.log_pmf for point in view.points], binom.logpmf(counts, n, p)
    )


def test_tiny_tail_cropping_retains_its_relative_mass_and_extends_past_observed_count():
    view = build_binomial_view(result_for(5000, 3600))
    assert view.tail_probability_underflow
    assert view.full_tail_probability == 0
    assert view.upper_count > view.observed_count
    assert view.tail_clipped
    assert isfinite(view.log_omitted_right_mass)
    assert isfinite(view.log_shown_tail_mass)
    assert view.omitted_tail_fraction <= 1e-6
    assert view.probability_label != "0%"
    assert "ln p =" in view.probability_label
    assert "below floating-point range" in view.probability_label
    assert view.frequency_label is None


def test_all_assignments_to_target_has_exact_endpoint_log_probability():
    view = build_binomial_view(result_for(5000, 5000))
    assert view.full_log_tail_probability == pytest.approx(5000 * log(0.4))
    assert view.log_shown_tail_mass == pytest.approx(5000 * log(0.4))
    assert view.upper_count == 5000
    assert view.omitted_right_mass == 0
    assert view.log_omitted_right_mass == -np.inf
    assert view.omitted_tail_fraction == 0
    assert view.tail_clipped is False
    assert sum(point.in_tail for point in view.points) == 1
    assert "inf" not in view.probability_label


@pytest.mark.parametrize(
    "n,k,p", [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1), (5000, 5000, 0), (5000, 0, 1)]
)
def test_degenerate_models_and_impossible_tails(n, k, p):
    view = build_binomial_view(result_for(n, k, p))
    assert fsum(point.pmf for point in view.points) == 1
    assert view.omitted_mass == 0
    assert view.omitted_tail_fraction == 0
    assert view.frequency_label is None
    assert view.tail_probability_underflow is False
    if p == 0 and k > 0:
        assert view.full_tail_probability == view.shown_tail_mass == 0
        assert view.full_log_tail_probability == view.log_shown_tail_mass == -np.inf
        assert "impossible under the specified model" in view.probability_label
    else:
        assert view.full_tail_probability == view.shown_tail_mass == 1
        assert view.probability_label == "100%"


def test_stricter_cropping_budget_never_shrinks_the_displayed_domain():
    result = result_for(5000, 2250)
    default = build_binomial_view(result)
    tighter = build_binomial_view(result, max_omitted_mass=1e-9)
    assert tighter.lower_count <= default.lower_count
    assert tighter.upper_count >= default.upper_count
    assert tighter.omitted_mass <= 1e-9
    assert tighter.omitted_tail_fraction <= 1e-9
    assert "not rescaled" in tighter.range_note


@pytest.mark.parametrize(
    "p,frequency",
    [
        (0.4, "about 40 in 100 repeated allocations of this size"),
        (0.5, "about 50 in 100 repeated allocations of this size"),
        (0.1, "about 10 in 100 repeated allocations of this size"),
        (0.01, "about 1 in 100 repeated allocations of this size"),
        (0.009, "about 1 in 111 repeated allocations of this size"),
        (0.005, "about 1 in 200 repeated allocations of this size"),
        (0.500001, None),
        (1e-6, "about 1 in 1,000,000 repeated allocations of this size"),
        (0.999999e-6, None),
        (1, None),
    ],
)
def test_reciprocal_labels_are_bounded_and_refer_to_whole_allocations(p, frequency):
    view = build_binomial_view(result_for(1, 1, p))
    assert view.frequency_label == frequency


def test_probability_labels_have_explicit_units_without_false_zero_or_certainty():
    assert build_binomial_view(result_for(1, 1, 0.4)).probability_label == "40%"
    assert build_binomial_view(result_for(1, 1, 0.999999)).probability_label == "greater than 99.9%"
    assert (
        build_binomial_view(result_for(1, 1, 2.01e-25)).probability_label == "2.01e-25 probability"
    )
    assert (
        build_binomial_view(result_for(1, 1, 9.9999e-20)).probability_label == "1e-19 probability"
    )
    subnormal = build_binomial_view(result_for(813, 813))
    assert "probability" in subnormal.probability_label
    assert subnormal.probability_label != "0%"


def test_probability_rounded_to_one_is_not_labeled_mathematically_certain():
    result = result_for(100, 1, 0.5)
    assert result.p_value == 1  # True tail is 1 - 2**(-100), not exactly 1.
    view = build_binomial_view(result)
    assert view.probability_label == "greater than 99.9%"
    assert view.frequency_label is None


@pytest.mark.parametrize("budget", [0, -1e-6, 1e-5, True, "1e-6", float("nan"), float("inf")])
def test_invalid_omission_budgets(budget):
    with pytest.raises(ValueError):
        build_binomial_view(result_for(10, 4), max_omitted_mass=budget)


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_size": 0},
        {"sample_size": 5001},
        {"sample_size": True},
        {"sample_size": 1.5},
        {"observed_count": -1},
        {"observed_count": 11},
        {"observed_count": True},
        {"expected_probability": -0.1},
        {"expected_probability": float("nan")},
        {"expected_probability": True},
        {"expected_count": 3},
        {"log_p_value": float("nan")},
        {"log_p_value": 1},
        {"log_p_value": -np.inf},
        {"p_value": -0.1},
        {"p_value_underflow": True},
    ],
)
def test_invalid_or_inconsistent_test_results(changes):
    with pytest.raises(ValueError):
        build_binomial_view(replace(result_for(10, 4), **changes))


def test_requires_an_exact_target_result():
    with pytest.raises(ValueError):
        build_binomial_view(None)
