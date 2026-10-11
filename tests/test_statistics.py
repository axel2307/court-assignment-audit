"""Independent small-sample checks of tails, calibration, and validation."""

from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from math import comb, factorial, inf, isclose, isfinite, log, sqrt

import numpy as np
import pytest

from court_assignment_audit.statistics import (
    exact_target_test,
    exploratory_court_tests,
    global_multinomial_test,
    holm_adjust,
)


def manual_binomial_upper_tail(n, k, p):
    """Exact rational calculation, independent of scipy's binomial routines."""
    return sum(
        (Fraction(comb(n, count)) * p**count * (1 - p) ** (n - count) for count in range(k, n + 1)),
        Fraction(0),
    )


def rational_pearson(counts, probabilities):
    n = sum(counts)
    return sum(
        (
            (count - n * probability) ** 2 / (n * probability)
            for count, probability in zip(counts, probabilities, strict=True)
        ),
        Fraction(0),
    )


def enumerated_multinomial_tail(observed, probabilities):
    """Enumerate all count vectors; use exact arithmetic for ties and masses."""
    n = sum(observed)
    observed_statistic = rational_pearson(observed, probabilities)
    tail = Fraction(0)
    for prefix in product(range(n + 1), repeat=len(probabilities) - 1):
        remaining = n - sum(prefix)
        if remaining < 0:
            continue
        counts = (*prefix, remaining)
        if rational_pearson(counts, probabilities) < observed_statistic:
            continue
        probability_mass = Fraction(factorial(n))
        for count, probability in zip(counts, probabilities, strict=True):
            probability_mass *= probability**count / factorial(count)
        tail += probability_mass
    return observed_statistic, tail


@pytest.mark.parametrize(
    "n,k,p,known_tail",
    [
        (10, 8, Fraction(1, 2), Fraction(56, 1024)),
        (4, 3, Fraction(1, 4), Fraction(13, 256)),
        (8, 4, Fraction(1, 10), None),
        (8, 0, Fraction(1, 10), Fraction(1)),
        (10, 0, Fraction(0), Fraction(1)),
        (10, 1, Fraction(0), Fraction(0)),
        (10, 10, Fraction(1), Fraction(1)),
        # A greater-tail test does not test depletion, including at p=1.
        (10, 9, Fraction(1), Fraction(1)),
    ],
)
def test_exact_binomial_matches_independent_combinatorial_tail(n, k, p, known_tail):
    tail = manual_binomial_upper_tail(n, k, p)
    if known_tail is not None:
        assert tail == known_tail
    result = exact_target_test([k, n - k], [float(p), float(1 - p)])
    assert result.p_value == pytest.approx(float(tail), abs=1e-15)
    assert result.alternative == "greater"
    assert result.observed_count == k
    assert result.sample_size == n


def test_exact_target_index_and_effect_sizes_use_known_unbalanced_probability():
    result = exact_target_test([5, 4, 1], [0.6, 0.2, 0.2], target_index=1)
    assert result.target_index == 1
    assert result.expected_count == pytest.approx(2)
    assert result.observed_proportion == pytest.approx(0.4)
    assert result.lift_pp == pytest.approx(20)
    assert result.enrichment_ratio == pytest.approx(2)
    assert result.p_value == pytest.approx(float(manual_binomial_upper_tail(10, 4, Fraction(1, 5))))


def test_zero_expected_target_ratio_is_undefined():
    result = exact_target_test([0, 10], [0, 1])
    assert result.enrichment_ratio is None
    assert result.expected_count == 0
    assert result.lift_pp == 0


def decimal_binomial_log_upper_tail(n, k, probability):
    """Independent high-precision probability sum: exact integer coefficient.

    Unlike the production routine, this calculates the probability itself
    using Decimal and math.comb, without scipy, log-PMFs, or binary floats.
    """
    with localcontext() as context:
        context.prec = 70
        p = Decimal(probability)
        term = Decimal(comb(n, k)) * p**k * (1 - p) ** (n - k)
        total = term
        for count in range(k + 1, n + 1):
            term *= Decimal(n - count + 1) * p / (Decimal(count) * (1 - p))
            next_total = total + term
            if next_total == total:
                break
            total = next_total
        return float(total.ln())


def test_large_supported_sample_retains_finite_log_tail_when_float_underflows():
    result = exact_target_test([57_795, 42_205], [0.4, 0.6])
    independent_log_tail = decimal_binomial_log_upper_tail(100_000, 57_795, "0.4")
    assert result.p_value == 0
    assert result.p_value_underflow is True
    assert isfinite(result.log_p_value)
    assert result.log_p_value == pytest.approx(independent_log_tail, abs=1e-8, rel=0)


def test_log_tail_distinguishes_numerical_underflow_from_impossibility():
    impossible = exact_target_test([1, 9], [0, 1])
    certain = exact_target_test([0, 10], [0, 1])
    ordinary = exact_target_test([8, 2], [0.5, 0.5])
    assert impossible.p_value == 0
    assert impossible.log_p_value == -inf
    assert impossible.p_value_underflow is False
    assert certain.log_p_value == 0
    assert certain.p_value_underflow is False
    assert ordinary.log_p_value == pytest.approx(log(56 / 1024))
    assert ordinary.p_value_underflow is False


def test_positive_subnormal_tail_keeps_accurate_log_probability():
    result = exact_target_test([813, 0], [0.4, 0.6])
    # P(X=n) = p**n. Logging the rounded subnormal p instead would be off
    # by roughly 0.5 natural-log units despite that p being nonzero.
    assert result.p_value > 0
    assert result.p_value < np.finfo(float).tiny
    assert result.p_value_underflow is False
    assert result.log_p_value == pytest.approx(813 * log(0.4), abs=1e-11, rel=0)
    assert abs(result.log_p_value - log(result.p_value)) > 0.1


def test_holm_preserves_order_of_multiple_underflowed_tails_in_log_space():
    results = exploratory_court_tests([1_200, 800, 0], [0.01, 0.01, 0.98])
    first_log = decimal_binomial_log_upper_tail(2_000, 1_200, "0.01")
    second_log = decimal_binomial_log_upper_tail(2_000, 800, "0.01")
    assert results[0].log_p_value == pytest.approx(first_log, abs=1e-8, rel=0)
    assert results[1].log_p_value == pytest.approx(second_log, abs=1e-8, rel=0)
    assert results[0].adjusted_log_p_value == pytest.approx(first_log + log(3), abs=1e-8, rel=0)
    assert results[1].adjusted_log_p_value == pytest.approx(second_log + log(2), abs=1e-8, rel=0)
    assert results[0].adjusted_log_p_value < results[1].adjusted_log_p_value
    assert all(result.p_value_underflow for result in results[:2])
    assert all(result.adjusted_p_value_underflow for result in results[:2])
    assert results[2].adjusted_p_value == 1
    assert results[2].adjusted_log_p_value == 0
    assert results[2].adjusted_p_value_underflow is False


def test_holm_recovers_representable_adjusted_tail_from_underflowed_raw_tail():
    result = exploratory_court_tests([814, 0, 0], [0.4, 0.3, 0.3])[0]
    assert result.p_value == 0
    assert result.p_value_underflow is True
    assert result.adjusted_p_value > 0
    assert result.adjusted_p_value_underflow is False
    assert result.adjusted_log_p_value == pytest.approx(814 * log(0.4) + log(3))


def test_holm_mathematically_impossible_tail_remains_distinct_from_underflow():
    result = exploratory_court_tests([1, 9], [0, 1])[0]
    assert result.p_value == result.adjusted_p_value == 0
    assert result.log_p_value == result.adjusted_log_p_value == -inf
    assert result.p_value_underflow is result.adjusted_p_value_underflow is False


@pytest.mark.parametrize(
    "observed,probabilities,known_statistic,known_tail",
    [
        ([0, 2], [Fraction(1, 2), Fraction(1, 2)], Fraction(2), Fraction(1, 2)),
        ([2, 0], [Fraction(1, 4), Fraction(3, 4)], Fraction(6), Fraction(1, 16)),
        ([0, 2], [Fraction(1, 4), Fraction(3, 4)], Fraction(2, 3), Fraction(1)),
        ([1, 1], [Fraction(1, 4), Fraction(3, 4)], Fraction(2, 3), Fraction(1)),
        ([0, 1, 1], [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)], Fraction(2), Fraction(1, 2)),
        ([5, 1, 0], [Fraction(2, 5), Fraction(7, 20), Fraction(1, 4)], None, None),
    ],
)
def test_global_mc_matches_independent_exact_enumeration(
    observed, probabilities, known_statistic, known_tail
):
    statistic, exact_tail = enumerated_multinomial_tail(observed, probabilities)
    if known_statistic is not None:
        assert statistic == known_statistic
    if known_tail is not None:
        assert exact_tail == known_tail
    simulations = 50_000
    result = global_multinomial_test(
        observed, [float(p) for p in probabilities], simulations=simulations, seed=2718
    )
    assert result.statistic == pytest.approx(float(statistic))
    # Six theoretical Monte Carlo SEs plus one simulation's resolution. This
    # checks calibration without demanding an exact random frequency.
    q = float(exact_tail)
    tolerance = 6 * sqrt(q * (1 - q) / simulations) + 1 / (simulations + 1)
    assert abs(result.p_value - q) <= tolerance
    if q == 1:
        assert result.exceedances == simulations
        assert result.p_value == 1


def test_global_mc_matches_independent_scalar_simulation_loop():
    observed = [3, 2, 0]
    probabilities = [0.4, 0.3, 0.3]
    seed = 123456
    simulations = 257
    n = sum(observed)
    expected = [n * p for p in probabilities]
    observed_statistic = sum(
        (count - mean) ** 2 / mean for count, mean in zip(observed, expected, strict=True)
    )
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(simulations):
        draw = rng.multinomial(n, probabilities)
        simulated_statistic = sum(
            (int(count) - mean) ** 2 / mean for count, mean in zip(draw, expected, strict=True)
        )
        if simulated_statistic > observed_statistic or isclose(
            simulated_statistic, observed_statistic, abs_tol=1e-12
        ):
            exceedances += 1
    result = global_multinomial_test(observed, probabilities, simulations=simulations, seed=seed)
    assert result.exceedances == exceedances
    assert result.p_value == (exceedances + 1) / (simulations + 1)


def test_mc_finite_correction_and_uncertainty_with_controlled_draws(monkeypatch):
    # Against observed (0,2), exactly three of these nine draws have T >= 2.
    draws = np.array([[0, 2], [2, 0], [0, 2]] + [[1, 1]] * 6)

    class ControlledRng:
        def multinomial(self, n, probabilities, *, size):
            assert n == 2
            assert tuple(probabilities) == (0.5, 0.5)
            assert size == 9
            return draws

    monkeypatch.setattr(np.random, "default_rng", lambda seed: ControlledRng())
    result = global_multinomial_test([0, 2], [0.5, 0.5], simulations=9, seed=11)
    assert result.exceedances == 3
    assert result.p_value == 0.4
    assert result.min_attainable_p_value == 0.1
    assert result.mc_standard_error == pytest.approx(sqrt((1 / 3) * (2 / 3) / 9))
    assert result.mc_tail_ci_low < 1 / 3 < result.mc_tail_ci_high
    assert result.mc_confidence_level == 0.95
    assert "zero SE" in result.mc_standard_error_note


def test_mc_accumulates_full_and_partial_batches(monkeypatch):
    sizes = []

    class ControlledRng:
        def multinomial(self, n, probabilities, *, size):
            sizes.append(size)
            batch = np.tile([1, 1], (size, 1))
            # Two extreme observations in batch 1, one in batch 2, and none
            # in the partial batch: independent of the implementation's sum.
            extremes = [2, 1, 0][len(sizes) - 1]
            batch[:extremes] = [0, 2]
            return batch

    monkeypatch.setattr(np.random, "default_rng", lambda seed: ControlledRng())
    result = global_multinomial_test([0, 2], [0.5, 0.5], simulations=20_013, seed=11)
    assert sizes == [10_000, 10_000, 13]
    assert result.exceedances == 3
    assert result.p_value == 4 / 20_014


def test_zero_exceedances_report_floor_and_nonzero_interval():
    result = global_multinomial_test([20, 0], [0.01, 0.99], simulations=99, seed=11)
    assert result.exceedances == 0
    assert result.p_value == result.min_attainable_p_value == 0.01
    assert result.mc_standard_error == 0
    assert result.mc_tail_ci_low == 0
    assert result.mc_tail_ci_high == pytest.approx(1 - 0.025 ** (1 / 99))


def test_degenerate_distribution_and_zero_observed_cells():
    result = global_multinomial_test([12, 0, 0], [1, 0, 0], simulations=99, seed=0)
    assert result.statistic == 0
    assert result.p_value == 1
    assert result.exceedances == 99
    assert result.mc_standard_error == 0
    assert result.mc_tail_ci_low == pytest.approx(0.025 ** (1 / 99))
    assert result.mc_tail_ci_high == 1
    assert result.total_variation == 0
    assert result.support_violation is False


def test_null_support_violation_is_explicit_and_mc_result_is_conservative_floor():
    result = global_multinomial_test([11, 1], [1, 0], simulations=99, seed=0)
    assert result.statistic == inf
    assert result.support_violation is True
    assert result.exceedances == 0
    assert result.p_value == 0.01
    assert result.expected_counts == (12.0, 0.0)


def test_global_effect_size_and_metadata():
    result = global_multinomial_test([7, 3, 0], [0.4, 0.3, 0.3], simulations=11, seed=47)
    assert result.total_variation == pytest.approx(0.3)
    assert result.observed_counts == (7, 3, 0)
    assert result.expected_counts == pytest.approx((4, 3, 3))
    assert result.simulations == 11
    assert result.seed == 47


def test_seed_reproducibility_and_no_global_numpy_random_state_change():
    np.random.seed(19)
    state = np.random.get_state()
    first = global_multinomial_test([4, 1, 0], [0.4, 0.3, 0.3], simulations=101, seed=4)
    second = global_multinomial_test([4, 1, 0], [0.4, 0.3, 0.3], simulations=101, seed=4)
    assert first == second
    state_after = np.random.get_state()
    assert state[0] == state_after[0]
    np.testing.assert_array_equal(state[1], state_after[1])
    assert state[2:] == state_after[2:]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ([0.01, 0.04, 0.03, 1], [0.04, 0.09, 0.09, 1]),
        ([0.01, 0.01, 0.4], [0.03, 0.03, 0.4]),
        ([1, 0, 0.05], [1, 0, 0.1]),
        ([0.8, 0.9], [1, 1]),
        ([0.125], [0.125]),
        ([], []),
    ],
)
def test_holm_adjustment_original_order_ties_and_capping(raw, expected):
    assert holm_adjust(raw) == pytest.approx(expected)


def test_exploration_adjusts_full_court_family_and_keeps_original_indices():
    results = exploratory_court_tests([8, 1, 1, 0], [0.4, 0.3, 0.2, 0.1])
    raw = [
        float(manual_binomial_upper_tail(10, k, p))
        for k, p in zip(
            [8, 1, 1, 0],
            [Fraction(2, 5), Fraction(3, 10), Fraction(1, 5), Fraction(1, 10)],
            strict=True,
        )
    ]
    assert [result.target_index for result in results] == [0, 1, 2, 3]
    assert [result.p_value for result in results] == pytest.approx(raw)
    assert all(result.family_size == 4 for result in results)
    assert results[0].adjusted_p_value == pytest.approx(4 * raw[0])
    assert [result.adjusted_p_value for result in results[1:]] == [1, 1, 1]


@pytest.mark.parametrize(
    "counts,probabilities",
    [
        ([], []),
        ([0, 0], [0.5, 0.5]),
        ([1], [0.5, 0.5]),
        ([[1, 2]], [0.5, 0.5]),
        ([1, 2], [[0.5, 0.5]]),
        ([-1, 2], [0.5, 0.5]),
        ([0.5, 2], [0.5, 0.5]),
        ([True, 2], [0.5, 0.5]),
        (["1", 2], [0.5, 0.5]),
        ([float("nan"), 2], [0.5, 0.5]),
        ([float("inf"), 2], [0.5, 0.5]),
        ([2**63, 1], [0.5, 0.5]),
        ([10**400, 1], [0.5, 0.5]),
        ([1, 2], [-0.1, 1.1]),
        ([1, 2], [10**400, 0.5]),
        ([1, 2], [float("nan"), 0.5]),
        ([1, 2], [float("inf"), 0.5]),
        ([1, 2], ["0.5", 0.5]),
        ([1, 2], [True, False]),
        ([1, 2], [0.2, 0.3]),
        ([1, 2], [0, 0]),
        ([1, 2], [0.500000001, 0.5]),
    ],
)
def test_invalid_count_probability_inputs_rejected_consistently(counts, probabilities):
    for function in (exact_target_test, global_multinomial_test, exploratory_court_tests):
        with pytest.raises(ValueError):
            function(counts, probabilities)


@pytest.mark.parametrize("target_index", [-1, 2, 0.5, True, "0"])
def test_invalid_target_indices(target_index):
    with pytest.raises(ValueError):
        exact_target_test([1, 2], [0.5, 0.5], target_index=target_index)


@pytest.mark.parametrize("simulations", [0, -1, 2.5, True, "10"])
def test_invalid_simulation_counts(simulations):
    with pytest.raises(ValueError):
        global_multinomial_test([1, 2], [0.5, 0.5], simulations=simulations)


@pytest.mark.parametrize("seed", [-1, 1.5, True, "42"])
def test_invalid_simulation_seeds(seed):
    with pytest.raises(ValueError):
        global_multinomial_test([1, 2], [0.5, 0.5], seed=seed)


@pytest.mark.parametrize(
    "p_values", [[-0.1], [1.1], [10**400], [float("nan")], [float("inf")], [True], ["0.1"], [[0.1]]]
)
def test_invalid_holm_inputs(p_values):
    with pytest.raises(ValueError):
        holm_adjust(p_values)


def test_accepted_rounding_noise_and_numpy_integer_inputs():
    result = exact_target_test(
        np.array([1, 2], dtype=np.int64), [0.5, 0.5 + 1e-14], target_index=np.int64(0)
    )
    assert result.expected_probability == pytest.approx(0.5)
    assert result.p_value == pytest.approx(0.875)
