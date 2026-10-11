"""UI readings respect test direction, simulation uncertainty, and sample size."""

from dataclasses import replace

import numpy as np
import pytest

from court_assignment_audit.presentation import (
    global_reading,
    reference_allocations,
    target_reading,
)
from court_assignment_audit.reporting import MC_SEED
from court_assignment_audit.statistics import exact_target_test, global_multinomial_test


@pytest.fixture
def global_result():
    return global_multinomial_test([4, 3, 2, 1], [0.4, 0.3, 0.2, 0.1], simulations=99)


@pytest.mark.parametrize(
    "p_value,low,high,status",
    [
        (0.02, 0.01, 0.04, "unusual"),
        (0.06, 0.051, 0.08, "no_clear_departure"),
        (0.02, 0.01, 0.05, "borderline"),
        (0.06, 0.05, 0.08, "borderline"),
        (0.049, 0.03, 0.07, "borderline"),
        (0.05, 0.04, 0.06, "borderline"),
        (0.05, 0.01, 0.04, "borderline"),
        (0.05, 0.05, 0.05, "borderline"),
    ],
)
def test_global_reading_respects_mc_interval_and_inclusive_boundary(
    global_result, p_value, low, high, status
):
    result = replace(global_result, p_value=p_value, mc_tail_ci_low=low, mc_tail_ci_high=high)
    assert global_reading(result, n_cases=10).status == status


def test_global_reference_is_configurable(global_result):
    result = replace(global_result, p_value=0.02, mc_tail_ci_low=0.015, mc_tail_ci_high=0.03)
    assert global_reading(result, 10, reference=0.05).status == "unusual"
    reading = global_reading(result, 10, reference=0.01)
    assert reading.status == "no_clear_departure"
    assert "does not establish that the mechanism was followed" in reading.explanation


@pytest.mark.parametrize("reference,label", [(0.05, "5%"), (0.01, "1%")])
def test_borderline_headline_names_the_configured_p_value_reference(
    global_result, reference, label
):
    result = replace(
        global_result,
        p_value=reference,
        mc_tail_ci_low=reference / 2,
        mc_tail_ci_high=reference * 2,
    )
    reading = global_reading(result, 10, reference=reference)
    assert reading.status == "borderline"
    assert reading.headline == (
        "The simulations do not clearly resolve whether the p-value is below the "
        f"{label} reference threshold"
    )


@pytest.mark.parametrize(
    "p_value,low,high", [(0.02, 0.01, 0.04), (0.1, 0.08, 0.12), (0.05, 0.03, 0.07)]
)
def test_main_global_explanations_leave_technical_details_to_expanded_view(
    global_result, p_value, low, high
):
    result = replace(global_result, p_value=p_value, mc_tail_ci_low=low, mc_tail_ci_high=high)
    explanation = global_reading(result, 10).explanation.lower()
    for technical_term in ("p-value", "interval", "confidence", "holm", "monte carlo"):
        assert technical_term not in explanation


def test_single_assignment_always_gets_limited_reading_and_receiving_court_chance():
    result = global_multinomial_test([0, 0, 0, 1], [0.4, 0.3, 0.2, 0.1], simulations=99)
    result = replace(result, p_value=0.001, mc_tail_ci_low=0, mc_tail_ci_high=0.01)
    reading = global_reading(result, 1)
    assert reading.status == "limited"
    assert reading.headline == "One assignment cannot establish a stable allocation pattern."
    assert reading.explanation.startswith("Model probability of assignment to this court: 10%.")


def test_single_assignment_court_chance_is_distinct_from_global_tail():
    result = global_multinomial_test([1, 0, 0, 0], [0.25] * 4, simulations=99)
    assert result.p_value == 1
    reading = global_reading(result, 1)
    assert reading.status == "limited"
    assert reading.explanation.startswith("Model probability of assignment to this court: 25%.")


def test_single_assignment_with_zero_probability_explains_model_contradiction():
    result = global_multinomial_test([1, 0], [0, 1], simulations=99)
    reading = global_reading(result, 1)
    assert reading.status == "limited"
    assert reading.explanation.startswith("Model probability of assignment to this court: 0%.")
    assert "impossible under the specified model" in reading.explanation


@pytest.mark.parametrize("n_cases", [10, 5_000])
def test_known_model_support_violation_is_unusual_even_with_few_simulations(n_cases):
    result = global_multinomial_test([1, n_cases - 1], [0, 1], simulations=9)
    assert result.support_violation
    assert result.p_value == 0.1
    assert result.p_value == result.min_attainable_p_value
    reading = global_reading(result, n_cases)
    assert reading.status == "unusual"
    assert reading.headline == "The observed allocation is impossible under the specified model"
    assert "specified probability of zero" in reading.explanation


def test_zero_exceedances_do_not_override_mc_interval_precision():
    weak_precision = global_multinomial_test([20, 0], [0.01, 0.99], simulations=9, seed=7)
    assert weak_precision.exceedances == 0
    assert weak_precision.p_value > 0
    assert global_reading(weak_precision, 20).status == "borderline"
    enough_precision = global_multinomial_test([20, 0], [0.01, 0.99], simulations=99, seed=7)
    assert enough_precision.exceedances == 0
    assert enough_precision.p_value > 0
    assert global_reading(enough_precision, 20).status == "unusual"


def test_target_below_expectation_always_explains_enrichment_direction():
    result = exact_target_test([1, 9], [0.4, 0.6])
    reading = target_reading(result)
    assert reading.status == "no_clear_excess"
    assert reading.headline == "This check looks for extra assignments, not a shortfall"
    assert "does not assess whether assignments are unusually low" in reading.explanation


def test_target_equal_to_expectation_has_distinct_mean_matching_explanation():
    reading = target_reading(exact_target_test([4, 6], [0.4, 0.6]))
    assert reading.status == "no_clear_excess"
    assert reading.headline == "The preselected court matches the expected count"
    assert "average across repeated random allocations" in reading.explanation
    assert "does not establish that the assignment mechanism was followed" in reading.explanation
    assert "shortfall" not in reading.headline


def test_target_positive_lift_needs_tail_strictly_below_five_percent():
    result = exact_target_test([8, 2], [0.4, 0.6])
    assert target_reading(result).status == "unusual_excess"
    assert target_reading(replace(result, p_value=0.05)).status == "no_clear_excess"
    assert target_reading(replace(result, lift_pp=0)).status == "no_clear_excess"
    assert target_reading(exact_target_test([5, 5], [0.4, 0.6])).status == "no_clear_excess"


def test_target_equality_reading_tolerates_only_expected_count_roundoff():
    result = exact_target_test([2, 1, 1, 1], [0.4, 0.3, 0.2, 0.1])
    assert target_reading(result).headline == "The preselected court matches the expected count"
    assert target_reading(replace(result, expected_count=2.01)).headline == (
        "This check looks for extra assignments, not a shortfall"
    )


def test_target_underflowed_positive_tail_still_has_directional_reading():
    result = exact_target_test([57_795, 42_205], [0.4, 0.6])
    assert result.p_value_underflow
    assert target_reading(result).status == "unusual_excess"


@pytest.mark.parametrize("counts", [[8, 2], [5, 5]])
def test_main_target_explanations_leave_technical_details_to_expanded_view(counts):
    explanation = target_reading(exact_target_test(counts, [0.4, 0.6])).explanation.lower()
    for technical_term in ("p-value", "5%", "upper-tail"):
        assert technical_term not in explanation


def test_unusual_single_target_event_is_not_presented_as_a_stable_pattern():
    result = exact_target_test([1, 0], [0.01, 0.99])
    reading = target_reading(result)
    assert reading.status == "unusual_excess"
    assert (
        "One assignment alone does not establish a stable allocation pattern" in reading.explanation
    )


def test_target_impossibility_and_underflow_have_distinct_explanations():
    impossible = target_reading(exact_target_test([1, 9], [0, 1]))
    underflowed = target_reading(exact_target_test([57_795, 42_205], [0.4, 0.6]))
    assert "impossible" in impossible.explanation
    assert "impossible" not in underflowed.explanation


@pytest.mark.parametrize("n_cases", [1, 5_000])
def test_reference_allocations_are_integer_joint_draws_of_the_same_sample_size(n_cases):
    draws = reference_allocations(n_cases, [0.4, 0.3, 0.2, 0.1])
    assert draws.shape == (12, 4)
    assert np.issubdtype(draws.dtype, np.integer)
    assert np.all(draws >= 0)
    np.testing.assert_array_equal(draws.sum(axis=1), np.full(12, n_cases))


def test_references_equal_the_first_unselected_null_draws_and_keep_global_rng_unchanged():
    probabilities = [0.65, 0.2, 0.1, 0.05]
    np.random.seed(19)
    state = np.random.get_state()
    actual = reference_allocations(5_000, probabilities)
    expected = np.random.default_rng(MC_SEED).multinomial(5_000, probabilities, size=12)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, reference_allocations(5_000, probabilities))
    state_after = np.random.get_state()
    assert state[0] == state_after[0]
    np.testing.assert_array_equal(state[1], state_after[1])
    assert state[2:] == state_after[2:]


def test_references_preserve_stream_prefix_and_accept_custom_seed_and_count():
    short = reference_allocations(20, [0.4, 0.6], seed=3, examples=3)
    long = reference_allocations(20, [0.4, 0.6], seed=3, examples=15)
    np.testing.assert_array_equal(short, long[:3])
    assert not np.array_equal(short, reference_allocations(20, [0.4, 0.6], seed=4, examples=3))


def test_references_allow_zero_probabilities_and_degenerate_mechanism():
    draws = reference_allocations(5_000, [0, 1, 0], seed=0)
    np.testing.assert_array_equal(draws, np.tile([0, 5_000, 0], (12, 1)))


@pytest.mark.parametrize("n_cases", [0, -1, True, 1.5, "1", 2**63])
def test_invalid_reference_sample_sizes(n_cases):
    with pytest.raises(ValueError):
        reference_allocations(n_cases, [0.4, 0.6])


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        [[0.4, 0.6]],
        [0.4, 0.4],
        [0, 0],
        [-0.1, 1.1],
        [True, False],
        ["0.4", 0.6],
        [float("nan"), 0.6],
        [float("inf"), 0.6],
        [10**400, 0],
    ],
)
def test_invalid_reference_probabilities(probabilities):
    with pytest.raises(ValueError):
        reference_allocations(20, probabilities)


@pytest.mark.parametrize("seed", [-1, True, 0.5, "7"])
def test_invalid_reference_seeds(seed):
    with pytest.raises(ValueError):
        reference_allocations(20, [0.4, 0.6], seed=seed)


@pytest.mark.parametrize("examples", [0, -1, True, 0.5, "12"])
def test_invalid_reference_example_counts(examples):
    with pytest.raises(ValueError):
        reference_allocations(20, [0.4, 0.6], examples=examples)


@pytest.mark.parametrize("reference", [0, 1, -0.1, True, "0.05", float("nan")])
def test_invalid_global_reference(global_result, reference):
    with pytest.raises(ValueError):
        global_reading(global_result, 10, reference=reference)


@pytest.mark.parametrize("n_cases", [0, -1, True, 1.5, 1, 9])
def test_global_reading_requires_a_valid_matching_sample_size(global_result, n_cases):
    with pytest.raises(ValueError):
        global_reading(global_result, n_cases)
