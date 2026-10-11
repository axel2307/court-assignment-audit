"""Tests for independent assignments under a fully specified mechanism.

Probabilities must be known before inspecting the assignments. These functions
do not estimate probabilities from the tested counts, use resolved outcomes,
or calculate the probability of misconduct. The target alternative is fixed
as enrichment (``greater``); selecting a court after seeing its count requires
the exploratory family adjustment instead of the preselected-target reading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isclose, isfinite, log, sqrt
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from scipy.stats import binom, binomtest


@dataclass(frozen=True)
class TargetTestResult:
    """One-sided, exact binomial enrichment test and descriptive effects.

    ``log_p_value`` is the natural logarithm of the tail probability. If
    ``p_value_underflow`` is true, the numeric ``p_value`` has rounded to zero:
    display the finite log probability instead of interpreting that zero as
    mathematical impossibility. A genuinely impossible tail has log = -inf
    and its underflow flag is false.
    """

    target_index: int
    observed_count: int
    expected_count: float
    sample_size: int
    expected_probability: float
    observed_proportion: float
    lift_pp: float
    enrichment_ratio: float | None
    p_value: float
    alternative: str
    log_p_value: float
    p_value_underflow: bool


@dataclass(frozen=True)
class ExploratoryCourtResult(TargetTestResult):
    """A court's enrichment result within the complete Holm-adjusted family."""

    adjusted_p_value: float
    family_size: int
    adjusted_log_p_value: float
    adjusted_p_value_underflow: bool


@dataclass(frozen=True)
class GlobalTestResult:
    """Monte Carlo calibration of Pearson's multinomial discrepancy.

    ``p_value`` is (exceedances + 1) / (simulations + 1). The SE and exact
    binomial interval describe Monte Carlo uncertainty about the underlying
    null tail probability, not uncertainty about any legal or causal claim.
    A zero plug-in SE at either boundary does not imply zero uncertainty; use
    the interval, which remains informative there.
    """

    statistic: float
    p_value: float
    exceedances: int
    simulations: int
    mc_standard_error: float
    mc_tail_ci_low: float
    mc_tail_ci_high: float
    mc_confidence_level: float
    min_attainable_p_value: float
    seed: int
    total_variation: float
    expected_counts: tuple[float, ...]
    observed_counts: tuple[int, ...]
    support_violation: bool
    mc_standard_error_note: str


def _validated_inputs(
    counts: Sequence[int], probabilities: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate aligned, finite vectors without silently coercing strings."""
    observed_raw = np.asarray(counts, dtype=object)
    probability_raw = np.asarray(probabilities, dtype=object)
    if observed_raw.ndim != 1 or probability_raw.ndim != 1:
        raise ValueError("Counts and probabilities must be one-dimensional.")
    if observed_raw.size == 0 or observed_raw.size != probability_raw.size:
        raise ValueError("Counts and probabilities must have equal, nonzero length.")

    observed_values = []
    for value in observed_raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Counts must be nonnegative whole numbers.")
        # Check integer objects without a float conversion; very large Python
        # integers should produce the supported-range error, not OverflowError.
        if value < 0 or (
            not isinstance(value, Integral) and (not isfinite(value) or value != int(value))
        ):
            raise ValueError("Counts must be nonnegative whole numbers.")
        observed_values.append(int(value))
    sample_size = sum(observed_values)
    if sample_size <= 0:
        raise ValueError("At least one assignment is required.")
    if sample_size > np.iinfo(np.int64).max:
        raise ValueError("Total count exceeds the supported integer range.")

    probability_values = []
    for value in probability_raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("Probabilities must be finite numbers between 0 and 1.")
        if not 0 <= value <= 1 or not isfinite(value):
            raise ValueError("Probabilities must be finite numbers between 0 and 1.")
        value = float(value)
        probability_values.append(value)
    probability_sum = sum(probability_values)
    if not isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Probabilities must sum to 1 (absolute tolerance 1e-12).")

    observed = np.asarray(observed_values, dtype=np.int64)
    # Normalize accepted rounding noise so the exact and simulated tests use
    # the same probability vector. This never estimates p from observed data.
    expected_probabilities = np.asarray(probability_values, dtype=float)
    expected_probabilities /= expected_probabilities.sum()
    return observed, expected_probabilities, sample_size


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    lower_bound = 0 if allow_zero else 1
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < lower_bound
    ):
        adjective = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {adjective} integer.")
    return int(value)


def _log_binomial_upper_tail(count: int, sample_size: int, probability: float) -> float:
    """Recover an underflowed upper tail without allocating a tail array.

    Once count is above the mode, successive binomial PMFs decrease. Sum
    their ratios relative to the first term and bound the uncomputed remainder
    by a geometric series. The cap bounds work for extreme library inputs;
    supported application samples (n <= 100,000) converge well before it.
    Never silently return a truncated approximation if convergence fails.
    """
    if probability == 0:
        return 0.0 if count == 0 else float("-inf")
    if probability == 1 or count == 0:
        return 0.0
    log_first_term = float(binom.logpmf(count, sample_size, probability))
    if not isfinite(log_first_term):
        raise ArithmeticError("Cannot represent the binomial log tail for these inputs.")
    total = 1.0
    term = 1.0
    position = count
    odds = probability / (1 - probability)
    relative_tolerance = 8 * np.finfo(float).eps
    for _ in range(100_000):
        if position == sample_size:
            return log_first_term + log(total)
        ratio = ((sample_size - position) / (position + 1)) * odds
        if not 0 <= ratio < 1:
            raise ArithmeticError("Underflow recovery requires a decreasing upper tail.")
        remaining_bound = term * ratio / (1 - ratio)
        if remaining_bound <= relative_tolerance * total:
            return log_first_term + log(total)
        term *= ratio
        total += term
        position += 1
    raise ArithmeticError("Binomial log-tail recovery did not converge within the work limit.")


def exact_target_test(
    counts: Sequence[int],
    probabilities: Sequence[float],
    *,
    target_index: int = 0,
) -> TargetTestResult:
    """Return P(X >= observed target count), X ~ Binomial(n, p_target).

    Both target and enrichment direction must be chosen before drawing data
    for a preselected interpretation. A negative ``lift_pp`` denotes depletion
    and is not evidence for this one-sided enrichment alternative. Ratios with
    zero expected probability are undefined and returned as ``None``.
    """
    observed, expected_probabilities, sample_size = _validated_inputs(counts, probabilities)
    target_index = _positive_integer(target_index, "target_index", allow_zero=True)
    if target_index >= len(observed):
        raise ValueError("target_index is outside the count vector.")
    count = int(observed[target_index])
    probability = float(expected_probabilities[target_index])
    observed_proportion = count / sample_size
    p_value = float(binomtest(count, sample_size, probability, alternative="greater").pvalue)
    if p_value < np.finfo(float).tiny:
        # Also recover the log for subnormal values, whose rounded probability
        # can have large relative error even before it rounds fully to zero.
        log_p_value = _log_binomial_upper_tail(count, sample_size, probability)
        p_value = exp(log_p_value)
    else:
        log_p_value = log(p_value)
    return TargetTestResult(
        target_index=target_index,
        observed_count=count,
        expected_count=sample_size * probability,
        sample_size=sample_size,
        expected_probability=probability,
        observed_proportion=observed_proportion,
        lift_pp=100 * (observed_proportion - probability),
        enrichment_ratio=(observed_proportion / probability if probability else None),
        p_value=p_value,
        alternative="greater",
        log_p_value=log_p_value,
        p_value_underflow=p_value == 0 and isfinite(log_p_value),
    )


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Holm step-down adjusted p-values in the original order.

    The input must contain the entire prespecified family, including courts
    that appear unremarkable. Holm controls family-wise error without assuming
    independent court counts. An empty family returns an empty tuple.
    Input zero is treated as mathematical zero. For underflowed exact tails,
    use ``exploratory_court_tests`` and its log-space family calculation.
    """
    raw = np.asarray(p_values, dtype=object)
    if raw.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    values = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("p_values must be finite numbers between 0 and 1.")
        if not 0 <= value <= 1 or not isfinite(value):
            raise ValueError("p_values must be finite numbers between 0 and 1.")
        value = float(value)
        values.append(value)
    if not values:
        return ()
    p_array = np.asarray(values)
    order = np.argsort(p_array, kind="stable")
    adjusted_sorted = np.minimum(
        1.0,
        np.maximum.accumulate(p_array[order] * np.arange(len(values), 0, -1)),
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return tuple(float(value) for value in adjusted)


def exploratory_court_tests(
    counts: Sequence[int], probabilities: Sequence[float]
) -> tuple[ExploratoryCourtResult, ...]:
    """Test enrichment for every court, with one complete Holm family.

    These are exploratory results. Adjustment covers the listed courts for
    this one dataset and this fixed direction, not repeated runs, alternative
    directions, or arbitrary filters tried after observing results.
    """
    observed, expected_probabilities, _ = _validated_inputs(counts, probabilities)
    tests = tuple(
        exact_target_test(observed, expected_probabilities, target_index=index)
        for index in range(len(observed))
    )
    # Preserve ordering and adjustment for positive tails smaller than the
    # floating-point range. A genuine zero remains -inf throughout.
    log_values = np.asarray([test.log_p_value for test in tests])
    order = np.argsort(log_values, kind="stable")
    adjusted_sorted = np.minimum(
        0.0,
        np.maximum.accumulate(log_values[order] + np.log(np.arange(len(tests), 0, -1))),
    )
    adjusted_logs = np.empty_like(adjusted_sorted)
    adjusted_logs[order] = adjusted_sorted
    return tuple(
        ExploratoryCourtResult(
            **asdict(test),
            adjusted_p_value=exp(adjusted_log),
            family_size=len(tests),
            adjusted_log_p_value=float(adjusted_log),
            adjusted_p_value_underflow=exp(adjusted_log) == 0 and isfinite(adjusted_log),
        )
        for test, adjusted_log in zip(tests, adjusted_logs, strict=True)
    )


def global_multinomial_test(
    counts: Sequence[int],
    probabilities: Sequence[float],
    *,
    simulations: int = 20_000,
    seed: int = 1729,
) -> GlobalTestResult:
    """Calibrate Pearson's statistic using independent multinomial draws.

    This uses no asymptotic chi-square approximation and allows sparse counts.
    Each draw conditions on the observed total and the fully specified null
    probabilities. The local RNG is separate from any data-generation RNG.

    Ties count in the upper tail. A tolerance of 64 machine epsilons times
    max(1, observed statistic) protects mathematically tied floating-point
    values. The finite-simulation p-value is (b+1)/(B+1), never zero.

    Zero-probability cells with zero observations contribute zero. A positive
    count in such a cell is outside the null's support: statistic = infinity,
    ``support_violation`` is true, and the conservative Monte Carlo result is
    the simulation floor. The true null tail probability in that special case
    is zero; the returned Monte Carlo floor must not be presented as exact.
    """
    observed, expected_probabilities, sample_size = _validated_inputs(counts, probabilities)
    simulations = _positive_integer(simulations, "simulations")
    seed = _positive_integer(seed, "seed", allow_zero=True)
    expected = sample_size * expected_probabilities
    active = expected_probabilities > 0
    support_violation = bool(np.any(observed[~active] > 0))
    if support_violation:
        statistic = float("inf")
    else:
        statistic = float(np.sum((observed[active] - expected[active]) ** 2 / expected[active]))
    tolerance = 64 * np.finfo(float).eps * max(1.0, statistic)
    threshold = statistic - tolerance if isfinite(statistic) else statistic
    rng = np.random.default_rng(seed)
    exceedances = 0
    # Bound the temporary allocation without changing the experiment or
    # deriving any seed from the dataset or its observed discrepancy.
    batch_size = max(1, min(10_000, 1_000_000 // len(observed)))
    for start in range(0, simulations, batch_size):
        batch = rng.multinomial(
            sample_size,
            expected_probabilities,
            size=min(batch_size, simulations - start),
        )
        simulated_statistics = np.sum(
            (batch[:, active] - expected[active]) ** 2 / expected[active], axis=1
        )
        exceedances += int(np.count_nonzero(simulated_statistics >= threshold))
    tail_frequency = exceedances / simulations
    confidence_level = 0.95
    interval = binomtest(exceedances, simulations).proportion_ci(
        confidence_level=confidence_level, method="exact"
    )
    return GlobalTestResult(
        statistic=statistic,
        p_value=(exceedances + 1) / (simulations + 1),
        exceedances=exceedances,
        simulations=simulations,
        mc_standard_error=sqrt(tail_frequency * (1 - tail_frequency) / simulations),
        mc_tail_ci_low=float(interval.low),
        mc_tail_ci_high=float(interval.high),
        mc_confidence_level=confidence_level,
        min_attainable_p_value=1 / (simulations + 1),
        seed=seed,
        total_variation=float(
            0.5 * np.sum(np.abs(observed / sample_size - expected_probabilities))
        ),
        expected_counts=tuple(float(value) for value in expected),
        observed_counts=tuple(int(value) for value in observed),
        support_violation=support_violation,
        mc_standard_error_note=(
            "Approximate binomial plug-in Monte Carlo SE using b/B; a zero SE "
            "at b=0 or b=B does not imply zero uncertainty. The 95% exact "
            "interval describes the underlying null tail probability."
        ),
    )
