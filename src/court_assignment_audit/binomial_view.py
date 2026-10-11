"""Exact binomial PMF data for the preselected-court teaching graphic.

The displayed domain may omit negligible probability, but bar heights always
remain the original discrete probabilities. Full-tail labels come from the
existing exact test, never from a renormalized or truncated visual.
"""

from dataclasses import dataclass
from math import ceil, exp, floor, fsum, isclose, isfinite, log
from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

from court_assignment_audit.statistics import TargetTestResult

MAX_VIEW_CASES = 5_000
MAX_OMITTED_MASS = 1e-6


@dataclass(frozen=True)
class BinomialPoint:
    count: int
    pmf: float
    log_pmf: float
    in_tail: bool


@dataclass(frozen=True)
class BinomialView:
    n_cases: int
    probability: float
    observed_count: int
    expected_count: float
    lower_count: int
    upper_count: int
    points: tuple[BinomialPoint, ...]
    omitted_left_mass: float
    omitted_right_mass: float
    omitted_mass: float
    log_omitted_left_mass: float
    log_omitted_right_mass: float
    log_omitted_mass: float
    omitted_tail_fraction: float
    shown_tail_mass: float
    log_shown_tail_mass: float
    full_tail_probability: float
    full_log_tail_probability: float
    tail_probability_underflow: bool
    tail_clipped: bool
    max_omitted_mass: float
    probability_label: str
    frequency_label: str | None
    range_note: str


def _integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
    return int(value)


def _labels(log_probability: float, underflow: bool, *, certain: bool) -> tuple[str, str | None]:
    """Percent for ordinary tails; scientific probability for very small tails.

    Frequencies refer to whole repeated allocations with this sample size.
    Probabilities from 0.01 through 0.5 use a denominator of 100; reciprocals
    are reserved for probabilities from 1e-6 up to, but excluding, 0.01.
    """
    if log_probability == float("-inf"):
        return "0% (impossible under the specified model)", None
    if log_probability == 0:
        # A valid positive complement can round away, making a non-certain
        # probability appear to be 1. Reserve 100% for known certain events.
        return ("100%" if certain else "greater than 99.9%"), None
    if log_probability >= log(1e-4):
        probability = exp(log_probability)
        if probability > 0.999:
            label = "greater than 99.9%"
        else:
            label = f"{100 * probability:.3g}%"
    else:
        exponent = floor(log_probability / log(10))
        mantissa = float(f"{exp(log_probability - exponent * log(10)):.3g}")
        if mantissa >= 10:
            mantissa /= 10
            exponent += 1
        label = f"{mantissa:.3g}e{exponent:+d} probability"
        if underflow:
            label += f" (ln p = {log_probability:.6g}; below floating-point range)"
    frequency = None
    if log(0.01) <= log_probability <= log(0.5):
        frequency = (
            f"about {100 * exp(log_probability):.3g} in 100 repeated allocations of this size"
        )
    elif log(1e-6) <= log_probability < log(0.01):
        denominator = exp(-log_probability)
        frequency = f"about 1 in {denominator:,.0f} repeated allocations of this size"
    return label, frequency


def build_binomial_view(
    result: TargetTestResult, *, max_omitted_mass: float = MAX_OMITTED_MASS
) -> BinomialView:
    """Build an exact, discrete visual for 1 through 5,000 assignments.

    The contiguous domain includes the observed count, both integers around
    the expected count, and central probability mass. Total omitted mass is
    no more than ``max_omitted_mass`` (at most 1e-6). For a positive full tail,
    the omitted right tail is also no more than that fraction of the *full
    tail*, even when the tail is below floating-point probability range.

    A full support vector of at most 5,001 values bounds memory and work.
    Log-PMF cumulative sums preserve tiny probabilities that scipy.logsf can
    itself underflow. Scalar omitted masses use stable CDF/SF calls. No PMF
    normalization, normal approximation, density, histogram, or test is added.
    """
    if not isinstance(result, TargetTestResult):
        raise ValueError("result must be a TargetTestResult.")
    n_cases = _integer(result.sample_size, "sample_size", 1, MAX_VIEW_CASES)
    observed = _integer(result.observed_count, "observed_count", 0, n_cases)
    probability = result.expected_probability
    if (
        isinstance(probability, (bool, np.bool_))
        or not isinstance(probability, Real)
        or not 0 <= probability <= 1
        or not isfinite(probability)
    ):
        raise ValueError("expected_probability must be finite and between 0 and 1.")
    probability = float(probability)
    expected = n_cases * probability
    if not isfinite(result.expected_count) or not isclose(
        result.expected_count, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("expected_count must equal sample_size times expected_probability.")
    if (
        isinstance(max_omitted_mass, (bool, np.bool_))
        or not isinstance(max_omitted_mass, Real)
        or not 0 < max_omitted_mass <= MAX_OMITTED_MASS
        or not isfinite(max_omitted_mass)
    ):
        raise ValueError("max_omitted_mass must be positive and at most 1e-6.")
    max_omitted_mass = float(max_omitted_mass)
    log_full_tail = result.log_p_value
    impossible_tail = probability == 0 and observed > 0
    if (
        not isinstance(log_full_tail, Real)
        or isinstance(log_full_tail, (bool, np.bool_))
        or log_full_tail > 0
        or (not isfinite(log_full_tail) and not (impossible_tail and log_full_tail == -np.inf))
    ):
        raise ValueError("log_p_value must describe a valid full binomial upper tail.")
    if (
        not isinstance(result.p_value, Real)
        or isinstance(result.p_value, (bool, np.bool_))
        or not 0 <= result.p_value <= 1
        or not isfinite(result.p_value)
        or result.p_value_underflow != (result.p_value == 0 and isfinite(log_full_tail))
        or (impossible_tail and (result.p_value != 0 or log_full_tail != -np.inf))
    ):
        raise ValueError("p_value and its underflow flag must be consistent.")
    log_full_tail = float(log_full_tail)

    counts = np.arange(n_cases + 1)
    pmfs = binom.pmf(counts, n_cases, probability)
    log_pmfs = binom.logpmf(counts, n_cases, probability)
    log_left = np.concatenate(([-np.inf], np.logaddexp.accumulate(log_pmfs)[:-1]))
    log_right = np.concatenate((np.logaddexp.accumulate(log_pmfs[::-1])[::-1][1:], [-np.inf]))
    # Half-budgets guarantee both total omission and the relative tail bound;
    # they also leave margin for roundoff in PMF and tail implementations.
    log_half_budget = log(max_omitted_mass) - log(2)
    central_lower = int(np.flatnonzero(log_left <= log_half_budget)[-1])
    central_upper = int(np.flatnonzero(log_right <= log_half_budget)[0])
    tail_upper = int(np.flatnonzero(log_right <= log_full_tail + log_half_budget)[0])
    lower = min(central_lower, observed, floor(expected))
    upper = max(central_upper, tail_upper, observed, ceil(expected))
    # Check stable scalar CDF/SF values too, without ever subtracting from 1.
    while lower > 0 and binom.cdf(lower - 1, n_cases, probability) > max_omitted_mass / 2:
        lower -= 1
    while upper < n_cases and binom.sf(upper, n_cases, probability) > max_omitted_mass / 2:
        upper += 1

    points = tuple(
        BinomialPoint(int(count), float(pmfs[count]), float(log_pmfs[count]), count >= observed)
        for count in range(lower, upper + 1)
    )
    omitted_left = float(binom.cdf(lower - 1, n_cases, probability))
    omitted_right = float(binom.sf(upper, n_cases, probability))
    log_omitted_left = float(log_left[lower])
    log_omitted_right = float(log_right[upper])
    omitted_tail_fraction = exp(log_omitted_right - log_full_tail) if not impossible_tail else 0.0
    shown_tail = fsum(point.pmf for point in points if point.in_tail)
    log_shown_tail = float(logsumexp(log_pmfs[observed : upper + 1]))
    label, frequency = _labels(
        log_full_tail, result.p_value_underflow, certain=observed == 0 or probability == 1
    )
    if lower == 0 and upper == n_cases:
        range_note = (
            f"All possible counts from 0 through {n_cases:,} are shown. "
            "Bar heights retain their original binomial probabilities."
        )
    else:
        range_note = (
            f"Counts {lower:,} through {upper:,} are shown. The omitted total probability "
            f"is at most {max_omitted_mass:g}; the omitted upper tail is also at most "
            f"{max_omitted_mass:g} of a positive full tail. Bar heights are not rescaled."
        )
    return BinomialView(
        n_cases=n_cases,
        probability=probability,
        observed_count=observed,
        expected_count=expected,
        lower_count=lower,
        upper_count=upper,
        points=points,
        omitted_left_mass=omitted_left,
        omitted_right_mass=omitted_right,
        omitted_mass=omitted_left + omitted_right,
        log_omitted_left_mass=log_omitted_left,
        log_omitted_right_mass=log_omitted_right,
        log_omitted_mass=float(np.logaddexp(log_omitted_left, log_omitted_right)),
        omitted_tail_fraction=omitted_tail_fraction,
        shown_tail_mass=shown_tail,
        log_shown_tail_mass=log_shown_tail,
        full_tail_probability=float(result.p_value),
        full_log_tail_probability=log_full_tail,
        tail_probability_underflow=bool(result.p_value_underflow),
        tail_clipped=isfinite(log_omitted_right),
        max_omitted_mass=max_omitted_mass,
        probability_label=label,
        frequency_label=frequency,
        range_note=range_note,
    )
