"""Plain-language readings and unselected examples of the declared mechanism."""

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from court_assignment_audit.reporting import MC_SEED
from court_assignment_audit.statistics import GlobalTestResult, TargetTestResult


@dataclass(frozen=True)
class Reading:
    """A descriptive UI reading, conditional on the specified assignment model."""

    status: str
    headline: str
    explanation: str


def _integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        adjective = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {adjective} integer.")
    return int(value)


def global_reading(result: GlobalTestResult, n_cases: int, reference: float = 0.05) -> Reading:
    """Read the global result using its simulation uncertainty interval.

    Endpoint equality is deliberately borderline. A one-assignment sample
    receives the sample-size caveat regardless of its simulated tail result.
    """
    n_cases = _integer(n_cases, "n_cases")
    if n_cases != sum(result.observed_counts):
        raise ValueError("n_cases must match the global result's observed total.")
    if (
        isinstance(reference, (bool, np.bool_))
        or not isinstance(reference, Real)
        or not 0 < reference < 1
        or not isfinite(reference)
    ):
        raise ValueError("reference must be a finite number strictly between 0 and 1.")

    if n_cases == 1:
        receiving_index = result.observed_counts.index(1)
        chance = result.expected_counts[receiving_index]
        support_note = (
            " The observed assignment is impossible under the specified model."
            if chance == 0
            else " One observed assignment does not provide a stable allocation pattern."
        )
        return Reading(
            status="limited",
            headline="One assignment cannot establish a stable allocation pattern.",
            explanation=(
                f"Model probability of assignment to this court: {100 * chance:.3g}%.{support_note}"
            ),
        )

    if result.support_violation:
        return Reading(
            status="unusual",
            headline="The observed allocation is impossible under the specified model",
            explanation=(
                "At least one court received an assignment despite having a specified "
                "probability of zero. This observation cannot occur under the declared "
                "assignment model."
            ),
        )

    reference_label = f"{100 * reference:.3g}%"
    if result.p_value < reference and result.mc_tail_ci_high < reference:
        return Reading(
            status="unusual",
            headline=(
                "The overall allocation is unusual under the specified random-assignment model"
            ),
            explanation=(
                "Random allocations of the same number of cases rarely show this much "
                "or more overall discrepancy."
            ),
        )
    if result.mc_tail_ci_low > reference:
        return Reading(
            status="no_clear_departure",
            headline=(
                "The overall allocation does not show a clear departure from the specified model"
            ),
            explanation=(
                "The observed differences are compatible with variation under the specified "
                "random-assignment model. This does not establish that the mechanism was followed."
            ),
        )
    return Reading(
        status="borderline",
        headline=(
            "The simulations do not clearly resolve whether the p-value is below the "
            f"{reference_label} reference threshold"
        ),
        explanation=(
            "The current simulations do not give a clear reading. More simulations can "
            "improve precision; they do not add assignment data."
        ),
    )


def target_reading(result: TargetTestResult) -> Reading:
    """Read the fixed, preselected one-sided enrichment check at 5%."""
    if result.observed_count == result.expected_count:
        return Reading(
            status="no_clear_excess",
            headline="The preselected court matches the expected count",
            explanation=(
                "The observed count matches the model's expected count, which is an "
                "average across repeated random allocations. This match does not establish "
                "that the assignment mechanism was followed."
            ),
        )
    if result.observed_count < result.expected_count:
        return Reading(
            status="no_clear_excess",
            headline="This check looks for extra assignments, not a shortfall",
            explanation=(
                "The preselected court is below its expected count. This one-sided "
                "enrichment check does not assess whether assignments are unusually low."
            ),
        )
    if result.lift_pp > 0 and result.p_value < 0.05:
        support_note = (
            " An assignment to a court with zero specified probability is impossible "
            "under that model."
            if result.expected_probability == 0
            else ""
        )
        sample_note = (
            " One assignment alone does not establish a stable allocation pattern."
            if result.sample_size == 1
            else ""
        )
        return Reading(
            status="unusual_excess",
            headline="The preselected court shows an unusual excess under the specified model",
            explanation=(
                "With the court's stated assignment chance, random allocations rarely give "
                "it this many or more assignments. Both the court and the decision to look "
                "for extra assignments must have been chosen in advance."
                f"{support_note}{sample_note}"
            ),
        )
    return Reading(
        status="no_clear_excess",
        headline="No clear excess for the preselected court",
        explanation=(
            "The observed count is above its expectation, but there is no clear evidence "
            "of excess under the specified model."
        ),
    )


def reference_allocations(
    n_cases: int,
    probabilities: Sequence[float],
    seed: int = MC_SEED,
    examples: int = 12,
) -> np.ndarray:
    """Return the first joint multinomial vectors from one fixed random stream.

    Examples use the full declared probabilities and observed sample size.
    They are illustrative, with no selection based on observed data, apparent
    similarity, discrepancy, or a significance threshold.
    """
    n_cases = _integer(n_cases, "n_cases")
    if n_cases > np.iinfo(np.int64).max:
        raise ValueError("n_cases exceeds the supported integer range.")
    seed = _integer(seed, "seed", allow_zero=True)
    examples = _integer(examples, "examples")
    raw = np.asarray(probabilities, dtype=object)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("probabilities must be a nonempty one-dimensional sequence.")
    values = []
    for value in raw:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not 0 <= value <= 1
            or not isfinite(value)
        ):
            raise ValueError("Probabilities must be finite numbers between 0 and 1.")
        values.append(float(value))
    if not isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Probabilities must sum to 1 (absolute tolerance 1e-12).")
    probabilities_array = np.asarray(values)
    probabilities_array /= probabilities_array.sum()
    return np.random.default_rng(seed).multinomial(n_cases, probabilities_array, size=examples)
