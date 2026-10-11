"""Reproducible, entirely synthetic independent court-assignment draws.

The null scenario samples from the stated assignment probabilities. The anomaly
scenario samples from a separate illustrative probability vector. Neither
scenario encodes misconduct, and no source records are used by this module.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from typing import Literal

COURTS = ("Court A", "Court B", "Court C", "Court D")
PRESELECTED_TARGET = COURTS[0]
DEFAULT_NULL_PROBABILITIES = (0.4, 0.3, 0.2, 0.1)
DEFAULT_ANOMALY_PROBABILITIES = (0.58, 0.21, 0.14, 0.07)
GENERATOR_VERSION = "1.0.0"
MAX_CASES = 100_000


def _validated_probabilities(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    """Validate and remove only floating-point sum roundoff from a vector."""
    try:
        probabilities = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain one probability per court.") from exc
    if len(probabilities) != len(COURTS):
        raise ValueError(f"{name} must contain exactly {len(COURTS)} probabilities.")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in probabilities):
        raise ValueError(f"{name} must contain finite numeric probabilities.")
    try:
        probabilities = tuple(float(value) for value in probabilities)
    except OverflowError as exc:
        raise ValueError(f"{name} probabilities must be between 0 and 1.") from exc
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError(f"{name} probabilities must be finite and between 0 and 1.")
    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} probabilities must sum to 1.")
    return tuple(value / total for value in probabilities)


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters fixed before drawing independent categorical assignments.

    Probability entries follow ``COURTS`` order. The preselected target is
    always Court A. Alternative probabilities affect only the anomaly scenario.
    Custom vectors are a programmatic feature; the demo can use the defaults.
    """

    n_cases: int = 1_000
    seed: int = 20261011
    scenario: Literal["null", "anomaly"] = "null"
    null_probabilities: tuple[float, ...] = DEFAULT_NULL_PROBABILITIES
    anomaly_probabilities: tuple[float, ...] = DEFAULT_ANOMALY_PROBABILITIES

    def __post_init__(self) -> None:
        if isinstance(self.n_cases, bool) or not isinstance(self.n_cases, int):
            raise ValueError("n_cases must be an integer.")
        if not 1 <= self.n_cases <= MAX_CASES:
            raise ValueError(f"n_cases must be between 1 and {MAX_CASES:,}.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if self.scenario not in ("null", "anomaly"):
            raise ValueError("scenario must be 'null' or 'anomaly'.")
        for name in ("null_probabilities", "anomaly_probabilities"):
            object.__setattr__(self, name, _validated_probabilities(getattr(self, name), name))

    @property
    def generating_probabilities(self) -> tuple[float, ...]:
        """The known categorical probabilities actually used for the draws."""
        return self.null_probabilities if self.scenario == "null" else self.anomaly_probabilities


@dataclass(frozen=True)
class Assignment:
    case_id: str
    court: str


@dataclass(frozen=True)
class SyntheticDataset:
    config: SimulationConfig
    rows: tuple[Assignment, ...]

    @property
    def counts(self) -> dict[str, int]:
        """Observed counts in court order, including courts with zero cases."""
        observed = Counter(row.court for row in self.rows)
        return {court: observed[court] for court in COURTS}

    @property
    def metadata(self) -> dict[str, object]:
        """Return reproducible JSON-compatible metadata, with no wall-clock time."""
        return {
            "provenance": (
                "100% synthetic; generated from independent categorical draws; "
                "no source records used."
            ),
            "synthetic": True,
            "generator_version": GENERATOR_VERSION,
            "random_generator": "Python random.Random / Mersenne Twister",
            "sampling_method": (
                "One independent categorical draw per case, with replacement; "
                "no fixed court totals."
            ),
            "n_cases": self.config.n_cases,
            "seed": self.config.seed,
            "scenario": self.config.scenario,
            "court_order": list(COURTS),
            "preselected_target": PRESELECTED_TARGET,
            "target_selection": "Fixed before any draws; independent of observed counts.",
            "null_probabilities": list(self.config.null_probabilities),
            "anomaly_probabilities": list(self.config.anomaly_probabilities),
            "generating_probabilities": list(self.config.generating_probabilities),
        }

    def to_records(self) -> list[dict[str, str]]:
        """Return fresh records suitable for a DataFrame or CSV export."""
        return [{"case_id": row.case_id, "court": row.court} for row in self.rows]


def generate_dataset(config: SimulationConfig | None = None) -> SyntheticDataset:
    """Generate rows without reading files or changing global random state.

    An isolated seeded PRNG supplies one uniform variate for each case. Repeated
    assignments are allowed; totals are outcomes of sampling, never quotas.
    """
    if config is None:
        config = SimulationConfig()
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig instance.")
    probabilities = config.generating_probabilities
    cumulative = tuple(math.fsum(probabilities[: index + 1]) for index in range(len(COURTS)))
    last_positive = max(index for index, probability in enumerate(probabilities) if probability > 0)
    rng = random.Random(config.seed)
    rows = []
    for case_number in range(1, config.n_cases + 1):
        draw = rng.random()
        court_index = next(
            (index for index, boundary in enumerate(cumulative) if draw < boundary),
            last_positive,
        )
        rows.append(Assignment(case_id=f"CASE-{case_number:06d}", court=COURTS[court_index]))
    return SyntheticDataset(config=config, rows=tuple(rows))
