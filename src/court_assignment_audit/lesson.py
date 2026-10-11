"""An explicitly authored classroom example and separate random repetitions.

The sixty court labels below are a designed teaching fixture, not a random
sample or a discovery. Simulation uses independent draws with replacement;
expected counts are long-run averages, never assignment quotas.
"""

import csv
import hashlib
import io
import json
import platform
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from random import Random

import numpy as np
import scipy
from scipy.stats import binom

from court_assignment_audit.statistics import TargetTestResult, exact_target_test

N_CASES = 60
COURTS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L")
OBSERVED_A = 10
TARGET_PROBABILITY = 1 / 12
SIMULATIONS = 1_000
BASE_SEED = 271828
LESSON_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
CONSTRUCTION = "authored_teaching_fixture"

# Literal, authored assignment order. No shuffle, seed search, or generator is
# used to construct this fixture. A has 10; B-G have 5 each; H-L have 4 each.
_AUTHORED_COURTS = (
    "A",
    "D",
    "H",
    "B",
    "A",
    "J",
    "F",
    "L",
    "C",
    "I",
    "E",
    "K",
    "G",
    "A",
    "C",
    "E",
    "K",
    "B",
    "I",
    "A",
    "L",
    "D",
    "H",
    "F",
    "J",
    "F",
    "A",
    "H",
    "C",
    "L",
    "G",
    "E",
    "A",
    "B",
    "K",
    "D",
    "I",
    "E",
    "B",
    "A",
    "J",
    "G",
    "D",
    "K",
    "F",
    "H",
    "C",
    "L",
    "A",
    "G",
    "I",
    "B",
    "A",
    "E",
    "J",
    "F",
    "C",
    "G",
    "A",
    "D",
)


@dataclass(frozen=True)
class SimulationBatch:
    """One complete, unfiltered batch; frequency is the estimate b / 1,000.

    This teaching frequency is not a finite-correction Monte Carlo test
    p-value. Every retained vector represents exactly sixty new assignments.
    """

    batch_number: int
    seed: int
    target_counts: tuple[int, ...]
    allocations: tuple[tuple[int, ...], ...]
    exceedances: int
    frequency: float


def _integer(value: int, name: str, minimum: int, maximum: int | None = None) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bounds = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer {bounds}.")
    return int(value)


def lesson_records() -> list[dict[str, str | bool]]:
    """Return fresh mutable records backed by an immutable authored sequence."""
    return [
        {
            "case_id": f"LESSON-{index:03d}",
            "court": court,
            "synthetic": True,
            "construction": CONSTRUCTION,
        }
        for index, court in enumerate(_AUTHORED_COURTS, start=1)
    ]


def lesson_counts() -> tuple[int, ...]:
    """Return the fixed fixture counts in COURTS order."""
    return tuple(_AUTHORED_COURTS.count(court) for court in COURTS)


def exact_count_probability(k: int) -> float:
    """P(X = k), for X ~ Binomial(60, 1/12)."""
    k = _integer(k, "k", 0, N_CASES)
    return float(binom.pmf(k, N_CASES, TARGET_PROBABILITY))


def target_result(k: int = OBSERVED_A) -> TargetTestResult:
    """The fixed enrichment question, or a separately labeled practice count.

    This returns a calculation only; it never changes the authored records.
    The second category combines the other eleven courts for this binomial
    calculation. Complete simulations retain all twelve categories.
    """
    k = _integer(k, "k", 0, N_CASES)
    return exact_target_test(
        (k, N_CASES - k), (TARGET_PROBABILITY, 1 - TARGET_PROBABILITY), target_index=0
    )


def exact_tail_probability(k: int) -> float:
    """P(X >= k), including the observed count, under the fixed lesson model."""
    return target_result(k).p_value


def simulate_batch(batch_number: int = 1) -> SimulationBatch:
    """Generate exactly 1,000 full trials with seed BASE_SEED + batch - 1.

    A local Python Random instance makes sixty randrange(12) draws per trial.
    No trial is selected, discarded, stopped early, or replaced. Asking for
    the same batch number replays the same complete batch without affecting
    the module-level or NumPy random streams.
    """
    batch_number = _integer(batch_number, "batch_number", 1)
    seed = BASE_SEED + batch_number - 1
    rng = Random(seed)
    allocations = []
    for _ in range(SIMULATIONS):
        counts = [0] * len(COURTS)
        for _ in range(N_CASES):
            counts[rng.randrange(len(COURTS))] += 1
        allocations.append(tuple(counts))
    immutable_allocations = tuple(allocations)
    target_counts = tuple(counts[0] for counts in immutable_allocations)
    exceedances = sum(count >= OBSERVED_A for count in target_counts)
    return SimulationBatch(
        batch_number=batch_number,
        seed=seed,
        target_counts=target_counts,
        allocations=immutable_allocations,
        exceedances=exceedances,
        frequency=exceedances / SIMULATIONS,
    )


def lesson_csv() -> str:
    """Serialize only the authored fixture; UTF-8 encoding uses LF, no BOM."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=("case_id", "court", "synthetic", "construction"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(lesson_records())
    return output.getvalue()


def _validate_batch(batch: SimulationBatch) -> None:
    if not isinstance(batch, SimulationBatch):
        raise ValueError("batch must be a SimulationBatch or None.")
    number = _integer(batch.batch_number, "batch_number", 1)
    if batch.seed != BASE_SEED + number - 1:
        raise ValueError("batch seed does not match its batch number.")
    if len(batch.allocations) != SIMULATIONS or len(batch.target_counts) != SIMULATIONS:
        raise ValueError("A batch must contain all 1,000 complete trials.")
    for row in batch.allocations:
        if len(row) != len(COURTS):
            raise ValueError("Each allocation must contain all twelve courts.")
        values = tuple(_integer(value, "allocation count", 0, N_CASES) for value in row)
        if sum(values) != N_CASES:
            raise ValueError("Each allocation must total sixty assignments.")
    targets = tuple(row[0] for row in batch.allocations)
    if batch.target_counts != targets:
        raise ValueError("Target counts must match Court A in every complete allocation.")
    exceedances = sum(value >= OBSERVED_A for value in targets)
    if _integer(batch.exceedances, "exceedances", 0, SIMULATIONS) != exceedances:
        raise ValueError("Exceedances must count all trials with Court A >= 10.")
    if (
        isinstance(batch.frequency, (bool, np.bool_))
        or not isinstance(batch.frequency, Real)
        or not isfinite(batch.frequency)
        or batch.frequency != exceedances / SIMULATIONS
    ):
        raise ValueError("Batch frequency must equal exceedances / 1,000.")
    if batch.allocations != simulate_batch(number).allocations:
        raise ValueError("Batch allocations do not replay from the declared seed.")


def lesson_manifest(
    batch: SimulationBatch | None = None, practice_threshold: int | None = None
) -> dict:
    """Return JSON-safe provenance with fixed, practice, and simulation sections.

    Changing practice_threshold changes only the practice section. Optional
    practice frequencies reuse the supplied batch; they never draw new data.
    The CSV hash always refers to the same sixty authored records.
    Supplied simulation batches must replay exactly from their declared seed.
    """
    if batch is not None:
        _validate_batch(batch)
    if practice_threshold is not None:
        practice_threshold = _integer(practice_threshold, "practice_threshold", 0, N_CASES)
    observed = target_result()
    simulation = None
    if batch is not None:
        allocations = [[int(count) for count in row] for row in batch.allocations]
        allocations_bytes = json.dumps(allocations, separators=(",", ":")).encode("utf-8")
        simulation = {
            "batch_number": int(batch.batch_number),
            "seed": int(batch.seed),
            "base_seed": BASE_SEED,
            "seed_rule": "BASE_SEED + batch_number - 1",
            "rng": "Python random.Random(seed).randrange(12)",
            "trials": SIMULATIONS,
            "assignments_per_trial": N_CASES,
            "court_order": list(COURTS),
            "threshold": OBSERVED_A,
            "exceedances": int(batch.exceedances),
            "frequency": float(batch.frequency),
            "frequency_definition": "exceedances / trials; empirical simulation estimate",
            "target_counts": [int(count) for count in batch.target_counts],
            "allocations": allocations,
            "allocations_sha256": hashlib.sha256(allocations_bytes).hexdigest(),
            "hash_encoding": "UTF-8 compact JSON of allocations, separators=(',', ':')",
        }
    practice = None
    if practice_threshold is not None:
        practice = {
            "court": "A",
            "threshold": practice_threshold,
            "exact_count_probability": exact_count_probability(practice_threshold),
            "exact_tail_probability": exact_tail_probability(practice_threshold),
            "changes_fixed_dataset": False,
            "simulation": None,
        }
        if batch is not None:
            practice_exceedances = sum(count >= practice_threshold for count in batch.target_counts)
            practice["simulation"] = {
                "batch_number": int(batch.batch_number),
                "threshold": practice_threshold,
                "exceedances": practice_exceedances,
                "frequency": practice_exceedances / SIMULATIONS,
                "uses_same_complete_batch": True,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "lesson_version": LESSON_VERSION,
        "dataset": {
            "name": "Fictional company assignment lesson",
            "synthetic": True,
            "construction": CONSTRUCTION,
            "provenance": "Literal authored teaching fixture; not randomly sampled or discovered.",
            "fixture_seed": None,
            "case_count": N_CASES,
            "court_order": list(COURTS),
            "observed_counts": dict(zip(COURTS, lesson_counts(), strict=True)),
            "csv_sha256": hashlib.sha256(lesson_csv().encode("utf-8")).hexdigest(),
            "hash_encoding": "UTF-8 lesson.csv with LF newlines and no BOM",
        },
        "model": {
            "assignment_mechanism": "independent_with_replacement",
            "probabilities": {court: TARGET_PROBABILITY for court in COURTS},
            "preselected_teaching_target": "A",
            "direction": "greater",
            "expected_target_count": N_CASES * TARGET_PROBABILITY,
            "quotas": False,
        },
        "observed_analysis": {
            "court": "A",
            "count": OBSERVED_A,
            "expected_count": observed.expected_count,
            "exact_count_probability": exact_count_probability(OBSERVED_A),
            "exact_tail_probability": observed.p_value,
            "log_tail_probability": observed.log_p_value,
            "tail_definition": "P(X >= 10), including ten",
        },
        "practice": practice,
        "simulation": simulation,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "lesson": LESSON_VERSION,
        },
        "interpretation": (
            "This is a constructed lesson, not an empirical discovery. Probabilities describe "
            "counts under the specified assignment model, not wrongdoing or causal explanations. "
            "A zero simulation frequency does not make an event impossible."
        ),
    }
