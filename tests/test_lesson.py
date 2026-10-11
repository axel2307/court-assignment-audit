"""Independent mathematics, authored-fixture integrity, and complete batch exports."""

import csv
import hashlib
import io
import json
import random
import runpy
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pytest

from court_assignment_audit.lesson import (
    BASE_SEED,
    CONSTRUCTION,
    COURTS,
    N_CASES,
    OBSERVED_A,
    SIMULATIONS,
    TARGET_PROBABILITY,
    exact_count_probability,
    exact_tail_probability,
    lesson_counts,
    lesson_csv,
    lesson_manifest,
    lesson_records,
    simulate_batch,
    target_result,
)


@pytest.fixture(scope="module")
def batch_one():
    return simulate_batch(1)


@pytest.fixture(scope="module")
def rational_masses():
    p = Fraction(1, 12)
    return tuple(Fraction(comb(60, k)) * p**k * (1 - p) ** (60 - k) for k in range(61))


@pytest.mark.parametrize("k", range(61))
def test_every_exact_count_and_inclusive_tail_matches_independent_rational_sum(k, rational_masses):
    expected_tail = float(sum(rational_masses[k:], Fraction(0)))
    assert exact_count_probability(k) == pytest.approx(float(rational_masses[k]), rel=2e-13, abs=0)
    assert exact_tail_probability(k) == pytest.approx(expected_tail, rel=2e-13, abs=0)
    result = target_result(k)
    assert result.observed_count == k
    assert result.sample_size == 60
    assert result.expected_probability == 1 / 12
    assert result.expected_count == 5
    assert result.p_value == exact_tail_probability(k)


def test_lesson_constants_and_authored_counts():
    assert N_CASES == 60
    assert COURTS == tuple("ABCDEFGHIJKL")
    assert OBSERVED_A == 10
    assert TARGET_PROBABILITY == 1 / 12
    assert SIMULATIONS == 1000
    assert BASE_SEED == 271828
    assert lesson_counts() == (10, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4)
    assert sum(lesson_counts()) == N_CASES
    assert exact_count_probability(10) == pytest.approx(0.01570711556597809)
    assert exact_tail_probability(10) == pytest.approx(0.025754254728917188)


def test_authored_records_are_complete_explicit_and_isolated_from_external_mutation():
    original = lesson_records()
    assert len(original) == 60
    assert [row["case_id"] for row in original] == [
        f"LESSON-{number:03d}" for number in range(1, 61)
    ]
    assert all(row["synthetic"] is True for row in original)
    assert all(row["construction"] == "authored_teaching_fixture" for row in original)
    observed = Counter(row["court"] for row in original)
    assert tuple(observed[court] for court in COURTS) == lesson_counts()
    changed = lesson_records()
    changed[0]["court"] = "L"
    changed.pop()
    changed.reverse()
    assert lesson_records() == original
    assert lesson_counts()[0] == 10


def test_simulation_replays_independent_counter_based_full_trials(batch_one):
    rng = random.Random(271828)
    independent = []
    for _ in range(1000):
        trial = Counter(rng.randrange(12) for _ in range(60))
        independent.append(tuple(trial[court] for court in range(12)))
    assert batch_one.allocations == tuple(independent)
    assert batch_one.allocations[0] == (6, 5, 6, 6, 5, 5, 5, 2, 3, 9, 6, 2)
    assert batch_one.allocations[-1] == (3, 4, 6, 6, 6, 7, 4, 7, 3, 4, 4, 6)
    assert batch_one.target_counts == tuple(row[0] for row in independent)
    assert sum(batch_one.target_counts) == 5039
    assert batch_one.exceedances == 26
    assert batch_one.frequency == 26 / 1000
    assert batch_one.frequency != 27 / 1001  # This is an empirical frequency, not a test p-value.


def test_simulations_retain_every_complete_trial_and_use_distinct_reproducible_batches(batch_one):
    assert batch_one.batch_number == 1
    assert batch_one.seed == BASE_SEED
    assert len(batch_one.allocations) == len(batch_one.target_counts) == SIMULATIONS
    assert all(len(row) == len(COURTS) and sum(row) == N_CASES for row in batch_one.allocations)
    assert all(type(count) is int and count >= 0 for row in batch_one.allocations for count in row)
    assert simulate_batch(1) == batch_one
    second = simulate_batch(2)
    assert second.seed == BASE_SEED + 1
    assert second.allocations != batch_one.allocations
    assert second == simulate_batch(2)


def test_simulation_batch_is_deeply_immutable(batch_one):
    with pytest.raises(FrozenInstanceError):
        batch_one.seed = 0
    with pytest.raises(TypeError):
        batch_one.allocations[0][0] = 0
    with pytest.raises(TypeError):
        batch_one.target_counts[0] = 0


def test_local_rng_leaves_python_and_numpy_global_streams_unchanged():
    random.seed(81)
    np.random.seed(92)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    simulate_batch(3)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]


def test_csv_bytes_round_trip_and_hash_are_deterministic():
    csv_text = lesson_csv()
    csv_bytes = csv_text.encode("utf-8")
    assert csv_text == lesson_csv()
    assert csv_bytes.endswith(b"\n")
    assert b"\r" not in csv_bytes
    assert not csv_bytes.startswith(b"\xef\xbb\xbf")
    records = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(records) == 60
    assert [row["court"] for row in records] == [row["court"] for row in lesson_records()]
    assert all(
        row["synthetic"] == "True" and row["construction"] == CONSTRUCTION for row in records
    )
    manifest = lesson_manifest()
    assert manifest["dataset"]["csv_sha256"] == hashlib.sha256(csv_bytes).hexdigest()
    assert manifest["dataset"]["fixture_seed"] is None
    assert manifest["dataset"]["construction"] == CONSTRUCTION
    assert manifest["simulation"] is None
    assert manifest["practice"] is None


def test_manifest_serializes_all_trials_and_verifiable_hashes(batch_one):
    manifest = lesson_manifest(batch_one)
    assert manifest == json.loads(json.dumps(manifest, allow_nan=False))
    simulation = manifest["simulation"]
    assert simulation["seed"] == 271828
    assert simulation["threshold"] == 10
    assert simulation["exceedances"] == 26
    assert simulation["frequency"] == 0.026
    assert len(simulation["allocations"]) == len(simulation["target_counts"]) == 1000
    encoded = json.dumps(simulation["allocations"], separators=(",", ":")).encode("utf-8")
    assert simulation["allocations_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert manifest["model"]["quotas"] is False
    assert manifest["model"]["assignment_mechanism"] == "independent_with_replacement"
    assert manifest["observed_analysis"]["count"] == 10
    assert manifest["observed_analysis"]["exact_tail_probability"] == exact_tail_probability(10)
    assert set(manifest["software"]) == {"python", "numpy", "scipy", "lesson"}


@pytest.mark.parametrize("threshold", [0, 5, 12, 60])
def test_practice_changes_only_practice_and_reuses_the_same_complete_batch(batch_one, threshold):
    initial_csv = lesson_csv()
    initial = lesson_manifest(batch_one)
    practice = lesson_manifest(batch_one, practice_threshold=threshold)
    assert lesson_csv() == initial_csv
    assert lesson_counts()[0] == 10
    assert {key: value for key, value in practice.items() if key != "practice"} == {
        key: value for key, value in initial.items() if key != "practice"
    }
    hypothetical = practice["practice"]
    assert hypothetical["threshold"] == threshold
    assert hypothetical["changes_fixed_dataset"] is False
    assert hypothetical["exact_tail_probability"] == exact_tail_probability(threshold)
    assert hypothetical["simulation"]["exceedances"] == sum(
        k >= threshold for k in batch_one.target_counts
    )
    assert hypothetical["simulation"]["uses_same_complete_batch"] is True
    if threshold == 60:
        assert hypothetical["simulation"]["frequency"] == 0
        assert hypothetical["exact_tail_probability"] > 0


def test_manifest_returns_fresh_mutable_data_without_changing_future_exports(batch_one):
    original = lesson_manifest(batch_one, practice_threshold=5)
    changed = lesson_manifest(batch_one, practice_threshold=5)
    changed["dataset"]["observed_counts"]["A"] = 999
    changed["simulation"]["allocations"][0][0] = 999
    assert lesson_manifest(batch_one, practice_threshold=5) == original
    assert batch_one.allocations[0][0] == 6


@pytest.mark.parametrize("k", [-1, 61, True, False, 1.5, "10", None, float("nan")])
def test_invalid_count_inputs(k):
    for function in (target_result, exact_count_probability, exact_tail_probability):
        with pytest.raises(ValueError):
            function(k)
    if k is not None:
        with pytest.raises(ValueError):
            lesson_manifest(practice_threshold=k)


@pytest.mark.parametrize("batch_number", [0, -1, True, False, 1.5, "1", None])
def test_invalid_batch_numbers(batch_number):
    with pytest.raises(ValueError):
        simulate_batch(batch_number)


def test_manifest_rejects_incomplete_or_inconsistent_batches(batch_one):
    invalid = [
        replace(batch_one, seed=BASE_SEED + 1),
        replace(batch_one, allocations=batch_one.allocations[:-1]),
        replace(batch_one, target_counts=batch_one.target_counts[:-1]),
        replace(batch_one, target_counts=(99,) + batch_one.target_counts[1:]),
        replace(batch_one, allocations=((59,) + (0,) * 11,) + batch_one.allocations[1:]),
        replace(batch_one, exceedances=25),
        replace(batch_one, frequency=0.5),
        replace(batch_one, frequency=float("nan")),
    ]
    for batch in invalid:
        with pytest.raises(ValueError):
            lesson_manifest(batch)
    with pytest.raises(ValueError):
        lesson_manifest(batch="1")


def test_manifest_rejects_structurally_valid_allocations_that_do_not_match_the_seed(batch_one):
    first = list(batch_one.allocations[0])
    first[1] += 1
    first[2] -= 1
    modified = replace(batch_one, allocations=(tuple(first),) + batch_one.allocations[1:])
    assert sum(modified.allocations[0]) == N_CASES
    assert modified.target_counts == batch_one.target_counts
    assert modified.exceedances == batch_one.exceedances
    with pytest.raises(ValueError, match="do not replay"):
        lesson_manifest(modified)


def test_export_script_writes_the_documented_artifacts_and_exact_csv_hash(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_lesson.py"
    main = runpy.run_path(str(script), run_name="lesson_export_test")["main"]
    destination = tmp_path / "lesson"
    assert main(["--output", str(destination), "--batch", "1", "--practice-threshold", "12"]) == 0
    csv_bytes = (destination / "lesson.csv").read_bytes()
    manifest_bytes = (destination / "lesson_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert csv_bytes == lesson_csv().encode("utf-8")
    assert manifest["dataset"]["csv_sha256"] == hashlib.sha256(csv_bytes).hexdigest()
    assert manifest["observed_analysis"]["count"] == 10
    assert manifest["practice"]["threshold"] == 12
    assert manifest["simulation"]["exceedances"] == 26
    assert b"\r" not in manifest_bytes
    assert main(["--output", str(destination), "--batch", "1", "--practice-threshold", "12"]) == 0
    assert (destination / "lesson_manifest.json").read_bytes() == manifest_bytes


def test_export_invalid_arguments_do_not_create_output(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_lesson.py"
    main = runpy.run_path(str(script), run_name="lesson_export_test")["main"]
    destination = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        main(["--output", str(destination), "--batch", "0"])
    assert error.value.code == 2
    assert not destination.exists()
