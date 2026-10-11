"""Generator contracts; no statistical inference or probabilistic pass/fail tests."""

import json
import math
import random
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from court_assignment_audit.synthetic import (
    COURTS,
    GENERATOR_VERSION,
    MAX_CASES,
    PRESELECTED_TARGET,
    SimulationConfig,
    generate_dataset,
)


class GeneratorTests(unittest.TestCase):
    def test_same_config_reproduces_rows_and_metadata(self):
        for scenario in ("null", "anomaly"):
            with self.subTest(scenario=scenario):
                config = SimulationConfig(n_cases=100, seed=7123, scenario=scenario)
                first = generate_dataset(config)
                second = generate_dataset(config)
                self.assertEqual(first, second)
                self.assertEqual(first.metadata, second.metadata)

    def test_stable_seed_fixture(self):
        # The first ten random.Random(42) uniforms map to these fixed intervals.
        dataset = generate_dataset(SimulationConfig(n_cases=10, seed=42))
        self.assertEqual(
            [row.court for row in dataset.rows],
            [
                "Court B",
                "Court A",
                "Court A",
                "Court A",
                "Court C",
                "Court B",
                "Court C",
                "Court A",
                "Court B",
                "Court A",
            ],
        )
        self.assertEqual(dataset.counts, {"Court A": 5, "Court B": 3, "Court C": 2, "Court D": 0})

    def test_every_case_uses_an_independent_draw_without_quotas(self):
        # Every draw can select the same court, even after its expectation is exceeded.
        with patch("court_assignment_audit.synthetic.random.Random") as random_class:
            random_class.return_value.random.return_value = 0.95
            dataset = generate_dataset(SimulationConfig(n_cases=50, seed=913))
        random_class.assert_called_once_with(913)
        self.assertEqual(random_class.return_value.random.call_count, 50)
        self.assertEqual(dataset.counts, {"Court A": 0, "Court B": 0, "Court C": 0, "Court D": 50})

    def test_weighted_probability_interval_boundaries(self):
        draws = [
            0.0,
            math.nextafter(0.4, 0.0),
            0.4,
            math.nextafter(0.7, 0.0),
            0.7,
            0.89,
            0.91,
            math.nextafter(1.0, 0.0),
        ]
        with patch("court_assignment_audit.synthetic.random.Random") as random_class:
            random_class.return_value.random.side_effect = draws
            dataset = generate_dataset(SimulationConfig(n_cases=len(draws)))
        self.assertEqual(
            [row.court for row in dataset.rows], [court for court in COURTS for _ in range(2)]
        )

    def test_anomaly_draws_use_alternative_not_null_probabilities(self):
        with patch("court_assignment_audit.synthetic.random.Random") as random_class:
            random_class.return_value.random.return_value = 0.5
            null = generate_dataset(SimulationConfig(n_cases=3, scenario="null"))
            anomaly = generate_dataset(SimulationConfig(n_cases=3, scenario="anomaly"))
        self.assertEqual(null.counts["Court B"], 3)
        self.assertEqual(anomaly.counts["Court A"], 3)
        self.assertEqual(anomaly.metadata["null_probabilities"], [0.4, 0.3, 0.2, 0.1])
        self.assertEqual(anomaly.metadata["generating_probabilities"], [0.58, 0.21, 0.14, 0.07])

    def test_zero_probability_courts_never_receive_boundary_draws(self):
        for probabilities, expected_court in (
            ((0, 1, 0, 0), "Court B"),
            ((0, 0, 0, 1), "Court D"),
            ((1, 0, 0, 0), "Court A"),
        ):
            with self.subTest(probabilities=probabilities):
                with patch("court_assignment_audit.synthetic.random.Random") as random_class:
                    random_class.return_value.random.side_effect = [0.0, math.nextafter(1.0, 0.0)]
                    dataset = generate_dataset(
                        SimulationConfig(n_cases=2, null_probabilities=probabilities)
                    )
                self.assertEqual(
                    [row.court for row in dataset.rows], [expected_court, expected_court]
                )
                self.assertEqual(set(dataset.counts), set(COURTS))

    def test_sample_length_and_ids(self):
        dataset = generate_dataset(SimulationConfig(n_cases=123, seed=0))
        self.assertEqual(len(dataset.rows), 123)
        self.assertEqual(sum(dataset.counts.values()), 123)
        self.assertEqual(dataset.rows[0].case_id, "CASE-000001")
        self.assertEqual(dataset.rows[-1].case_id, "CASE-000123")
        self.assertEqual(len({row.case_id for row in dataset.rows}), 123)

    def test_extending_a_sample_preserves_its_prefix(self):
        small = generate_dataset(SimulationConfig(n_cases=10, seed=100))
        large = generate_dataset(SimulationConfig(n_cases=100, seed=100))
        self.assertEqual(small.rows, large.rows[:10])

    def test_generator_leaves_global_rng_state_untouched(self):
        original_state = random.getstate()
        generate_dataset(SimulationConfig(n_cases=10, seed=999))
        self.assertEqual(random.getstate(), original_state)

    def test_metadata_is_reproducible_json_and_records_preselection(self):
        dataset = generate_dataset(SimulationConfig(n_cases=4, seed=71, scenario="anomaly"))
        metadata = dataset.metadata
        self.assertEqual(json.loads(json.dumps(metadata, allow_nan=False)), metadata)
        self.assertIs(metadata["synthetic"], True)
        self.assertEqual(metadata["generator_version"], GENERATOR_VERSION)
        self.assertEqual(metadata["seed"], 71)
        self.assertEqual(metadata["n_cases"], 4)
        self.assertEqual(metadata["scenario"], "anomaly")
        self.assertEqual(metadata["court_order"], list(COURTS))
        self.assertEqual(metadata["preselected_target"], PRESELECTED_TARGET)
        self.assertIn("Fixed before any draws", metadata["target_selection"])
        self.assertIn("100% synthetic", metadata["provenance"])
        self.assertNotIn("timestamp", metadata)

    def test_exports_cannot_mutate_dataset(self):
        dataset = generate_dataset(SimulationConfig(n_cases=2))
        records = dataset.to_records()
        records[0]["court"] = "changed"
        metadata = dataset.metadata
        metadata["court_order"].append("changed")
        counts = dataset.counts
        counts["Court A"] = -1
        self.assertIn(dataset.rows[0].court, COURTS)
        self.assertEqual(dataset.metadata["court_order"], list(COURTS))
        self.assertGreaterEqual(dataset.counts["Court A"], 0)
        with self.assertRaises(FrozenInstanceError):
            dataset.rows[0].court = "changed"

    def test_mutable_probability_input_is_copied(self):
        probabilities = [0.4, 0.3, 0.2, 0.1]
        config = SimulationConfig(null_probabilities=probabilities)
        probabilities[0] = 0.99
        self.assertEqual(config.null_probabilities, (0.4, 0.3, 0.2, 0.1))
        self.assertIsInstance(config.null_probabilities, tuple)

    def test_default_generation_is_available(self):
        dataset = generate_dataset()
        self.assertEqual(len(dataset.rows), SimulationConfig().n_cases)


class ConfigurationValidationTests(unittest.TestCase):
    def test_invalid_case_counts(self):
        for value in (0, -1, MAX_CASES + 1, True, 10.5, "10", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SimulationConfig(n_cases=value)

    def test_invalid_seeds(self):
        for value in (-1, True, 1.5, "42", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SimulationConfig(seed=value)

    def test_invalid_scenarios(self):
        for value in ("", "NULL", "biased", None, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SimulationConfig(scenario=value)

    def test_invalid_probability_vectors_for_both_scenarios(self):
        invalid_vectors = (
            (),
            (0.5, 0.5),
            (0.1, 0.2, 0.3, 0.4, 0),
            (0, 0, 0, 0),
            (0.4, 0.3, 0.2, 0.2),
            (-0.1, 0.4, 0.3, 0.4),
            (1.1, 0, 0, -0.1),
            (float("nan"), 0.3, 0.2, 0.1),
            (float("inf"), 0.3, 0.2, 0.1),
            (10**400, 0, 0, 0),
            (True, 0, 0, 0),
            ("0.4", 0.3, 0.2, 0.1),
            None,
        )
        for parameter in ("null_probabilities", "anomaly_probabilities"):
            for probabilities in invalid_vectors:
                with (
                    self.subTest(parameter=parameter, probabilities=probabilities),
                    self.assertRaises(ValueError),
                ):
                    SimulationConfig(**{parameter: probabilities})

    def test_floating_point_sum_roundoff_is_tolerated(self):
        config = SimulationConfig(null_probabilities=(0.1, 0.2, 0.3, 0.4000000000000001))
        self.assertAlmostEqual(math.fsum(config.null_probabilities), 1.0)

    def test_invalid_generator_config(self):
        for config in ({}, "null", 10):
            with self.subTest(config=config), self.assertRaises(TypeError):
                generate_dataset(config)


if __name__ == "__main__":
    unittest.main()
