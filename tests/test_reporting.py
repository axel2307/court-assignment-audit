import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from court_assignment_audit.cli import main
from court_assignment_audit.reporting import (
    MC_SEED,
    assignments_csv,
    build_run,
    manifest_json,
)
from court_assignment_audit.synthetic import COURTS, SimulationConfig


class ReportingTests(unittest.TestCase):
    def test_manifest_and_csv_are_reproducible_and_linked(self):
        config = SimulationConfig(n_cases=40, seed=812)
        first = build_run(config, simulations=200)
        second = build_run(config, simulations=200)
        self.assertEqual(assignments_csv(first), assignments_csv(second))
        self.assertEqual(manifest_json(first), manifest_json(second))
        manifest = json.loads(manifest_json(first))
        checksum = hashlib.sha256(assignments_csv(first).encode("utf-8")).hexdigest()
        self.assertEqual(manifest["assignments_sha256"], checksum)
        self.assertEqual(manifest["analysis"]["global"]["seed"], MC_SEED)
        self.assertEqual(manifest["generation"]["seed"], 812)
        self.assertTrue(manifest["generation"]["synthetic"])

    def test_csv_contains_only_generated_cases_and_generic_labels(self):
        run = build_run(SimulationConfig(n_cases=8), simulations=200)
        rows = list(csv.DictReader(io.StringIO(assignments_csv(run))))
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["case_id"], "CASE-000001")
        self.assertTrue(all(row["court"] in COURTS for row in rows))
        self.assertTrue(all(row["synthetic"] == "True" for row in rows))
        self.assertEqual(set(rows[0]), {"case_id", "court", "synthetic"})

    def test_anomaly_uses_null_baseline_for_analysis(self):
        run = build_run(SimulationConfig(n_cases=100, scenario="anomaly"), simulations=200)
        for actual, expected in zip(
            run["global"].expected_counts, (40.0, 30.0, 20.0, 10.0), strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(run["target"].expected_probability, 0.4)
        self.assertEqual(len(run["exploratory"]), 4)

    def test_monte_carlo_precision_does_not_change_assignments(self):
        config = SimulationConfig(n_cases=40, seed=71)
        self.assertEqual(
            assignments_csv(build_run(config, simulations=100)),
            assignments_csv(build_run(config, simulations=300)),
        )

    def test_zero_probability_support_violation_has_valid_json(self):
        config = SimulationConfig(
            n_cases=5,
            scenario="anomaly",
            null_probabilities=(0, 1, 0, 0),
            anomaly_probabilities=(1, 0, 0, 0),
        )
        manifest = json.loads(manifest_json(build_run(config, simulations=100)))
        result = manifest["analysis"]["global"]
        self.assertTrue(result["support_violation"])
        self.assertIsNone(result["statistic"])
        self.assertIn("Infinite", result["statistic_note"])
        self.assertEqual(result["p_value"], 1 / 101)

    def test_underflow_exports_finite_log_tail_instead_of_exact_zero(self):
        config = SimulationConfig(n_cases=100_000, scenario="anomaly", seed=20261011)
        manifest = json.loads(manifest_json(build_run(config, simulations=100)))
        target = manifest["analysis"]["preselected_target"]
        self.assertTrue(target["p_value_underflow"])
        self.assertIsNone(target["p_value"])
        self.assertLess(target["log_p_value"], -1000)
        self.assertEqual(manifest["analysis_version"], "1.0.0")

    def test_cli_exports_consistent_files(self):
        temporary_root = Path(__file__).resolve().parents[1] / "artifacts"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            args = ["demo", "--n-cases", "12", "--seed", "123", "--output-dir", folder]
            with patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO):
                main()
            output = Path(folder)
            contents = (output / "synthetic_assignments.csv").read_bytes()
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(list(csv.DictReader(io.StringIO(contents.decode())))), 12)
            self.assertEqual(manifest["assignments_sha256"], hashlib.sha256(contents).hexdigest())


if __name__ == "__main__":
    unittest.main()
