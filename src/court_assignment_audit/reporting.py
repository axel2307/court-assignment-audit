"""Reproducible exports built only from the in-process synthetic generator."""

import csv
import hashlib
import io
import json
import math
import platform
from dataclasses import asdict
from importlib.metadata import version

from court_assignment_audit.statistics import (
    exact_target_test,
    exploratory_court_tests,
    global_multinomial_test,
)
from court_assignment_audit.synthetic import COURTS, SimulationConfig, generate_dataset

MC_SEED = 271828
MC_SIMULATIONS = 20_000
ANALYSIS_VERSION = "1.0.0"


def _serialized_result(result) -> dict:
    values = asdict(result)
    nonfinite = {}
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite[key] = str(value)
            values[key] = None
    if nonfinite:
        values["nonfinite_fields"] = nonfinite
    for prefix in ("", "adjusted_"):
        if values.get(f"{prefix}p_value_underflow"):
            values[f"{prefix}p_value"] = None
    return values


def build_run(config: SimulationConfig, *, simulations: int = MC_SIMULATIONS) -> dict:
    """Generate cases and analyze them against the specified, known null mechanism."""
    dataset = generate_dataset(config)
    counts = tuple(dataset.counts[court] for court in COURTS)
    probabilities = config.null_probabilities
    return {
        "dataset": dataset,
        "global": global_multinomial_test(
            counts, probabilities, simulations=simulations, seed=MC_SEED
        ),
        "target": exact_target_test(counts, probabilities, target_index=0),
        "exploratory": exploratory_court_tests(counts, probabilities),
    }


def assignments_csv(run: dict) -> str:
    """Serialize generated assignments, retaining explicit synthetic provenance."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["case_id", "court", "synthetic"])
    writer.writeheader()
    for row in run["dataset"].to_records():
        writer.writerow({"case_id": row["case_id"], "court": row["court"], "synthetic": True})
    return output.getvalue()


def run_manifest(run: dict) -> dict:
    """Record assumptions, seeds, results and a checksum without volatile timestamps."""
    global_payload = _serialized_result(run["global"])
    if run["global"].support_violation:
        # JSON has no infinity value. Keep the structural contradiction explicit.
        global_payload["statistic"] = None
        global_payload["statistic_note"] = "Infinite: positive count in a zero-probability category"
    return {
        "schema_version": "1.0",
        "analysis_version": ANALYSIS_VERSION,
        "data_provenance": "100% synthetic; created from a defined random mechanism",
        "generation": run["dataset"].metadata,
        "analysis": {
            "primary_question": "Global departure from the known assignment probabilities",
            "global": global_payload,
            "preselected_target": {"court": COURTS[0], **_serialized_result(run["target"])},
            "exploratory_courts": [
                {"court": court, **_serialized_result(result)}
                for court, result in zip(COURTS, run["exploratory"], strict=True)
            ],
            "interpretation": (
                "P-values are conditional tail probabilities under the stated mechanism. "
                "They are not probabilities of fraud, guilt, or a hypothesis being true. "
                "The global and planned target results do not define a combined decision rule."
            ),
        },
        "assignments_sha256": hashlib.sha256(assignments_csv(run).encode("utf-8")).hexdigest(),
        "software": {
            "court-assignment-audit": version("court-assignment-audit"),
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "pandas", "streamlit")},
        },
    }


def manifest_json(run: dict) -> str:
    return json.dumps(run_manifest(run), indent=2, sort_keys=True, allow_nan=False) + "\n"
