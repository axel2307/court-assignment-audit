"""Synthetic court-assignment audit demonstration."""

from .synthetic import (
    COURTS,
    DEFAULT_ANOMALY_PROBABILITIES,
    DEFAULT_NULL_PROBABILITIES,
    GENERATOR_VERSION,
    MAX_CASES,
    PRESELECTED_TARGET,
    Assignment,
    SimulationConfig,
    SyntheticDataset,
    generate_dataset,
)

__all__ = [
    "Assignment",
    "COURTS",
    "DEFAULT_ANOMALY_PROBABILITIES",
    "DEFAULT_NULL_PROBABILITIES",
    "GENERATOR_VERSION",
    "MAX_CASES",
    "PRESELECTED_TARGET",
    "SimulationConfig",
    "SyntheticDataset",
    "generate_dataset",
]
