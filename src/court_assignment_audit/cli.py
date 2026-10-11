"""Generate a local, entirely synthetic CSV and its reproducibility manifest."""

import argparse
from pathlib import Path

from court_assignment_audit.reporting import assignments_csv, build_run, manifest_json
from court_assignment_audit.synthetic import SimulationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("null", "anomaly"), default="null")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--n-cases", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    args = parser.parse_args()
    try:
        run = build_run(
            SimulationConfig(n_cases=args.n_cases, seed=args.seed, scenario=args.scenario)
        )
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "synthetic_assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        handle.write(assignments_csv(run))
    (args.output_dir / "run_manifest.json").write_text(manifest_json(run), encoding="utf-8")
    print(f"Synthetic demo exported to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
