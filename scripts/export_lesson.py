"""Export the authored lesson and optional reproducible simulation metadata."""

import argparse
import json
from pathlib import Path

from court_assignment_audit.lesson import lesson_csv, lesson_manifest, simulate_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the two exports.")
    parser.add_argument("--batch", type=int, help="Optional positive simulation batch number.")
    parser.add_argument("--practice-threshold", type=int, help="Optional hypothetical count, 0-60.")
    args = parser.parse_args(argv)
    try:
        batch = simulate_batch(args.batch) if args.batch is not None else None
        manifest = lesson_manifest(batch=batch, practice_threshold=args.practice_threshold)
    except ValueError as error:
        parser.error(str(error))
    # Validate all arguments before creating output or writing either artifact.
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "lesson.csv").write_bytes(lesson_csv().encode("utf-8"))
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    (args.output / "lesson_manifest.json").write_bytes(manifest_bytes)
    print(f"Exported lesson.csv and lesson_manifest.json to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
