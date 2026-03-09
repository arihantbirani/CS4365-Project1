"""Command-line entry point."""

from __future__ import annotations

import json

from .pipeline import run_baseline_experiment


def main() -> None:
    metrics = run_baseline_experiment()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

