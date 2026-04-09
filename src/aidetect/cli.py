"""Command-line entry point."""

from __future__ import annotations

import argparse
import json

from .paraphrase import write_paraphrase_dataset
from .pipeline import run_full_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-generated text detection experiments")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Run the full Week 4-8 experiment")
    subparsers.add_parser("generate-paraphrases", help="Generate the paraphrased dataset from raw AI text")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-paraphrases":
        frame = write_paraphrase_dataset()
        print(json.dumps({"rows_written": len(frame)}, indent=2))
        return

    metrics = run_full_experiment()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
