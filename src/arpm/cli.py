"""CLI entry point for research experiments."""

from __future__ import annotations

import argparse
import sys

from arpm.agent.research_loop import run_research_experiment
from arpm.config import Settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arpm", description="Auto Research Prediction Markets")
    p.add_argument("task", help="Natural-language research task description")
    p.add_argument("--data", required=True, help="Path to CSV or Parquet trades dataset")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without Anthropic API (deterministic placeholder strategies)",
    )
    args = p.parse_args(argv)

    paths = run_research_experiment(args.task, args.data, Settings.from_env(), dry_run=args.dry_run)
    print(f"Experiment directory: {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
