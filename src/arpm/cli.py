"""CLI entry point for research experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from arpm.agent.research_loop import run_research_experiment
from arpm.config import Settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arpm", description="Auto Research Prediction Markets")
    p.add_argument("task", nargs="?", default="", help="Natural-language research task description")
    p.add_argument(
        "--task-file",
        type=str,
        default=None,
        help="Read task from file (UTF-8). If set, positional task is ignored.",
    )
    p.add_argument("--data", default=None, help="Path to CSV or Parquet trades dataset (not needed with --resume)")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="DIR",
        help="Resume from an existing experiment directory (uses manifest task + dataset)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without Anthropic API (deterministic placeholder strategies)",
    )
    args = p.parse_args(argv)

    task = args.task.strip()
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8").strip()
    if args.resume:
        if task:
            p.error("Do not pass a task when using --resume (loaded from manifest)")
        if args.data:
            p.error("Do not pass --data when using --resume (loaded from manifest)")
        paths = run_research_experiment(
            "",
            None,
            Settings.from_env(),
            dry_run=args.dry_run,
            resume_from=args.resume,
        )
    else:
        if not task:
            p.error("Provide a task string or --task-file")
        if not args.data:
            p.error("--data is required unless --resume")
        paths = run_research_experiment(task, args.data, Settings.from_env(), dry_run=args.dry_run)
    print(f"Experiment directory: {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
