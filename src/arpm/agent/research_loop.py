"""Iterative research loop: bounded time per iteration, bounded iterations per experiment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from arpm.agent.claude_client import ClaudeResearchClient
from arpm.backtest.engine import run_backtest
from arpm.config import Settings
from arpm.data.loaders import load_trades_table
from arpm.evaluation.metrics import evaluate_returns
from arpm.experiments.store import ExperimentPaths, append_iteration, create_experiment
from arpm.strategies.base import StrategySpec, strategy_from_spec


def _dry_run_specs(iteration: int) -> list[StrategySpec]:
    """Deterministic placeholder strategies when no API key is available."""
    base = [
        StrategySpec(type="threshold", params={"buy_below": 0.45 - 0.01 * (iteration % 5)}),
        StrategySpec(type="momentum", params={"lookback": 2 + (iteration % 3), "buy_if_rising": 1.0}),
    ]
    return base


def run_research_experiment(
    task: str,
    dataset_path: str | Path,
    settings: Settings | None = None,
    dry_run: bool = False,
) -> ExperimentPaths:
    """
    Load data, run up to `max_iterations` research iterations.
    Each iteration has a wall-clock budget of `max_seconds_per_iteration` seconds.
    """
    settings = settings or Settings.from_env()
    dataset_path = Path(dataset_path)
    trades = load_trades_table(dataset_path)
    paths = create_experiment(task, dataset_path, experiments_dir=settings.experiments_dir)

    client: ClaudeResearchClient | None = None
    if not dry_run:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Use dry_run=True for offline testing.")
        client = ClaudeResearchClient(api_key=settings.anthropic_api_key, model=settings.model)

    prior: list[dict[str, Any]] = []

    for iteration in range(1, settings.max_iterations + 1):
        iter_start = time.monotonic()
        deadline = iter_start + settings.max_seconds_per_iteration

        if dry_run:
            specs = _dry_run_specs(iteration)
        else:
            assert client is not None
            specs = client.propose_strategies(task, prior)

        iteration_records: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for spec in specs:
            if time.monotonic() > deadline:
                break
            strat = strategy_from_spec(spec)
            bt = run_backtest(trades, strat)
            ev = evaluate_returns(bt.per_market_pnl)
            rec = {
                "strategy": spec.model_dump(),
                "total_pnl": ev.total_pnl,
                "mean_pnl_per_market": ev.mean_pnl_per_market,
                "sharpe_per_market": ev.sharpe_per_market,
                "max_drawdown": ev.max_drawdown,
                "hit_rate": ev.hit_rate,
                "n_markets": ev.n_markets,
            }
            iteration_records.append(rec)
            if best is None or rec["total_pnl"] > best["total_pnl"]:
                best = rec

        elapsed = time.monotonic() - iter_start

        out: dict[str, Any] = {
            "iteration": iteration,
            "elapsed_seconds": round(elapsed, 3),
            "within_time_budget": elapsed <= settings.max_seconds_per_iteration,
            "candidates": iteration_records,
            "best_in_iteration": best,
        }
        prior.append(out)
        append_iteration(paths, out)

        if elapsed > settings.max_seconds_per_iteration:
            break

    return paths
