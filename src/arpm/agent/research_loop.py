"""Iterative research loop: bounded time per iteration, bounded iterations per experiment.

Key improvements over the original:
- Temporal train / test split (LLM sees train metrics only; test metrics stored for analysis)
- Resolution cutoff, fees, slippage via the backtest engine
- Richer per-iteration metrics (n_entered, entry_rate, profit_factor, ROC …)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from arpm.agent.claude_client import ClaudeResearchClient
from arpm.backtest.engine import run_backtest
from arpm.config import Settings
from arpm.data.loaders import load_trades_table
from arpm.data.splits import temporal_split
from arpm.evaluation.metrics import evaluate_backtest
from arpm.experiments.store import (
    ExperimentPaths,
    append_iteration,
    create_experiment,
    load_prior_iterations,
    open_existing_experiment,
)
from arpm.strategies.base import StrategySpec, strategy_from_spec

RESOLUTION_CUTOFF_S = 60.0
SLIPPAGE = 0.005
TRAIN_PCT = 0.70


def _dry_run_specs(iteration: int) -> list[StrategySpec]:
    """Deterministic placeholder strategies when no API key is available."""
    return [
        StrategySpec(type="threshold", params={"buy_below": 0.45 - 0.01 * (iteration % 5)}),
        StrategySpec(type="momentum", params={"lookback": 2 + (iteration % 3), "buy_if_rising": True}),
        StrategySpec(type="early_threshold", params={"buy_below": 0.40, "entry_window_pct": 0.5}),
    ]


def _ev_to_record(spec: StrategySpec, ev: Any) -> dict[str, Any]:
    """Flatten an EvaluationSummary into a JSON-safe dict."""
    return {
        "strategy": spec.model_dump(),
        "total_pnl": round(ev.total_pnl, 6),
        "mean_pnl_per_market": round(ev.mean_pnl_per_market, 6),
        "std_pnl_per_market": round(ev.std_pnl_per_market, 6),
        "sharpe_per_market": round(ev.sharpe_per_market, 4) if ev.sharpe_per_market is not None else None,
        "max_drawdown": round(ev.max_drawdown, 6),
        "hit_rate": round(ev.hit_rate, 4),
        "n_markets": ev.n_markets,
        "n_entered": ev.n_entered,
        "entry_rate": round(ev.entry_rate, 4),
        "profit_factor": round(ev.profit_factor, 4) if ev.profit_factor is not None else None,
        "avg_win": round(ev.avg_win, 6),
        "avg_loss": round(ev.avg_loss, 6),
        "expectancy": round(ev.expectancy, 6),
        "return_on_capital": round(ev.return_on_capital, 4) if ev.return_on_capital is not None else None,
        "total_fees": round(ev.total_fees, 6),
    }


def _strip_test(prior_entry: dict[str, Any]) -> dict[str, Any]:
    """Remove test_metrics from a prior record (for LLM consumption)."""
    out = {k: v for k, v in prior_entry.items() if k != "test_metrics"}
    return out


def run_research_experiment(
    task: str,
    dataset_path: str | Path | None,
    settings: Settings | None = None,
    dry_run: bool = False,
    *,
    resume_from: str | Path | None = None,
) -> ExperimentPaths:
    settings = settings or Settings.from_env()
    prior_full: list[dict[str, Any]] = []

    if resume_from is not None:
        paths, task, ds = open_existing_experiment(resume_from)
        dataset_path = Path(ds).resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset from manifest not found: {dataset_path}")
        trades = load_trades_table(dataset_path)
        prior_full = load_prior_iterations(paths)
        print(f"Resuming experiment: {paths.root}", flush=True)
        print(f"Loaded {len(prior_full)} prior iterations; continuing to {settings.max_iterations}.", flush=True)
    else:
        if not task or not task.strip():
            raise ValueError("task is required when not resuming")
        if dataset_path is None:
            raise ValueError("dataset_path is required when not resuming")
        dataset_path = Path(dataset_path)
        trades = load_trades_table(dataset_path)
        paths = create_experiment(task, dataset_path, experiments_dir=settings.experiments_dir)
        print(f"Experiment directory: {paths.root}", flush=True)

    # ── temporal split ────────────────────────────────────────────────
    train_data, test_data = temporal_split(trades, train_pct=TRAIN_PCT)
    n_train_markets = train_data["market_id"].nunique()
    n_test_markets = test_data["market_id"].nunique()
    print(
        f"Data split: {n_train_markets} train markets, {n_test_markets} test markets "
        f"(cutoff={RESOLUTION_CUTOFF_S}s, slippage={SLIPPAGE}, fees=on)",
        flush=True,
    )
    print(f"Max iterations: {settings.max_iterations}, budget per iteration: {settings.max_seconds_per_iteration}s", flush=True)

    client: ClaudeResearchClient | None = None
    if not dry_run:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Use dry_run=True for offline testing.")
        client = ClaudeResearchClient(
            api_key=settings.anthropic_api_key,
            model=settings.model,
            thinking_budget_tokens=settings.thinking_budget_tokens,
            max_output_tokens=settings.max_output_tokens,
            web_search_enabled=settings.web_search_enabled,
            web_search_max_uses=settings.web_search_max_uses,
        )

    # LLM sees only train metrics (strip test_metrics from prior)
    prior_for_llm: list[dict[str, Any]] = [_strip_test(p) for p in prior_full]

    start_iter = len(prior_full) + 1
    if start_iter > settings.max_iterations:
        print(f"Nothing to do: already have {len(prior_full)} iterations (max {settings.max_iterations}).", flush=True)
        return paths

    for iteration in range(start_iter, settings.max_iterations + 1):
        iter_start = time.monotonic()
        deadline = iter_start + settings.max_seconds_per_iteration
        print(f"\nIteration {iteration}/{settings.max_iterations} …", flush=True)

        if dry_run:
            specs = _dry_run_specs(iteration)
        else:
            assert client is not None
            line = os.environ.get("ARPM_RESEARCH_LINE", "").strip() or paths.root.parent.name
            specs = client.propose_strategies(
                task,
                prior_for_llm,
                experiment_root=str(paths.root.resolve()),
                research_line_label=line,
            )

        # ── evaluate candidates on TRAIN set ──────────────────────────
        iteration_records: list[dict[str, Any]] = []
        best_rec: dict[str, Any] | None = None

        for spec in specs:
            if time.monotonic() > deadline:
                break
            try:
                strat = strategy_from_spec(spec)
            except (ValueError, KeyError, TypeError) as exc:
                print(f"  skip invalid spec {spec}: {exc}", flush=True)
                continue

            bt = run_backtest(
                train_data, strat,
                resolution_cutoff_s=RESOLUTION_CUTOFF_S,
                slippage=SLIPPAGE,
                apply_fees=True,
            )
            ev = evaluate_backtest(bt)
            rec = _ev_to_record(spec, ev)
            iteration_records.append(rec)

            if best_rec is None or rec["total_pnl"] > best_rec["total_pnl"]:
                best_rec = rec

        # ── evaluate best candidate on TEST set ───────────────────────
        test_metrics: dict[str, Any] | None = None
        if best_rec is not None and not test_data.empty:
            best_spec = StrategySpec.model_validate(best_rec["strategy"])
            best_strat = strategy_from_spec(best_spec)
            bt_test = run_backtest(
                test_data, best_strat,
                resolution_cutoff_s=RESOLUTION_CUTOFF_S,
                slippage=SLIPPAGE,
                apply_fees=True,
            )
            ev_test = evaluate_backtest(bt_test)
            test_metrics = _ev_to_record(best_spec, ev_test)
            del test_metrics["strategy"]

        elapsed = time.monotonic() - iter_start

        # ── record visible to LLM (train only) ───────────────────────
        llm_record: dict[str, Any] = {
            "iteration": iteration,
            "elapsed_seconds": round(elapsed, 3),
            "within_time_budget": elapsed <= settings.max_seconds_per_iteration,
            "candidates": iteration_records,
            "best_in_iteration": best_rec,
        }
        prior_for_llm.append(llm_record)

        # ── full record saved to disk (train + test) ──────────────────
        full_record = {**llm_record, "test_metrics": test_metrics}
        prior_full.append(full_record)
        append_iteration(paths, full_record)

        best_pnl = best_rec["total_pnl"] if best_rec else "n/a"
        test_pnl = test_metrics["total_pnl"] if test_metrics else "n/a"
        print(
            f"Iteration {iteration} done: {len(iteration_records)} candidates | "
            f"train best PnL={best_pnl} | test PnL={test_pnl}",
            flush=True,
        )

        if elapsed > settings.max_seconds_per_iteration:
            break

    print(f"\nFinished. Experiment directory: {paths.root}", flush=True)
    return paths
