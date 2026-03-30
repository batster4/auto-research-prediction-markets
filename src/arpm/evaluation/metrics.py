"""Evaluation metrics — clearly defined, reusable across experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from arpm.backtest.engine import BacktestResult


@dataclass(frozen=True)
class EvaluationSummary:
    total_pnl: float
    mean_pnl_per_market: float
    std_pnl_per_market: float
    sharpe_per_market: float | None
    max_drawdown: float
    hit_rate: float
    n_markets: int
    n_entered: int
    entry_rate: float
    profit_factor: float | None
    avg_win: float
    avg_loss: float
    expectancy: float
    total_capital_deployed: float
    return_on_capital: float | None
    total_fees: float


# ── helpers ───────────────────────────────────────────────────────────────

def _equity_curve(period_returns: list[float]) -> np.ndarray:
    if not period_returns:
        return np.array([0.0])
    return np.cumsum(np.asarray(period_returns, dtype=float))


def max_drawdown_from_pnl(per_period: list[float]) -> float:
    """Max drawdown on cumulative PnL curve."""
    if not per_period:
        return 0.0
    equity = _equity_curve(per_period)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)) if len(dd) else 0.0


def sharpe_like(per_market: list[float]) -> float | None:
    """Mean / std of per-market PnL (no annualization)."""
    if len(per_market) < 2:
        return None
    arr = np.asarray(per_market, dtype=float)
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return None
    return float(np.mean(arr) / std)


def hit_rate(per_market: list[float]) -> float:
    if not per_market:
        return 0.0
    return sum(1 for x in per_market if x > 0) / len(per_market)


# ── main evaluation ──────────────────────────────────────────────────────

def evaluate_backtest(result: BacktestResult) -> EvaluationSummary:
    """Compute comprehensive metrics from a :class:`BacktestResult`."""
    all_pnl = result.per_market_pnl
    n = len(all_pnl)
    total = result.total_pnl

    entered = [r for r in result.market_results if r.entered]
    n_entered = len(entered)
    entry_rate_ = n_entered / n if n else 0.0

    # Time-sorted equity curve (sort entered trades by entry time)
    entered_sorted = sorted(entered, key=lambda r: r.entry_time or pd.Timestamp.min)
    time_pnl = [r.pnl for r in entered_sorted]
    dd = max_drawdown_from_pnl(time_pnl)

    entered_pnl = [r.pnl for r in entered]
    wins = [p for p in entered_pnl if p > 0]
    losses = [p for p in entered_pnl if p < 0]

    sum_wins = sum(wins)
    sum_losses_abs = abs(sum(losses))
    pf = sum_wins / sum_losses_abs if sum_losses_abs > 1e-12 else None

    aw = sum_wins / len(wins) if wins else 0.0
    al = sum_losses_abs / len(losses) if losses else 0.0

    hr_entered = len(wins) / n_entered if n_entered else 0.0

    capital = result.total_capital_deployed
    roc = total / capital if capital > 1e-12 else None

    exp = hr_entered * aw - (1 - hr_entered) * al

    mean = total / n if n else 0.0
    std = float(np.std(np.asarray(all_pnl, dtype=float), ddof=1)) if n > 1 else 0.0
    sh = sharpe_like(all_pnl)

    return EvaluationSummary(
        total_pnl=total,
        mean_pnl_per_market=mean,
        std_pnl_per_market=std,
        sharpe_per_market=sh if sh is not None and not math.isnan(sh) else None,
        max_drawdown=dd,
        hit_rate=hr_entered,
        n_markets=n,
        n_entered=n_entered,
        entry_rate=entry_rate_,
        profit_factor=pf,
        avg_win=aw,
        avg_loss=al,
        expectancy=exp,
        total_capital_deployed=capital,
        return_on_capital=roc,
        total_fees=result.total_fees,
    )


def evaluate_returns(per_market_pnl: list[float]) -> EvaluationSummary:
    """Legacy wrapper: build a minimal BacktestResult and evaluate."""
    from arpm.backtest.engine import MarketTradeResult

    mrs = [
        MarketTradeResult(
            market_id=str(i),
            outcome=0.0,
            entered=pnl != 0.0,
            entry_price=None,
            pnl=pnl,
        )
        for i, pnl in enumerate(per_market_pnl)
    ]
    return evaluate_backtest(BacktestResult(market_results=mrs))
