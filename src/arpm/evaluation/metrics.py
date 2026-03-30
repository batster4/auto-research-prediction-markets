"""Evaluation metrics — clearly defined, reusable across experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvaluationSummary:
    total_pnl: float
    mean_pnl_per_market: float
    std_pnl_per_market: float
    sharpe_per_market: float | None
    max_drawdown: float
    hit_rate: float
    n_markets: int


def _equity_curve(period_returns: list[float]) -> np.ndarray:
    if not period_returns:
        return np.array([0.0])
    s = np.cumsum(np.array(period_returns, dtype=float))
    return s


def max_drawdown_from_pnl(per_period: list[float]) -> float:
    """Max drawdown on cumulative PnL curve (not dollar-weighted portfolio)."""
    if not per_period:
        return 0.0
    equity = _equity_curve(per_period)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)) if len(dd) else 0.0


def sharpe_like(per_market: list[float]) -> float | None:
    """Simple Sharpe-like ratio: mean/std of per-market PnL (annualization not applied)."""
    if len(per_market) < 2:
        return None
    arr = np.array(per_market, dtype=float)
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return None
    return float(np.mean(arr) / std)


def hit_rate(per_market: list[float]) -> float:
    if not per_market:
        return 0.0
    wins = sum(1 for x in per_market if x > 0)
    return wins / len(per_market)


def evaluate_returns(per_market_pnl: list[float]) -> EvaluationSummary:
    n = len(per_market_pnl)
    total = float(sum(per_market_pnl))
    mean = total / n if n else 0.0
    std = float(np.std(np.array(per_market_pnl, dtype=float), ddof=1)) if n > 1 else 0.0
    sharpe = sharpe_like(per_market_pnl)
    dd = max_drawdown_from_pnl(per_market_pnl)
    hr = hit_rate(per_market_pnl)
    return EvaluationSummary(
        total_pnl=total,
        mean_pnl_per_market=mean,
        std_pnl_per_market=std,
        sharpe_per_market=sharpe if sharpe is not None and not math.isnan(sharpe) else None,
        max_drawdown=dd,
        hit_rate=hr,
        n_markets=n,
    )
