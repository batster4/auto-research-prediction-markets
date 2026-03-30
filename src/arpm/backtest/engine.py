"""Backtesting engine — simulation only; strategies and metrics stay separate."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from arpm.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    per_market_pnl: list[float]
    markets_traded: int

    @property
    def total_pnl(self) -> float:
        return float(sum(self.per_market_pnl))


def run_backtest(trades: pd.DataFrame, strategy: Strategy) -> BacktestResult:
    """Run one strategy across all markets in `trades` (canonical columns)."""
    if trades.empty:
        return BacktestResult(per_market_pnl=[], markets_traded=0)

    per_market: list[float] = []
    for _, group in trades.groupby("market_id", sort=False):
        pnl = float(strategy.run_market(group))
        per_market.append(pnl)

    return BacktestResult(per_market_pnl=per_market, markets_traded=len(per_market))
