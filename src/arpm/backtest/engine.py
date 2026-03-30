"""Backtesting engine — realistic simulation with resolution cutoff, fees,
and slippage.  Strategies never see the outcome column or the last
*resolution_cutoff_s* seconds of each market."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from arpm.backtest.fees import taker_fee_per_share
from arpm.strategies.base import Strategy


def _parse_resolution_unix(market_id: str) -> int | None:
    """Extract resolution unix timestamp from IDs like ``btc-updown-5m-1773573000``."""
    parts = str(market_id).rsplit("-", 1)
    if len(parts) == 2:
        try:
            return int(parts[-1])
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class MarketTradeResult:
    """Result for one market."""
    market_id: str
    outcome: float
    entered: bool
    entry_price: float | None = None
    entry_time: pd.Timestamp | None = None  # type: ignore[assignment]
    gross_pnl: float = 0.0
    fee: float = 0.0
    slippage_cost: float = 0.0
    pnl: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    """Aggregated backtest output — wraps per-market results."""
    market_results: list[MarketTradeResult]

    @property
    def per_market_pnl(self) -> list[float]:
        return [r.pnl for r in self.market_results]

    @property
    def total_pnl(self) -> float:
        return sum(r.pnl for r in self.market_results)

    @property
    def markets_traded(self) -> int:
        return sum(1 for r in self.market_results if r.entered)

    @property
    def total_capital_deployed(self) -> float:
        return sum(
            r.entry_price for r in self.market_results
            if r.entered and r.entry_price is not None
        )

    @property
    def total_fees(self) -> float:
        return sum(r.fee for r in self.market_results if r.entered)


def run_backtest(
    trades: pd.DataFrame,
    strategy: Strategy,
    *,
    resolution_cutoff_s: float = 60.0,
    slippage: float = 0.005,
    apply_fees: bool = True,
) -> BacktestResult:
    """Run *strategy* across all markets in *trades*.

    Realism knobs
    -------------
    resolution_cutoff_s : seconds before each market's last timestamp that
        are **removed** from the tradeable window (prevents near-resolution
        exploitation).
    slippage : fixed amount added to the signal price to model execution cost.
    apply_fees : apply Polymarket taker fee schedule per share.
    """
    if trades.empty:
        return BacktestResult(market_results=[])

    results: list[MarketTradeResult] = []

    for mid, group in trades.groupby("market_id", sort=False):
        sorted_rows = group.sort_values("timestamp")
        outcome = float(sorted_rows["outcome"].iloc[-1])
        last_ts = sorted_rows["timestamp"].iloc[-1]
        cutoff_ts = last_ts - pd.Timedelta(seconds=resolution_cutoff_s)

        tradeable = (
            sorted_rows
            .loc[sorted_rows["timestamp"] <= cutoff_ts, ["timestamp", "price_yes"]]
            .copy()
        )

        if tradeable.empty:
            results.append(MarketTradeResult(
                market_id=str(mid), outcome=outcome, entered=False,
            ))
            continue

        # Add time-to-expiry column (seconds until resolution).
        # BS-aware strategies use this; others just ignore it.
        # market_id timestamp is the WINDOW START; resolution = start + 300s.
        res_unix = _parse_resolution_unix(str(mid))
        if res_unix is not None:
            res_ts = pd.Timestamp(res_unix + 300, unit="s", tz="UTC")
        else:
            res_ts = last_ts
        tradeable["time_to_expiry_s"] = (
            (res_ts - tradeable["timestamp"]).dt.total_seconds()
        )

        if tradeable.empty:
            results.append(MarketTradeResult(
                market_id=str(mid), outcome=outcome, entered=False,
            ))
            continue

        signal = strategy.decide(tradeable)

        if signal is not None:
            effective_price = min(signal.price + slippage, 0.99)
            fee = taker_fee_per_share(effective_price) if apply_fees else 0.0
            gross = outcome - effective_price
            net = gross - fee

            results.append(MarketTradeResult(
                market_id=str(mid),
                outcome=outcome,
                entered=True,
                entry_price=effective_price,
                entry_time=signal.timestamp,
                gross_pnl=gross,
                fee=fee,
                slippage_cost=slippage,
                pnl=net,
            ))
        else:
            results.append(MarketTradeResult(
                market_id=str(mid), outcome=outcome, entered=False,
            ))

    return BacktestResult(market_results=results)
