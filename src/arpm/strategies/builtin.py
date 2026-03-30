"""Built-in strategies — each inspects (timestamp, price_yes) and returns an
EntrySignal or None.  Strategies never see the outcome column."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arpm.strategies.base import EntrySignal


# ── Threshold ─────────────────────────────────────────────────────────────

@dataclass
class ThresholdStrategy:
    """Buy YES the first time ``price_yes <= buy_below``."""

    buy_below: float = 0.40
    name: str = "threshold"

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        for _, row in prices.iterrows():
            if float(row["price_yes"]) <= self.buy_below:
                return EntrySignal(price=float(row["price_yes"]),
                                   timestamp=row["timestamp"])
        return None


# ── Hold (baseline) ──────────────────────────────────────────────────────

@dataclass
class HoldStrategy:
    """Never trade — flat PnL baseline."""

    name: str = "hold"

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        return None


# ── Momentum ──────────────────────────────────────────────────────────────

@dataclass
class MomentumStrategy:
    """Buy when *lookback*-tick momentum direction matches *buy_if_rising*."""

    lookback: int = 3
    buy_if_rising: bool = True
    name: str = "momentum"

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        vals = prices["price_yes"].astype(float).values
        ts = prices["timestamp"].values
        n = len(vals)
        if n < self.lookback + 1:
            return None
        for i in range(self.lookback, n):
            rising = vals[i] > vals[i - self.lookback]
            if rising == self.buy_if_rising:
                return EntrySignal(price=float(vals[i]),
                                   timestamp=pd.Timestamp(ts[i]))
        return None


# ── Early threshold ───────────────────────────────────────────────────────

@dataclass
class EarlyThresholdStrategy:
    """Threshold restricted to the first *entry_window_pct* of tradeable rows."""

    buy_below: float = 0.40
    entry_window_pct: float = 0.50
    name: str = "early_threshold"

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        n = len(prices)
        cutoff = max(1, int(n * self.entry_window_pct))
        for _, row in prices.iloc[:cutoff].iterrows():
            if float(row["price_yes"]) <= self.buy_below:
                return EntrySignal(price=float(row["price_yes"]),
                                   timestamp=row["timestamp"])
        return None


# ── Mean reversion ────────────────────────────────────────────────────────

@dataclass
class MeanReversionStrategy:
    """Buy after a drop of at least *drop_pct* from the rolling high."""

    drop_pct: float = 0.10
    lookback: int = 5
    name: str = "mean_reversion"

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        vals = prices["price_yes"].astype(float).values
        ts = prices["timestamp"].values
        n = len(vals)
        if n < self.lookback + 1:
            return None
        for i in range(self.lookback, n):
            window_high = float(np.max(vals[max(0, i - self.lookback):i]))
            cur = float(vals[i])
            if window_high > 1e-9 and (window_high - cur) / window_high >= self.drop_pct:
                return EntrySignal(price=cur, timestamp=pd.Timestamp(ts[i]))
        return None


# ── Relative value ────────────────────────────────────────────────────────

@dataclass
class RelativeValueStrategy:
    """Buy when price implies probability below *fair_value − edge_required*."""

    fair_value: float = 0.50
    edge_required: float = 0.05
    name: str = "relative_value"

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        target = self.fair_value - self.edge_required
        for _, row in prices.iterrows():
            if float(row["price_yes"]) <= target:
                return EntrySignal(price=float(row["price_yes"]),
                                   timestamp=row["timestamp"])
        return None


# ── Moving-average crossover ──────────────────────────────────────────────

@dataclass
class MACrossoverStrategy:
    """Buy when price drops below its *window*-tick MA by *discount*."""

    window: int = 10
    discount: float = 0.05
    name: str = "ma_crossover"

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        vals = prices["price_yes"].astype(float).values
        ts = prices["timestamp"].values
        n = len(vals)
        if n < self.window + 1:
            return None
        for i in range(self.window, n):
            ma = float(np.mean(vals[i - self.window:i]))
            cur = float(vals[i])
            if cur < ma - self.discount:
                return EntrySignal(price=cur, timestamp=pd.Timestamp(ts[i]))
        return None
