"""Built-in strategies — each inspects (timestamp, price_yes) and optionally
time_to_expiry_s, then returns EntrySignal or None.
Strategies never see the outcome column."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm  # type: ignore[import-untyped]

from arpm.strategies.base import EntrySignal

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


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


# ══════════════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES / BINARY OPTION STRATEGIES
#  These use the ``time_to_expiry_s`` column added by the engine.
# ══════════════════════════════════════════════════════════════════════════

def _safe_ppf(p: float) -> float:
    """Inverse-normal CDF, clamped to avoid ±inf."""
    return float(_norm.ppf(np.clip(p, 0.005, 0.995)))


# ── BS fair-value ─────────────────────────────────────────────────────────

@dataclass
class BSFairValueStrategy:
    """Estimate fair value of a binary option via Black-Scholes and buy when
    the market is cheap.

    Uses the opening price_yes to infer initial moneyness (ln S/K), then
    re-prices with the BS binary-call formula at each tick using estimated
    vol and remaining time.  Buys when market price < fair − edge.
    """

    vol_annual: float = 0.80
    edge_required: float = 0.05
    min_tte_pct: float = 0.20
    max_tte_pct: float = 0.80
    warmup_ticks: int = 5
    name: str = "bs_fair_value"

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        if "time_to_expiry_s" not in prices.columns:
            return None
        vals = prices["price_yes"].astype(float).values
        ts = prices["timestamp"].values
        tte = prices["time_to_expiry_s"].astype(float).values
        n = len(vals)
        if n < self.warmup_ticks + 1:
            return None

        total_window = float(tte[0]) + 60.0  # approx full window (first TTE + cutoff)
        if total_window < 1.0:
            return None

        p_open = float(np.mean(vals[: min(3, n)]))
        if p_open <= 0.01 or p_open >= 0.99:
            return None
        d2_open = _safe_ppf(p_open)
        tau_open = float(tte[0]) / _SECONDS_PER_YEAR
        if tau_open < 1e-12:
            return None
        sigma = self.vol_annual
        sqrt_tau_open = np.sqrt(tau_open)
        moneyness = d2_open * sigma * sqrt_tau_open + 0.5 * sigma ** 2 * tau_open

        for i in range(self.warmup_ticks, n):
            p = vals[i]
            tau_s = tte[i]
            tau_pct = tau_s / total_window
            if tau_pct < self.min_tte_pct or tau_pct > self.max_tte_pct:
                continue
            if p <= 0.01 or p >= 0.99:
                continue
            tau_y = tau_s / _SECONDS_PER_YEAR
            if tau_y < 1e-12:
                continue
            sqrt_tau = np.sqrt(tau_y)
            d2_now = (moneyness - 0.5 * sigma ** 2 * tau_y) / (sigma * sqrt_tau)
            fair = float(_norm.cdf(d2_now))
            if p < fair - self.edge_required:
                return EntrySignal(price=float(p), timestamp=pd.Timestamp(ts[i]))
        return None


# ── BS overreaction ───────────────────────────────────────────────────────

@dataclass
class BSOverreactionStrategy:
    """Buy when a price drop exceeds the BS-predicted standard deviation of
    moves over the elapsed time.

    Computes expected per-tick volatility of the binary option price from
    the BS model, then triggers when the actual drop is ≥ z_threshold σ.
    """

    vol_annual: float = 0.80
    z_threshold: float = 2.0
    vol_window: int = 10
    name: str = "bs_overreaction"

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        if "time_to_expiry_s" not in prices.columns:
            return None
        vals = prices["price_yes"].astype(float).values
        ts = prices["timestamp"].values
        tte = prices["time_to_expiry_s"].astype(float).values
        n = len(vals)
        if n < self.vol_window + 2:
            return None

        for i in range(self.vol_window + 1, n):
            p = vals[i]
            tau_s = tte[i]
            tau_y = tau_s / _SECONDS_PER_YEAR
            if tau_y < 1e-12 or p <= 0.01 or p >= 0.99:
                continue

            window = vals[max(0, i - self.vol_window): i + 1]
            diffs = np.diff(window)
            if len(diffs) < 2:
                continue
            realized_std = float(np.std(diffs, ddof=1))
            if realized_std < 1e-9:
                continue

            rolling_high = float(np.max(vals[max(0, i - self.vol_window): i]))
            drop = rolling_high - p
            if drop > self.z_threshold * realized_std and drop > 0.02:
                return EntrySignal(price=float(p), timestamp=pd.Timestamp(ts[i]))
        return None


# ── Gamma scalp ───────────────────────────────────────────────────────────

@dataclass
class GammaScalpStrategy:
    """Buy near-ATM binary options after a dip, exploiting high gamma.

    Binary option gamma peaks when the option is near ATM (price ≈ 0.5).
    High gamma means small underlying moves cause large option price swings
    → the market is likely to overreact.  Buy the dip in high-gamma regimes.
    """

    atm_band: float = 0.15
    dip_threshold: float = 0.05
    max_tte_pct: float = 0.75
    min_tte_pct: float = 0.15
    lookback: int = 5
    name: str = "gamma_scalp"

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        if "time_to_expiry_s" not in prices.columns:
            return None
        vals = prices["price_yes"].astype(float).values
        ts = prices["timestamp"].values
        tte = prices["time_to_expiry_s"].astype(float).values
        n = len(vals)
        if n < self.lookback + 2:
            return None

        total_window = float(tte[0]) + 60.0
        if total_window < 1.0:
            return None

        for i in range(self.lookback + 1, n):
            p = vals[i]
            tau_pct = tte[i] / total_window
            if tau_pct < self.min_tte_pct or tau_pct > self.max_tte_pct:
                continue
            if abs(p - 0.5) > self.atm_band:
                continue
            window_high = float(np.max(vals[max(0, i - self.lookback): i]))
            drop = window_high - p
            if drop >= self.dip_threshold:
                return EntrySignal(price=float(p), timestamp=pd.Timestamp(ts[i]))
        return None
