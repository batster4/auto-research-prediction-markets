"""Reference strategies — simple, testable templates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ThresholdStrategy:
    """Buy YES the first time price_yes <= buy_below; hold until resolution."""

    buy_below: float = 0.4
    name: str = "threshold"

    def run_market(self, market_rows: pd.DataFrame) -> float:
        df = market_rows.sort_values("timestamp")
        if df.empty:
            return 0.0
        outcome = float(df["outcome"].iloc[-1])
        for _, row in df.iterrows():
            p = float(row["price_yes"])
            if p <= self.buy_below:
                return float(outcome - p)
        return 0.0


@dataclass
class HoldStrategy:
    """No trade — flat PnL per market."""

    name: str = "hold"

    def run_market(self, market_rows: pd.DataFrame) -> float:
        return 0.0


@dataclass
class MomentumStrategy:
    """If short-term momentum of price_yes is rising (or falling), buy YES once."""

    lookback: int = 3
    buy_if_rising: bool = True
    name: str = "momentum"

    def run_market(self, market_rows: pd.DataFrame) -> float:
        df = market_rows.sort_values("timestamp").reset_index(drop=True)
        if len(df) < self.lookback + 1:
            return 0.0
        outcome = float(df["outcome"].iloc[-1])
        prices = df["price_yes"].astype(float)
        for i in range(self.lookback, len(df)):
            prev = float(prices.iloc[i - 1])
            cur = float(prices.iloc[i])
            rising = cur > prev
            if rising == self.buy_if_rising:
                return float(outcome - cur)
        return 0.0
