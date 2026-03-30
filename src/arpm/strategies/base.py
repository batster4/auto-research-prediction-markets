"""Strategy protocol and JSON specs (decoupled from backtest engine).

Key design:  strategies receive only (timestamp, price_yes) — **never** the
outcome column — so lookahead is structurally impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, Field


# ── Signal returned by a strategy ────────────────────────────────────────

@dataclass(frozen=True)
class EntrySignal:
    """Buy-YES signal with the row price and timestamp at which to enter."""
    price: float
    timestamp: pd.Timestamp


# ── Strategy protocol ────────────────────────────────────────────────────

@runtime_checkable
class Strategy(Protocol):
    """Inspect a price series for one market and decide whether to buy YES."""

    name: str

    def decide(self, prices: pd.DataFrame) -> EntrySignal | None:
        """*prices* has columns ``[timestamp, price_yes]``, sorted by time.

        Return an :class:`EntrySignal` to buy, or ``None`` to pass.
        The DataFrame **never** contains the outcome column.
        """
        ...


# ── Strategy spec (LLM ↔ engine contract) ───────────────────────────────

STRATEGY_TYPES = Literal[
    "threshold",
    "momentum",
    "hold",
    "early_threshold",
    "mean_reversion",
    "relative_value",
    "ma_crossover",
    "bs_fair_value",
    "bs_overreaction",
    "gamma_scalp",
]


class StrategySpec(BaseModel):
    """Structured strategy description (from the LLM or configs)."""

    type: STRATEGY_TYPES  # type: ignore[valid-type]
    params: dict[str, Any] = Field(default_factory=dict)


# ── Spec → concrete strategy ─────────────────────────────────────────────

def strategy_from_spec(spec: StrategySpec) -> Strategy:
    """Instantiate a built-in strategy from a spec."""
    from arpm.strategies.builtin import (
        BSFairValueStrategy,
        BSOverreactionStrategy,
        EarlyThresholdStrategy,
        GammaScalpStrategy,
        HoldStrategy,
        MACrossoverStrategy,
        MeanReversionStrategy,
        MomentumStrategy,
        RelativeValueStrategy,
        ThresholdStrategy,
    )

    t = spec.type
    p = spec.params

    if t == "threshold":
        return ThresholdStrategy(buy_below=float(p.get("buy_below", 0.4)))
    if t == "hold":
        return HoldStrategy()
    if t == "momentum":
        raw = p.get("buy_if_rising", True)
        if isinstance(raw, bool):
            buy_if_rising = raw
        elif isinstance(raw, (int, float)):
            buy_if_rising = float(raw) >= 0.5
        else:
            buy_if_rising = str(raw).strip().lower() in ("1", "true", "yes")
        return MomentumStrategy(
            lookback=int(p.get("lookback", 3)),
            buy_if_rising=buy_if_rising,
        )
    if t == "early_threshold":
        return EarlyThresholdStrategy(
            buy_below=float(p.get("buy_below", 0.4)),
            entry_window_pct=float(p.get("entry_window_pct", 0.5)),
        )
    if t == "mean_reversion":
        return MeanReversionStrategy(
            drop_pct=float(p.get("drop_pct", 0.10)),
            lookback=int(p.get("lookback", 5)),
        )
    if t == "relative_value":
        return RelativeValueStrategy(
            fair_value=float(p.get("fair_value", 0.50)),
            edge_required=float(p.get("edge_required", 0.05)),
        )
    if t == "ma_crossover":
        return MACrossoverStrategy(
            window=int(p.get("window", 10)),
            discount=float(p.get("discount", 0.05)),
        )
    if t == "bs_fair_value":
        return BSFairValueStrategy(
            vol_annual=float(p.get("vol_annual", 0.80)),
            edge_required=float(p.get("edge_required", 0.05)),
            min_tte_pct=float(p.get("min_tte_pct", 0.20)),
            max_tte_pct=float(p.get("max_tte_pct", 0.80)),
            warmup_ticks=int(p.get("warmup_ticks", 5)),
        )
    if t == "bs_overreaction":
        return BSOverreactionStrategy(
            vol_annual=float(p.get("vol_annual", 0.80)),
            z_threshold=float(p.get("z_threshold", 2.0)),
            vol_window=int(p.get("vol_window", 10)),
        )
    if t == "gamma_scalp":
        return GammaScalpStrategy(
            atm_band=float(p.get("atm_band", 0.15)),
            dip_threshold=float(p.get("dip_threshold", 0.05)),
            max_tte_pct=float(p.get("max_tte_pct", 0.75)),
            min_tte_pct=float(p.get("min_tte_pct", 0.15)),
            lookback=int(p.get("lookback", 5)),
        )
    raise ValueError(f"Unknown strategy type: {t}")
