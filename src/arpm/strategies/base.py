"""Strategy protocol and JSON specs (decoupled from backtest engine)."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, Field


class StrategySpec(BaseModel):
    """Structured strategy description (e.g. from the model or configs)."""

    type: Literal["threshold", "momentum", "hold"]
    params: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    """Per-market strategy: given time-ordered rows for one market, return realized PnL."""

    name: str

    def run_market(self, market_rows: pd.DataFrame) -> float:
        """Return PnL in dollars for one unit of notional (1 share) unless scaled in params."""
        ...


def strategy_from_spec(spec: StrategySpec) -> Strategy:
    """Instantiate a built-in strategy from a spec."""
    from arpm.strategies.builtin import HoldStrategy, MomentumStrategy, ThresholdStrategy

    if spec.type == "threshold":
        return ThresholdStrategy(buy_below=float(spec.params.get("buy_below", 0.4)))
    if spec.type == "hold":
        return HoldStrategy()
    if spec.type == "momentum":
        raw = spec.params.get("buy_if_rising", 1.0)
        if isinstance(raw, bool):
            buy_if_rising = raw
        elif isinstance(raw, (int, float)):
            buy_if_rising = float(raw) >= 0.5
        else:
            buy_if_rising = str(raw).strip().lower() in ("1", "true", "yes")
        return MomentumStrategy(
            lookback=int(spec.params.get("lookback", 3)),
            buy_if_rising=buy_if_rising,
        )
    raise ValueError(f"Unknown strategy type: {spec.type}")
