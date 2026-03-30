"""Temporal train / test splitting for walk-forward backtesting."""

from __future__ import annotations

import pandas as pd


def temporal_split(
    trades: pd.DataFrame,
    train_pct: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *trades* into train / test by each market's start time.

    Markets whose first timestamp falls in the earliest *train_pct* fraction
    go to train; the rest go to test.  This avoids any lookahead bias
    (test markets are strictly later in time).
    """
    if trades.empty:
        return trades.copy(), trades.iloc[:0].copy()

    starts = trades.groupby("market_id")["timestamp"].min().sort_values()
    n_train = max(1, int(len(starts) * train_pct))
    train_ids = set(starts.iloc[:n_train].index)

    mask = trades["market_id"].isin(train_ids)
    return trades.loc[mask].copy(), trades.loc[~mask].copy()
