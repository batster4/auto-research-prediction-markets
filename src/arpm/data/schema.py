"""Column contracts for historical trade / quote datasets."""

from __future__ import annotations

import pandas as pd

# Canonical columns after normalization
REQUIRED_TRADE_COLUMNS = ("timestamp", "market_id", "price_yes", "outcome")

# Aliases accepted on load
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "ts", "datetime"),
    "market_id": ("market_id", "market", "condition_id", "event_id"),
    "price_yes": ("price_yes", "price", "yes_price", "mid", "p_yes"),
    "outcome": ("outcome", "resolved_yes", "y"),
}


def normalize_trades_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename aliases to canonical names and validate required columns."""
    lower_map: dict[str, str] = {}
    for canon, variants in COLUMN_ALIASES.items():
        for v in variants:
            lower_map[v.lower()] = canon

    rename: dict[str, str] = {}
    for c in df.columns:
        key = str(c).lower()
        if key in lower_map:
            rename[c] = lower_map[key]

    out = df.rename(columns=rename)
    missing = [c for c in REQUIRED_TRADE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns after alias mapping: {missing}. Have: {list(out.columns)}")

    out = out.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    bad_ts = out["timestamp"].isna()
    if bad_ts.all():
        raise ValueError("Invalid timestamp values present (all rows).")
    if bad_ts.any():
        out = out.loc[~bad_ts].copy()

    out["market_id"] = out["market_id"].astype(str)
    out["price_yes"] = pd.to_numeric(out["price_yes"], errors="coerce")
    if out["price_yes"].isna().any():
        raise ValueError("Invalid price_yes values present.")

    # outcome: 0/1 or boolean or YES/NO strings
    oc = out["outcome"]
    if oc.dtype == object:
        upper = oc.astype(str).str.upper().str.strip()
        mapped = upper.map({"YES": 1.0, "NO": 0.0, "TRUE": 1.0, "FALSE": 0.0})
        out["outcome"] = mapped.fillna(pd.to_numeric(oc, errors="coerce"))
    else:
        out["outcome"] = pd.to_numeric(oc, errors="coerce")

    if out["outcome"].isna().any():
        raise ValueError("Invalid outcome values; use 0/1 or YES/NO.")

    out["outcome"] = out["outcome"].astype(float).clip(0.0, 1.0)
    bad = (out["price_yes"] < 0.0) | (out["price_yes"] > 1.0)
    if bad.any():
        raise ValueError("price_yes must be in [0, 1].")

    return out[list(REQUIRED_TRADE_COLUMNS)]
