"""Load trade/quote datasets without coupling core logic to file formats."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arpm.data.schema import normalize_trades_dataframe


def load_trades_table(path: str | Path) -> pd.DataFrame:
    """Load CSV or Parquet and return a normalized trades DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {suffix} (use .csv or .parquet)")

    return normalize_trades_dataframe(df)
