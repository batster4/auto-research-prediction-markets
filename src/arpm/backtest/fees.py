"""Polymarket taker fee model — interpolated from the official schedule."""

from __future__ import annotations

import numpy as np

# (price, fee_per_share) derived from Polymarket's published table.
# Fee per share = published fee-for-100-shares / 100.
_PRICE_GRID: list[float] = [
    0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95, 0.99, 1.00,
]
_FEE_GRID: list[float] = [
    0.0, 0.0, 0.00003, 0.00020, 0.00060, 0.00130, 0.00220,
    0.00330, 0.00450, 0.00580, 0.00690, 0.00780,
    0.00840, 0.00860, 0.00840, 0.00770, 0.00660,
    0.00510, 0.00350, 0.00180, 0.00050, 0.0, 0.0,
]

_PRICE_ARR = np.asarray(_PRICE_GRID, dtype=np.float64)
_FEE_ARR = np.asarray(_FEE_GRID, dtype=np.float64)


def taker_fee_per_share(price: float) -> float:
    """Return Polymarket taker fee per share at *price*, linearly interpolated."""
    return float(np.interp(np.clip(price, 0.0, 1.0), _PRICE_ARR, _FEE_ARR))
