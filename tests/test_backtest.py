from pathlib import Path

from arpm.backtest.engine import run_backtest
from arpm.data.loaders import load_trades_table
from arpm.strategies.builtin import ThresholdStrategy


def test_load_fixture():
    root = Path(__file__).resolve().parent / "fixtures" / "sample_trades.csv"
    df = load_trades_table(root)
    assert len(df) >= 4
    assert set(df.columns) == {"timestamp", "market_id", "price_yes", "outcome"}


def test_threshold_backtest_no_fees():
    """Core PnL logic — fees and slippage off so we can check exact values."""
    root = Path(__file__).resolve().parent / "fixtures" / "sample_trades.csv"
    df = load_trades_table(root)
    strat = ThresholdStrategy(buy_below=0.5)
    bt = run_backtest(df, strat, resolution_cutoff_s=60, slippage=0, apply_fees=False)
    # m1: enters at 0.48 (first row <= 0.50 after cutoff), outcome=1 → PnL = 0.52
    # m2: enters at 0.30, outcome=0 → PnL = -0.30
    # With cutoff=60s, last row (T+3h) is removed; rows at T+0,T+1h,T+2h remain.
    # m1 tradeable prices: 0.50, 0.48, 0.52 → first <=0.50 is 0.50 at T+0h
    # PnL = 1 - 0.50 = 0.50
    # m2 tradeable prices: 0.30, 0.30, 0.30 → first <=0.50 is 0.30
    # PnL = 0 - 0.30 = -0.30
    assert abs(bt.total_pnl - 0.2) < 1e-9
    assert bt.markets_traded == 2


def test_threshold_backtest_with_fees():
    """With fees and slippage, PnL is lower than the ideal case."""
    root = Path(__file__).resolve().parent / "fixtures" / "sample_trades.csv"
    df = load_trades_table(root)
    strat = ThresholdStrategy(buy_below=0.5)
    bt = run_backtest(df, strat, resolution_cutoff_s=60, slippage=0.005, apply_fees=True)
    assert bt.total_pnl < 0.2  # lower than no-fee case
    assert bt.markets_traded == 2
    assert bt.total_fees > 0


def test_resolution_cutoff_prevents_late_entry():
    """A strategy with buy_below=0.0 should enter zero markets when cutoff
    removes the near-resolution ticks where price=0 appears."""
    import pandas as pd
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01T00:04:50Z",  # 10s before end
            "2024-01-01T00:04:55Z",
            "2024-01-01T00:05:00Z",  # last tick
        ]),
        "market_id": ["m1", "m1", "m1"],
        "price_yes": [0.40, 0.10, 0.00],
        "outcome": [1, 1, 1],
    })
    strat = ThresholdStrategy(buy_below=0.01)
    bt = run_backtest(df, strat, resolution_cutoff_s=60, slippage=0, apply_fees=False)
    assert bt.markets_traded == 0  # all rows within 60s of last → no tradeable data
