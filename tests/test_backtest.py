from pathlib import Path

from arpm.backtest.engine import run_backtest
from arpm.data.loaders import load_trades_table
from arpm.strategies.builtin import ThresholdStrategy


def test_load_fixture():
    root = Path(__file__).resolve().parent / "fixtures" / "sample_trades.csv"
    df = load_trades_table(root)
    assert len(df) >= 4
    assert set(df.columns) == {"timestamp", "market_id", "price_yes", "outcome"}


def test_threshold_backtest_total_pnl():
    root = Path(__file__).resolve().parent / "fixtures" / "sample_trades.csv"
    df = load_trades_table(root)
    strat = ThresholdStrategy(buy_below=0.5)
    bt = run_backtest(df, strat)
    # Fixture designed: m1 YES at 0.5 -> +0.5; m2 YES at 0.3 when outcome 0 -> -0.3
    assert abs(bt.total_pnl - 0.2) < 1e-9
