from arpm.evaluation.metrics import evaluate_returns


def test_evaluate_returns_basic():
    s = evaluate_returns([0.1, -0.05, 0.2])
    assert abs(s.total_pnl - 0.25) < 1e-9
    assert s.n_markets == 3
    assert s.n_entered == 3  # all non-zero PnLs count as "entered" in legacy wrapper
    assert abs(s.hit_rate - 2 / 3) < 1e-9


def test_evaluate_returns_with_losses():
    s = evaluate_returns([0.5, -0.3, 0.0, -0.1])
    assert s.n_markets == 4
    assert s.n_entered == 3  # 0.5, -0.3, -0.1
    assert abs(s.hit_rate - 1 / 3) < 1e-9
    assert s.profit_factor is not None
    assert abs(s.profit_factor - 0.5 / 0.4) < 1e-6
    assert s.max_drawdown >= 0
