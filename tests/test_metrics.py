from arpm.evaluation.metrics import evaluate_returns


def test_evaluate_returns_basic():
    s = evaluate_returns([0.1, -0.05, 0.2])
    assert s.total_pnl == 0.25
    assert s.n_markets == 3
    assert s.hit_rate == 2 / 3
