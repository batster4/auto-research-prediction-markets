from pathlib import Path

from arpm.agent.research_loop import run_research_experiment
from arpm.config import Settings


def test_dry_run_experiment(tmp_path):
    csv = Path(__file__).resolve().parent / "fixtures" / "sample_trades.csv"
    settings = Settings(
        anthropic_api_key=None,
        model="claude-sonnet-4-6",
        max_iterations=2,
        max_seconds_per_iteration=300,
        experiments_dir=str(tmp_path),
        thinking_budget_tokens=1024,
        max_output_tokens=4096,
        web_search_enabled=False,
        web_search_max_uses=0,
    )
    paths = run_research_experiment("test task", csv, settings=settings, dry_run=True)
    assert paths.root.exists()
    text = paths.iterations.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 2
