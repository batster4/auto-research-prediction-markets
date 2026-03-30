from pathlib import Path

from arpm.agent.research_loop import run_research_experiment
from arpm.config import Settings


def _settings(tmp_path: Path, max_iters: int = 2) -> Settings:
    return Settings(
        anthropic_api_key=None,
        model="claude-sonnet-4-6",
        max_iterations=max_iters,
        max_seconds_per_iteration=300,
        experiments_dir=str(tmp_path),
        thinking_budget_tokens=1024,
        max_output_tokens=4096,
        web_search_enabled=False,
        web_search_max_uses=0,
    )


def test_dry_run_experiment(tmp_path):
    csv = Path(__file__).resolve().parent / "fixtures" / "sample_trades.csv"
    paths = run_research_experiment("test task", csv, settings=_settings(tmp_path), dry_run=True)
    assert paths.root.exists()
    lines = paths.iterations.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_resume_dry_run(tmp_path):
    csv = Path(__file__).resolve().parent / "fixtures" / "sample_trades.csv"
    paths = run_research_experiment("test task", csv, settings=_settings(tmp_path, 2), dry_run=True)
    paths2 = run_research_experiment(
        "", None, settings=_settings(tmp_path, 5), dry_run=True, resume_from=paths.root,
    )
    assert paths2.root == paths.root
    lines = paths2.iterations.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
