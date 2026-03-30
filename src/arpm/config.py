"""Runtime configuration: model, API, and research-loop limits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_from_project_root() -> None:
    """Load `.env` from repo root if present (no python-dotenv dependency)."""
    root = Path(__file__).resolve().parent.parent.parent
    path = root / ".env"
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass

# Primary model per product requirements (Anthropic Messages API)
DEFAULT_MODEL = "claude-sonnet-4-6"

# Research loop: one experiment may run many iterations; each iteration is bounded in wall-clock time.
MAX_RESEARCH_ITERATIONS_PER_EXPERIMENT = 100
MAX_SECONDS_PER_RESEARCH_ITERATION = 300  # 5 minutes

EXPERIMENTS_DIR = "experiments"


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    model: str
    max_iterations: int
    max_seconds_per_iteration: int
    experiments_dir: str

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv_from_project_root()
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=os.environ.get("ARPM_MODEL", DEFAULT_MODEL),
            max_iterations=int(os.environ.get("ARPM_MAX_ITERATIONS", MAX_RESEARCH_ITERATIONS_PER_EXPERIMENT)),
            max_seconds_per_iteration=int(
                os.environ.get("ARPM_MAX_SECONDS_PER_ITERATION", MAX_SECONDS_PER_RESEARCH_ITERATION)
            ),
            experiments_dir=os.environ.get("ARPM_EXPERIMENTS_DIR", EXPERIMENTS_DIR),
        )
