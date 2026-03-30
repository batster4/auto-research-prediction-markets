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

# Claude Messages API: extended thinking + output cap (must exceed thinking budget).
DEFAULT_THINKING_BUDGET_TOKENS = 16_000
DEFAULT_MAX_OUTPUT_TOKENS = 32_000

EXPERIMENTS_DIR = "experiments"


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _coerce_max_output_tokens(max_out: int, thinking_budget: int) -> int:
    """API requires max_tokens > thinking budget when extended thinking is enabled."""
    if thinking_budget <= 0:
        return max_out
    if max_out <= thinking_budget:
        return thinking_budget + 4096
    return max_out


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    model: str
    max_iterations: int
    max_seconds_per_iteration: int
    experiments_dir: str
    thinking_budget_tokens: int
    max_output_tokens: int
    web_search_enabled: bool
    web_search_max_uses: int

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
            thinking_budget_tokens=int(
                os.environ.get("ARPM_THINKING_BUDGET_TOKENS", DEFAULT_THINKING_BUDGET_TOKENS)
            ),
            max_output_tokens=_coerce_max_output_tokens(
                int(os.environ.get("ARPM_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
                int(os.environ.get("ARPM_THINKING_BUDGET_TOKENS", DEFAULT_THINKING_BUDGET_TOKENS)),
            ),
            web_search_enabled=_env_bool("ARPM_WEB_SEARCH", True),
            web_search_max_uses=int(os.environ.get("ARPM_WEB_SEARCH_MAX_USES", "5")),
        )
