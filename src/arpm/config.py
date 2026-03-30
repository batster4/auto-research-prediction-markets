"""Runtime configuration: model, API, and research-loop limits."""

from __future__ import annotations

import os
from dataclasses import dataclass

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
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            model=os.environ.get("ARPM_MODEL", DEFAULT_MODEL),
            max_iterations=int(os.environ.get("ARPM_MAX_ITERATIONS", MAX_RESEARCH_ITERATIONS_PER_EXPERIMENT)),
            max_seconds_per_iteration=int(
                os.environ.get("ARPM_MAX_SECONDS_PER_ITERATION", MAX_SECONDS_PER_RESEARCH_ITERATION)
            ),
            experiments_dir=os.environ.get("ARPM_EXPERIMENTS_DIR", EXPERIMENTS_DIR),
        )
