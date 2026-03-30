# Prediction Markets Auto-Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LLM pretraining autoresearch with a modular **Auto Research Prediction Markets** system: Claude Sonnet 4.6 drives iterative strategy research with pluggable datasets, isolated backtesting, explicit metrics, and reproducible experiment artifacts.

**Architecture:** Core Python package `arpm` under `src/` with layers: `domain` (prediction-market knowledge text for prompts), `data` (loaders + schema), `strategies` (protocol + built-in templates), `backtest` (engine separate from strategy definitions), `evaluation` (metrics), `experiments` (JSON manifests + JSONL results), `agent` (Anthropic SDK client + time-bounded research loop). CLI via `python -m arpm`. Remove `train.py`, `prepare.py`, and notebook artifacts from the old stack.

**Tech Stack:** Python 3.10+, `anthropic`, `pandas`, `numpy`, `pydantic` (structured parsing), `pytest` for tests.

---

## File map

| Area | Responsibility |
|------|----------------|
| `src/arpm/` | Installable package |
| `src/arpm/domain/knowledge.py` | Base prediction-market mechanics (prompt + human-readable) |
| `src/arpm/data/schema.py` | Trade/market column contracts |
| `src/arpm/data/loaders.py` | CSV/Parquet → normalized DataFrame |
| `src/arpm/strategies/base.py` | `Strategy` protocol + registry |
| `src/arpm/strategies/builtin.py` | Parameterized templates (e.g. threshold, momentum) |
| `src/arpm/backtest/engine.py` | Simulation; no LLM calls |
| `src/arpm/evaluation/metrics.py` | Returns, Sharpe-like, drawdown, hit rate |
| `src/arpm/experiments/store.py` | Experiment dirs, reproducibility metadata |
| `src/arpm/agent/claude_client.py` | Anthropic Messages API wrapper |
| `src/arpm/agent/research_loop.py` | ≤100 iterations, ≤300s wall-clock per iteration |
| `src/arpm/config.py` | Model ID, limits, env |
| `src/arpm/cli.py` + `__main__.py` | Entry points |

---

### Task 1: Project metadata and dependencies

**Files:**
- Modify: `pyproject.toml`
- Delete: `train.py`, `prepare.py`, `analysis.ipynb`

- [x] Set `name` to `auto-research-prediction-markets`, description, readme
- [x] Dependencies: `anthropic`, `pandas`, `numpy`, `pydantic`; dev: `pytest`
- [x] Configure `[build-system]` and `[tool.setuptools.packages.find]` for `src`
- [x] Run `uv lock` / `uv sync`

---

### Task 2: Domain knowledge module

**Files:**
- Create: `src/arpm/domain/knowledge.py`

- [x] Single exported string (Markdown) covering: definition, trading, shares, payouts, probabilities, resolution, PnL
- [x] Optional `get_domain_context()` for agent prompts

---

### Task 3: Data + backtest + metrics

**Files:**
- Create: `src/arpm/data/schema.py`, `loaders.py`
- Create: `src/arpm/strategies/base.py`, `builtin.py`
- Create: `src/arpm/backtest/engine.py`
- Create: `src/arpm/evaluation/metrics.py`
- Create: `tests/test_backtest.py`, `tests/test_metrics.py`
- Create: `tests/fixtures/sample_trades.csv`

- [x] Schema documents required columns; loader validates
- [x] Backtest: per-market binary YES; settlement from `outcome`
- [x] Metrics: total return, mean return per market, max drawdown on equity curve

---

### Task 4: Experiments + Claude + loop

**Files:**
- Create: `src/arpm/experiments/store.py`
- Create: `src/arpm/agent/claude_client.py`, `research_loop.py`
- Create: `src/arpm/config.py`

- [x] `ANTHROPIC_API_KEY` from environment
- [x] Default model `claude-sonnet-4-6`
- [x] Loop: `max_iterations=100`, `max_seconds_per_iteration=300`
- [x] Persist experiment dir with task text, dataset hash, iteration logs

---

### Task 5: CLI, README, program.md

**Files:**
- Create: `src/arpm/cli.py`, `src/arpm/__main__.py`
- Modify: `README.md`, `program.md`, `.gitignore`

- [x] Document rename to **Auto Research Prediction Markets**
- [x] `program.md` describes workflow aligned with new system
- [x] Ignore `experiments/` output

---

## Verification

Run: `uv sync && uv run pytest -q`

Expected: all tests pass; import `arpm` works.
