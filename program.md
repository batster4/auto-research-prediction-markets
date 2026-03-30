# Auto Research Prediction Markets — agent program

This file is the lightweight **program** for an autonomous researcher working in this repository.

## Purpose

Run **iterative research loops** to discover and refine **prediction market** trading strategies: load historical trades, propose strategies (via **Anthropic Claude**, model **`claude-sonnet-4-6`**), backtest, evaluate, and keep the best ideas in experiment logs.

## Setup

1. Read `README.md` and `src/arpm/domain/knowledge.py` so reasoning matches real PM mechanics.
2. Ensure `uv sync` works and tests pass: `uv run pytest -q`.
3. For live runs, confirm `ANTHROPIC_API_KEY` is set (official **Anthropic Python SDK**).
4. **Web search** is enabled by default (`ARPM_WEB_SEARCH=1`): the agent may query the web for **recent** papers, quant notes, or related strategies (binary options, GBM, calibration, etc.) and map ideas into JSON strategy specs. Disable with `ARPM_WEB_SEARCH=0` if your org disables server tools.
5. **Extended thinking** is on by default (`ARPM_THINKING_BUDGET_TOKENS=16000`); output cap `ARPM_MAX_OUTPUT_TOKENS` must exceed the thinking budget.

## Workflow

1. **Research task** — agree with the user on the hypothesis and constraints; capture it as the CLI `task` string or `--task-file`.
2. **Dataset** — obtain a CSV/Parquet path; columns must map to the schema in `arpm.data` (see README).
3. **Experiment** — run `uv run python -m arpm "<task>" --data <path>` or use `--dry-run` when no API key is available.
4. **Iterations** — each iteration receives **full prior results** (`iterations.jsonl` content as JSON): candidate strategies, metrics, and `best_in_iteration`. The model is instructed to **not repeat** failed parameter sets and to diversify. Each iteration may also use **web search** (when enabled) before emitting the JSON array of strategies.
5. **Limits** — default **100** iterations per experiment and **≤5 minutes** wall-clock per iteration (configurable via env vars in `arpm.config`).

## What you edit

- Prefer extending **`src/arpm/strategies/builtin.py`** or adding new templates plus registration in `strategy_from_spec` rather than changing the backtest core.
- Keep **backtest** and **strategy logic** separate: strategies live under `arpm.strategies`; simulation under `arpm.backtest`.

## Outputs

- Each run creates `experiments/<id>/manifest.json` (task, dataset path, hash) and `iterations.jsonl` (per-iteration candidate metrics and best-in-iteration).
- The **same** `iterations.jsonl` is fed back to Claude on the next iteration as **research memory** so the agent sees what was already tried.
- Summarize **best-performing strategies** for the user with reference to iteration indices and metric values.

## Constraints

- Do not send secrets to logs.
- Keep runs reproducible: note dataset path and `dataset_sha256` from the manifest when comparing runs.
