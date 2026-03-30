# Auto Research Prediction Markets — agent program

This file is the lightweight **program** for an autonomous researcher working in this repository.

## Purpose

Run **iterative research loops** to discover and refine **prediction market** trading strategies: load historical trades, propose strategies (via **Anthropic Claude**, model **`claude-sonnet-4-6`**), backtest, evaluate, and keep the best ideas in experiment logs.

## Setup

1. Read `README.md` and `src/arpm/domain/knowledge.py` so reasoning matches real PM mechanics.
2. Ensure `uv sync` works and tests pass: `uv run pytest -q`.
3. For live runs, confirm `ANTHROPIC_API_KEY` is set (official **Anthropic Python SDK**).

## Workflow

1. **Research task** — agree with the user on the hypothesis and constraints; capture it as the CLI `task` string.
2. **Dataset** — obtain a CSV/Parquet path; columns must map to the schema in `arpm.data` (see README).
3. **Experiment** — run `uv run python -m arpm "<task>" --data <path>` or use `--dry-run` when no API key is available.
4. **Iterations** — each iteration may analyze prior results (see `iterations.jsonl`), propose strategies (JSON specs: `threshold`, `momentum`, `hold`), backtest, and record metrics.
5. **Limits** — respect **≤100** iterations per experiment and **≤5 minutes** wall-clock per iteration (configurable via env vars in `arpm.config`).

## What you edit

- Prefer extending **`src/arpm/strategies/builtin.py`** or adding new templates plus registration in `strategy_from_spec` rather than changing the backtest core.
- Keep **backtest** and **strategy logic** separate: strategies live under `arpm.strategies`; simulation under `arpm.backtest`.

## Outputs

- Each run creates `experiments/<id>/manifest.json` (task, dataset path, hash) and `iterations.jsonl` (per-iteration candidate metrics).
- Summarize **best-performing strategies** for the user with reference to iteration indices and metric values.

## Constraints

- Do not send secrets to logs.
- Keep runs reproducible: note dataset path and `dataset_sha256` from the manifest when comparing runs.
