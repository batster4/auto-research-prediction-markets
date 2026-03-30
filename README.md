# Auto Research Prediction Markets

Automated research for **prediction market trading strategies**. The system accepts a research task and historical trade data, proposes candidate strategies through **Anthropic Claude** (primary model: **Claude Sonnet 4.6** via the official **Anthropic Python SDK**), backtests them in isolation, evaluates performance with explicit metrics, and logs reproducible experiments.

## What this repository does

1. **Receive** a research task (natural language) and a dataset path (CSV or Parquet).
2. **Load** normalized historical trades (pluggable format; see schema in `arpm.data`).
3. **Generate** candidate strategies (via Claude or deterministic `--dry-run` mode).
4. **Backtest** strategies using the engine in `arpm.backtest` (separate from strategy definitions).
5. **Evaluate** results with metrics in `arpm.evaluation`.
6. **Track** experiments under `experiments/` (manifest + JSONL iterations).

## Prediction market domain knowledge

Base mechanics (definitions, shares, payouts, resolution, PnL) live in:

- `src/arpm/domain/knowledge.py` — embedded reference used in agent prompts and for humans.

## Research loop constraints

| Setting | Default | Meaning |
|--------|---------|--------|
| Max iterations per experiment | `100` | Upper bound on research cycles (`ARPM_MAX_ITERATIONS`) |
| Max wall-clock per iteration | `300` s (5 min) | Each iteration must finish within this budget (`ARPM_MAX_SECONDS_PER_ITERATION`) |
| Thinking budget | `16000` | Extended thinking (`ARPM_THINKING_BUDGET_TOKENS`; `0` disables) |
| Max output tokens | `32000` | Must exceed thinking budget (`ARPM_MAX_OUTPUT_TOKENS`) |
| Web search | on (`1`) | Server tool (`ARPM_WEB_SEARCH`; `ARPM_WEB_SEARCH_MAX_USES` default `5`) |

The primary model ID is `claude-sonnet-4-6` (`ARPM_MODEL` overrides).

## Quick start

**Requirements:** Python 3.10+, `ANTHROPIC_API_KEY` for live runs.

With [uv](https://docs.astral.sh/uv/):

```bash
uv lock   # first time, or after dependency changes
uv sync --extra dev
uv run pytest -q
```

With plain pip (virtualenv recommended):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

**Offline / CI (no API key):**

```bash
uv run python -m arpm "Compare threshold vs momentum on this slice" --data tests/fixtures/sample_trades.csv --dry-run
```

**With Claude:**

```bash
export ANTHROPIC_API_KEY=...
uv run python -m arpm "Find robust entry rules under noisy prices" --data path/to/trades.csv
```

Artifacts are written under `experiments/<timestamp-uuid>/` (`manifest.json`, `iterations.jsonl`).

## Project layout

```
src/arpm/
  domain/          # Prediction market knowledge (prompt + reference)
  data/            # Loaders + schema (plug in new datasets without changing backtest core)
  strategies/      # Strategy specs and built-in templates
  backtest/        # Simulation engine (no LLM)
  evaluation/      # Metrics
  experiments/     # Manifests + iteration logs
  agent/           # Claude SDK client + research loop
```

## Dataset contract

Normalized columns (aliases accepted on load):

- `timestamp`, `market_id`, `price_yes` ∈ [0, 1], `outcome` ∈ {0, 1} (resolved YES/NO).

See `arpm.data.schema` for alias mapping.

## License

MIT
