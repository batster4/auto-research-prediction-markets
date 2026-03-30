#!/usr/bin/env bash
# Run three ARPM research jobs in parallel. Requires ANTHROPIC_API_KEY (.env loaded by arpm).
# Each run MUST use a different trades CSV (fully independent backtests). Set:
#   ARPM_DATA_BS   ARPM_DATA_GBM   ARPM_DATA_OPEN
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export ARPM_MAX_ITERATIONS="${ARPM_MAX_ITERATIONS:-100}"
export ARPM_MAX_SECONDS_PER_ITERATION="${ARPM_MAX_SECONDS_PER_ITERATION:-300}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"

DATA_BS="${ARPM_DATA_BS:-}"
DATA_GBM="${ARPM_DATA_GBM:-}"
DATA_OPEN="${ARPM_DATA_OPEN:-}"
if [[ -z "$DATA_BS" || -z "$DATA_GBM" || -z "$DATA_OPEN" ]]; then
  echo "Each run needs its own dataset file. Export three distinct paths, for example:"
  echo "  export ARPM_DATA_BS=/path/to/trades_bs.csv"
  echo "  export ARPM_DATA_GBM=/path/to/trades_gbm.csv"
  echo "  export ARPM_DATA_OPEN=/path/to/trades_open.csv"
  echo "Then: bash scripts/run_three_research_parallel.sh"
  exit 1
fi

_resolve() {
  if command -v realpath >/dev/null 2>&1; then realpath "$1" 2>/dev/null || echo "$1"
  else readlink -f "$1" 2>/dev/null || echo "$1"
  fi
}

_check_data() {
  local envname="$1" path="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing file for $envname: $path"
    exit 1
  fi
}
_check_data ARPM_DATA_BS "$DATA_BS"
_check_data ARPM_DATA_GBM "$DATA_GBM"
_check_data ARPM_DATA_OPEN "$DATA_OPEN"

R1="$(_resolve "$DATA_BS")"
R2="$(_resolve "$DATA_GBM")"
R3="$(_resolve "$DATA_OPEN")"
if [[ "$R1" == "$R2" || "$R1" == "$R3" || "$R2" == "$R3" ]]; then
  echo "Error: the three dataset paths must be different files (got BS=$R1 GBM=$R2 OPEN=$R3)."
  exit 1
fi

mkdir -p "$ROOT/research_runs"

run_one() {
  local taskfile="$1" expdir="$2" log="$3" datafile="$4"
  # Unique label per research line so the agent prompt does not treat parallel runs as one study.
  line="$(basename "$expdir")"
  nohup env ARPM_EXPERIMENTS_DIR="$expdir" \
    ARPM_RESEARCH_LINE="$line" \
    ARPM_MAX_ITERATIONS="$ARPM_MAX_ITERATIONS" \
    ARPM_MAX_SECONDS_PER_ITERATION="$ARPM_MAX_SECONDS_PER_ITERATION" \
    PYTHONPATH="$ROOT/src" \
    PYTHONUNBUFFERED=1 \
    "$PY" -m arpm --task-file "$taskfile" --data "$datafile" \
    >"$log" 2>&1 &
  echo "PID $!  log $log  data=$datafile  experiments -> $expdir"
}

run_one "$ROOT/research_tasks/01_binary_bs_vol_mispricing.txt" \
  "$ROOT/experiments/batch1_bs" "$ROOT/research_runs/run1_bs.log" "$DATA_BS"
run_one "$ROOT/research_tasks/02_last_second_gbm_edge.txt" \
  "$ROOT/experiments/batch1_gbm" "$ROOT/research_runs/run2_gbm.log" "$DATA_GBM"
run_one "$ROOT/research_tasks/03_opening_second_inefficiency.txt" \
  "$ROOT/experiments/batch1_open" "$ROOT/research_runs/run3_open.log" "$DATA_OPEN"

echo "Three jobs started in background. Tail: research_runs/run*.log"
