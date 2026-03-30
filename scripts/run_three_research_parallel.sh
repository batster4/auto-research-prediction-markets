#!/usr/bin/env bash
# Run three ARPM research jobs in parallel. Requires ANTHROPIC_API_KEY (.env loaded by arpm).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export ARPM_MAX_ITERATIONS="${ARPM_MAX_ITERATIONS:-100}"
export ARPM_MAX_SECONDS_PER_ITERATION="${ARPM_MAX_SECONDS_PER_ITERATION:-300}"
DATA="${ARPM_DATA:-$ROOT/data/arpm_c3_resolved_mar12plus_sample500k.csv}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -f "$DATA" ]]; then
  echo "Missing dataset: $DATA — build with scripts/build_arpm_dataset_c3_jsonl.py or set ARPM_DATA"
  exit 1
fi

mkdir -p "$ROOT/research_runs"

run_one() {
  local taskfile="$1" expdir="$2" log="$3"
  nohup env ARPM_EXPERIMENTS_DIR="$expdir" \
    ARPM_MAX_ITERATIONS="$ARPM_MAX_ITERATIONS" \
    ARPM_MAX_SECONDS_PER_ITERATION="$ARPM_MAX_SECONDS_PER_ITERATION" \
    PYTHONPATH="$ROOT/src" \
    PYTHONUNBUFFERED=1 \
    "$PY" -m arpm --task-file "$taskfile" --data "$DATA" \
    >"$log" 2>&1 &
  echo "PID $!  log $log  experiments -> $expdir"
}

run_one "$ROOT/research_tasks/01_binary_bs_vol_mispricing.txt" \
  "$ROOT/experiments/batch1_bs" "$ROOT/research_runs/run1_bs.log"
run_one "$ROOT/research_tasks/02_last_second_gbm_edge.txt" \
  "$ROOT/experiments/batch1_gbm" "$ROOT/research_runs/run2_gbm.log"
run_one "$ROOT/research_tasks/03_opening_second_inefficiency.txt" \
  "$ROOT/experiments/batch1_open" "$ROOT/research_runs/run3_open.log"

echo "Three jobs started in background. Tail: research_runs/run*.log"
