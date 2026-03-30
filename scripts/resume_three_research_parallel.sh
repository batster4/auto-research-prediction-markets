#!/usr/bin/env bash
# Resume the three parallel lines using existing experiment dirs (manifest + iterations.jsonl).
# Set RESUME_BS RESUME_GBM RESUME_OPEN to absolute paths of experiment folders, e.g.:
#   export RESUME_BS=/path/experiments/batch1_bs/20260330T115725Z-fee70f4a
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export ARPM_MAX_ITERATIONS="${ARPM_MAX_ITERATIONS:-100}"
export ARPM_MAX_SECONDS_PER_ITERATION="${ARPM_MAX_SECONDS_PER_ITERATION:-300}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"

RS_BS="${RESUME_BS:-}"
RS_GBM="${RESUME_GBM:-}"
RS_OPEN="${RESUME_OPEN:-}"
if [[ -z "$RS_BS" || -z "$RS_GBM" || -z "$RS_OPEN" ]]; then
  echo "Set three experiment directories to resume, e.g.:"
  echo "  export RESUME_BS=$ROOT/experiments/batch1_bs/<stamp-id>"
  echo "  export RESUME_GBM=$ROOT/experiments/batch1_gbm/<stamp-id>"
  echo "  export RESUME_OPEN=$ROOT/experiments/batch1_open/<stamp-id>"
  exit 1
fi

mkdir -p "$ROOT/research_runs"

run_one() {
  local resume_dir="$1" exp_parent="$2" log="$3"
  line="$(basename "$(dirname "$resume_dir")")"
  nohup env ARPM_EXPERIMENTS_DIR="$exp_parent" \
    ARPM_RESEARCH_LINE="$line" \
    ARPM_MAX_ITERATIONS="$ARPM_MAX_ITERATIONS" \
    ARPM_MAX_SECONDS_PER_ITERATION="$ARPM_MAX_SECONDS_PER_ITERATION" \
    PYTHONPATH="$ROOT/src" \
    PYTHONUNBUFFERED=1 \
    "$PY" -m arpm --resume "$resume_dir" \
    >>"$log" 2>&1 &
  echo "PID $!  resume $resume_dir  log $log (appended)"
}

run_one "$RS_BS" "$ROOT/experiments/batch1_bs" "$ROOT/research_runs/run1_bs.log"
run_one "$RS_GBM" "$ROOT/experiments/batch1_gbm" "$ROOT/research_runs/run2_gbm.log"
run_one "$RS_OPEN" "$ROOT/experiments/batch1_open" "$ROOT/research_runs/run3_open.log"

echo "Three resume jobs started. Tail: research_runs/run*.log"
