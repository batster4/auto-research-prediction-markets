#!/usr/bin/env bash
# Periodic health check for parallel ARPM research jobs (cron every 15 min).
# Logs to research_runs/health_monitor.log; set RESTART_ON_FAILURE=1 to auto-restart dead jobs.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="$ROOT/research_runs"
LOG="$RUNS/health_monitor.log"
DATA_BS="${ARPM_DATA_BS:-}"
DATA_GBM="${ARPM_DATA_GBM:-}"
DATA_OPEN="${ARPM_DATA_OPEN:-}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

mkdir -p "$RUNS"

{
  echo "===== $TS health check ====="

  # Running arpm processes
  nproc="$(pgrep -f "python.*-m arpm" 2>/dev/null | wc -l | tr -d ' ')"
  nproc="${nproc:-0}"
  echo "arpm_processes=$nproc"

  # Disk
  df -h "$ROOT" 2>/dev/null | tail -1 | awk '{print "disk_avail="$4" use="$5}'

  # Recent errors in research logs (last 800 lines each)
  for f in run1_bs.log run2_gbm.log run3_open.log; do
    p="$RUNS/$f"
    if [[ -f "$p" ]]; then
      errlines="$(grep -E "Traceback|Error:|Exception:|ValueError|ModuleNotFoundError|RuntimeError|CRITICAL" "$p" 2>/dev/null | tail -5 || true)"
      if [[ -n "$errlines" ]]; then
        echo "WARN $f errors/traceback (last matches):"
        echo "$errlines"
      else
        echo "OK $f no error headers in file"
      fi
      if grep -q "Experiment directory:" "$p" 2>/dev/null; then
        echo "OK $f finished (saw Experiment directory)"
      else
        echo "INFO $f not yet completed (no Experiment directory line)"
      fi
    else
      echo "WARN missing log $p"
    fi
  done

  # Failure if any process died before completion
  need_restart=0
  if [[ "$nproc" -eq 0 ]]; then
    done1=0 done2=0 done3=0
    grep -q "Experiment directory:" "$RUNS/run1_bs.log" 2>/dev/null && done1=1
    grep -q "Experiment directory:" "$RUNS/run2_gbm.log" 2>/dev/null && done2=1
    grep -q "Experiment directory:" "$RUNS/run3_open.log" 2>/dev/null && done3=1
    if [[ $done1 -eq 1 && $done2 -eq 1 && $done3 -eq 1 ]]; then
      echo "STATUS all three jobs completed successfully (no running processes)"
    else
      echo "CRITICAL no arpm processes but not all logs show completion"
      need_restart=1
    fi
  elif [[ "$nproc" -lt 3 ]]; then
    echo "WARN expected 3 parallel jobs; only $nproc running (may be finishing)"
  else
    echo "STATUS running ($nproc arpm processes)"
  fi

  if [[ "${RESTART_ON_FAILURE:-0}" == "1" && "$need_restart" -eq 1 ]]; then
    if [[ -n "$DATA_BS" && -n "$DATA_GBM" && -n "$DATA_OPEN" && -f "$DATA_BS" && -f "$DATA_GBM" && -f "$DATA_OPEN" ]]; then
      echo "ACTION restarting three jobs via run_three_research_parallel.sh"
      export ARPM_DATA_BS="$DATA_BS" ARPM_DATA_GBM="$DATA_GBM" ARPM_DATA_OPEN="$DATA_OPEN"
      export ARPM_MAX_ITERATIONS="${ARPM_MAX_ITERATIONS:-12}"
      bash "$ROOT/scripts/run_three_research_parallel.sh" || echo "WARN restart script failed"
    else
      echo "CRITICAL cannot restart: set ARPM_DATA_BS, ARPM_DATA_GBM, ARPM_DATA_OPEN to three existing CSV paths"
    fi
  fi

  echo ""
} >>"$LOG" 2>&1

# Trim log if huge (keep last 500 lines)
if [[ -f "$LOG" ]]; then
  lines=$(wc -l <"$LOG" | tr -d ' ')
  if [[ "${lines:-0}" -gt 2000 ]]; then
    tail -500 "$LOG" >"${LOG}.tmp" && mv "${LOG}.tmp" "$LOG"
  fi
fi

exit 0
