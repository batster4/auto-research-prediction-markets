#!/usr/bin/env bash
# Stop background `python -m arpm` jobs without matching the shell that runs pkill
# (see https://stackoverflow.com/questions/9379417/why-does-pkill-kill-my-ssh-session).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Pattern uses [p] so pkill's argv does not match itself.
pkill -f '[p]ython -m arpm' 2>/dev/null || true
echo "Stopped ARPM python jobs (if any were running)."
