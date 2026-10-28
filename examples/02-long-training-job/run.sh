#!/usr/bin/env bash
# Payload entrypoint for scripts/colab-job.sh. The contract:
#   * idempotent - safe to invoke again after a restart or VM swap
#   * writes incremental output under out/
#   * prints progress to stdout (keeps the control channel warm)
#   * prints the literal line JOB_COMPLETE when fully finished
# Break the first or last and the supervisor redoes work or polls forever.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p out state

echo "payload starting on $(hostname) at $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"

# one process per unit: a bad unit can't kill the run, and the per-unit output
# file is what makes this restartable
TOTAL="${TOTAL_UNITS:-20}"
for ((u=1; u<=TOTAL; u++)); do
  if [[ -f "out/unit_${u}.json" ]]; then
    echo "unit $u already done, skipping"
    continue
  fi
  echo "unit $u/$TOTAL starting"
  if python3 train_step.py "$u"; then
    echo "unit $u complete"
  else
    echo "unit $u FAILED (rc=$?) - continuing to the next unit"
  fi
done

DONE=$(find out -name 'unit_*.json' | wc -l)
echo "completed $DONE/$TOTAL units"

# gate on real state, not on reaching the end of the loop
if [[ "$DONE" -eq "$TOTAL" ]]; then
  echo "JOB_COMPLETE"
else
  echo "INCOMPLETE: $DONE of $TOTAL - supervisor will relaunch"
  exit 1
fi
