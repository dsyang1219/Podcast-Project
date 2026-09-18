#!/usr/bin/env bash
# Waits for the in-flight recovery ladder run to exit, then runs the HOLDOUT manifest.
# Chained, not concurrent: run_ladder.sh's halves share one queue dir and one STOP sentinel.
set -uo pipefail
cd "$(dirname "$0")/../.."
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[$(date -u +%FT%TZ)] holdout chain: waiting for supervisor pid=$WAIT_PID to exit"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[$(date -u +%FT%TZ)] prior run finished"
fi
sleep 30
export PPC_HTTP_TIMEOUT=120 PPC_HTTP_RETRIES=4
MANIFEST=data/output/sample_out/sample_manifest_holdout_rma2.csv \
WORKERS=8 MAX_QUEUED=120 START_ABOVE_FLOOR=2 \
  step3_transcribe/run_ladder.sh
echo "[$(date -u +%FT%TZ)] holdout run exited rc=$?"
