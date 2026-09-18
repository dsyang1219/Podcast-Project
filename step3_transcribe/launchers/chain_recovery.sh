#!/usr/bin/env bash
# Waits for the in-flight ladder run to finish, then runs the URL-recovery pass.
#
# Why chained rather than concurrent: run_ladder.sh's two halves share one queue
# directory and one STOP sentinel. A second downloader would write STOP as soon
# as IT finished dispatching, killing the transcriber while the first run still
# had episodes queued.
#
# Recovery-specific settings:
#   WORKERS=6          Buck Sexton's CDN (omny via podtrac) returned 246
#                      ReadTimeouts at 16-way concurrency; the URLs were always
#                      live -- a hand curl fetched 500 KB in 7.0s. Fewer workers.
#   PPC_HTTP_TIMEOUT   30s is tight for a 46-minute episode behind two redirects.
set -uo pipefail
cd "$(dirname "$0")/../.."
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[$(date -u +%FT%TZ)] waiting for supervisor pid=$WAIT_PID to exit"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[$(date -u +%FT%TZ)] prior run finished"
fi
sleep 30
export PPC_HTTP_TIMEOUT=120 PPC_HTTP_RETRIES=4
MANIFEST=data/output/sample_out/sample_manifest_recovery.csv \
WORKERS=6 MAX_QUEUED=120 START_ABOVE_FLOOR=2 \
  step3_transcribe/run_ladder.sh
echo "[$(date -u +%FT%TZ)] recovery run exited rc=$?"
