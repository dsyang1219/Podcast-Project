#!/usr/bin/env bash
# Unattended driver: carries the ad-span pipeline from the in-flight batch
# waves all the way to the corrected ideology R2, with no agent and no SSH
# session attached.
#
# Every step is resumable and idempotent, so re-running this after any
# interruption picks up where it stopped rather than redoing paid work:
#   run-waves  skips waves whose results are already on disk
#   collect    re-reads downloaded results; re-excises from scratch (free)
#   rechunk    deterministic re-segmentation
#   sweep      skips any K already fit (results are appended per K)
#
# Usage:  setsid nohup bash nlp/adspan_driver.sh > data/output/adspan/driver.log 2>&1 &

set -u -o pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LOG_DIR=data/output/adspan
mkdir -p "$LOG_DIR"

step() {
  local name="$1"; shift
  echo "=== [$(date '+%F %T')] START $name ==="
  if ! "$@"; then
    echo "=== [$(date '+%F %T')] FAILED $name (exit $?) -- stopping driver ==="
    exit 1
  fi
  echo "=== [$(date '+%F %T')] DONE  $name ==="
}

# The wave runner launched earlier is still going; do not start a second one
# (two concurrent runners would double-submit waves and double-bill).
if pgrep -f "adspan_phase_b run-waves" > /dev/null; then
  echo "[driver] existing wave runner detected -- waiting for it to exit"
  while pgrep -f "adspan_phase_b run-waves" > /dev/null; do sleep 60; done
  echo "[driver] wave runner exited"
fi

# Re-run run-waves regardless: it is a no-op if every wave already has results,
# and it retries any wave that --keep-going skipped after a non-completed status.
step "phase B: waves"        $PY -m nlp.adspan_phase_b run-waves --poll-interval 30 --keep-going
step "phase B: collect"      $PY -m nlp.adspan_phase_b collect
step "phase C: rechunk"      $PY -m nlp.adspan_rechunk --n-process 10
# --corpus clean is the default, but state it explicitly: this is the run whose
# numbers get reported, and it must never silently fall back to the preview
# corpus if the default ever changes.
step "phase C: tokens"       $PY -m nlp.adspan_phase_c --export-tokens

# Run the K grid as concurrent processes. Sequentially each fit used only
# TOMOTOPY_WORKERS(8) of this box's 20 cores, leaving 60% idle for ~3.5h.
# `workers` is deliberately NOT raised -- tomotopy's result depends on the
# worker count, and 8 is what the on-record 0.451 baseline used. The speedup
# comes from filling idle cores with the other K values, not from changing
# any fit setting. Each K writes its own file; --merge combines them.
echo "=== [$(date '+%F %T')] START phase C: K sweep (5 concurrent fits) ==="
pids=()
for K in 30 50 75 100 150; do
  $PY -m nlp.adspan_phase_c --sweep --corpus clean --only-k "$K" \
      > "$LOG_DIR/sweep_clean_k${K}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" -ne 0 ]; then
  echo "=== [$(date '+%F %T')] FAILED phase C: K sweep -- see $LOG_DIR/sweep_clean_k*.log ==="
  exit 1
fi
echo "=== [$(date '+%F %T')] DONE  phase C: K sweep ==="

step "phase C: merge"        $PY -m nlp.adspan_phase_c --merge  --corpus clean
step "phase C: report"       $PY -m nlp.adspan_phase_c --report --corpus clean

echo "=== [$(date '+%F %T')] PIPELINE COMPLETE ==="
echo "Headline results: $LOG_DIR/phase_c_report.json"
echo "K curve:          $LOG_DIR/phase_c_ksweep.json"
echo "Topic table:      $LOG_DIR/ideology_topics_adclean.csv"
echo "Excision audit:   $LOG_DIR/excision_audit.jsonl.gz"
echo "Per-show removed: $LOG_DIR/removed_share_per_show.csv"
