#!/usr/bin/env bash
# Run each curated regime's LDA refit as its OWN subprocess under a real
# cgroup-enforced 48GB memory cap (systemd-run --user --scope -p MemoryMax),
# one at a time, so (a) memory fully releases between regimes regardless of
# Python/tomotopy GC behavior, and (b) if one regime hits the cap, only that
# regime's run dies -- SIGKILL, exit 137 -- rather than losing all four.
# MemorySwapMax=0 so this is a real RAM ceiling, not swap-then-crawl.
set -uo pipefail
# Resolve the repo root from this script's own location -- the corpus lives at a
# different absolute path on rma1 vs rma2, and there is no `set -e` here, so a
# failed cd would otherwise run the whole loop in the wrong directory.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || { echo "FATAL: cannot cd to repo root" >&2; exit 1; }

LOG=data/output/regimes/lda_refit_log.txt
: > "$LOG"

for regime in minimal moderate aggressive aggressive_pos; do
  echo "=== starting regime=$regime (capped 48GB) ===" | tee -a "$LOG"
  systemd-run --user --scope \
    -p MemoryMax=48G -p MemorySwapMax=0 \
    -- .venv/bin/python -u -m nlp.lda_regime_refit --regime "$regime" \
    >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 137 ]; then
    echo "=== regime=$regime KILLED (OOM, hit 48GB cap), rc=$rc ===" | tee -a "$LOG"
  elif [ $rc -ne 0 ]; then
    echo "=== regime=$regime FAILED, rc=$rc ===" | tee -a "$LOG"
  else
    echo "=== regime=$regime completed OK ===" | tee -a "$LOG"
  fi
done
echo "ALL_REGIMES_DONE" | tee -a "$LOG"
