#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export PPC_HTTP_TIMEOUT=120 PPC_HTTP_RETRIES=4 LD_LIBRARY_PATH=/opt/ctranslate2/lib:${LD_LIBRARY_PATH:-}
echo "[$(date -u +%FT%TZ)] launch_holdout_rma2 start" >> step3_transcribe/logs/chain_holdout.log
MANIFEST=data/output/sample_out/sample_manifest_holdout_rma2.csv WORKERS=8 MAX_QUEUED=120 START_ABOVE_FLOOR=2 step3_transcribe/run_ladder.sh >> step3_transcribe/logs/chain_holdout.log 2>&1
echo "[$(date -u +%FT%TZ)] holdout rma2 exited rc=$?" >> step3_transcribe/logs/chain_holdout.log
