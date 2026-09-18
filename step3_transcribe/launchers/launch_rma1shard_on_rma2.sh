#!/usr/bin/env bash
# rma1-shard manifest on rma2. GATE=1 -> thermal start gate on (default protection); GATE=0 -> ungated.
cd "$(dirname "$0")/../.."; GATE="${GATE:-1}"
export PPC_HTTP_TIMEOUT=120 PPC_HTTP_RETRIES=4 LD_LIBRARY_PATH=/opt/ctranslate2/lib:${LD_LIBRARY_PATH:-}
echo "[$(date -u +%FT%TZ)] launch rma1-shard on rma2, GATE=$GATE" >> step3_transcribe/logs/chain_holdout.log
if [ "$GATE" = "1" ]; then export START_ABOVE_FLOOR=2; else unset START_ABOVE_FLOOR START_BELOW_F TARGET_TEMP_F; fi
MANIFEST=${MANIFEST_OVERRIDE:-data/output/sample_out/sample_manifest_holdout_rma1.csv} WORKERS=8 MAX_QUEUED=120 step3_transcribe/run_ladder.sh >> step3_transcribe/logs/chain_holdout.log 2>&1
echo "[$(date -u +%FT%TZ)] rma1-shard run on rma2 exited rc=$? (GATE=$GATE)" >> step3_transcribe/logs/chain_holdout.log
