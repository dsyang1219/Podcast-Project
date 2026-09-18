#!/usr/bin/env bash
# After rma2's own holdout shard finishes, pull rma1's completed HOLDOUT transcripts here (so the downloader
# skips them) and run rma1's shard manifest on rma2 too. rma1 keeps running; the two converge.
set -uo pipefail
cd "$(dirname "$0")/../.."
WAIT_PID="${1:-}"
echo "[$(date -u +%FT%TZ)] takeover chain: waiting for rma2 holdout supervisor pid=$WAIT_PID" >> step3_transcribe/logs/chain_holdout.log
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] rma2 shard finished; syncing rma1 HOLDOUT transcripts" >> step3_transcribe/logs/chain_holdout.log
rsync -a -e "ssh -i $HOME/.ssh/id_ed25519_rma1 -o BatchMode=yes" --include='HOLDOUT_*/' --include='HOLDOUT_*/**' --exclude='*' \
  rma1.caltech.edu:/home/dsyang/Podcast-Project/political-podcast-corpus/data/transcripts/ data/transcripts/ >> step3_transcribe/logs/chain_holdout.log 2>&1
echo "[$(date -u +%FT%TZ)] synced; launching rma1 shard on rma2" >> step3_transcribe/logs/chain_holdout.log
rm -f data/audio_queue/STOP
export PPC_HTTP_TIMEOUT=120 PPC_HTTP_RETRIES=4 LD_LIBRARY_PATH=/opt/ctranslate2/lib:${LD_LIBRARY_PATH:-}
MANIFEST=data/output/sample_out/sample_manifest_holdout_rma1.csv WORKERS=8 MAX_QUEUED=120 START_ABOVE_FLOOR=2 step3_transcribe/run_ladder.sh >> step3_transcribe/logs/chain_holdout.log 2>&1
echo "[$(date -u +%FT%TZ)] takeover run exited rc=$?" >> step3_transcribe/logs/chain_holdout.log
