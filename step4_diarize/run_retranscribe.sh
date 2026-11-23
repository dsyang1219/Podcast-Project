#!/usr/bin/env bash
# After the corpus diarization run: re-transcribe every episode whose archived
# audio differs from the originally transcribed file (dynamic ad insertion),
# relabel their segments from the stored turns, and redo host identification.
set -u
cd "$(dirname "$0")/.."
PY=../whisperx-env/bin/python
WORKERS="${WORKERS:-2}"
echo "[$(date -u +%FT%TZ)] waiting for run_corpus.sh to finish"
while pgrep -f "^bash step4_diarize/run_corpus.sh" >/dev/null; do sleep 900; done
echo "[$(date -u +%FT%TZ)] re-transcribing mismatched episodes with $WORKERS GPU workers"
for ((k = 0; k < WORKERS; k++)); do
  LD_LIBRARY_PATH=/opt/ctranslate2/lib $PY step4_diarize/retranscribe.py --from-diarization --shard "$k/$WORKERS" \
    --priority-shows data/output/diarization/priority_shows.txt \
    --priority-episodes data/output/diarization/priority_episodes_sandbox.txt \
    > "step4_diarize/logs/retranscribe_w$k.log" 2>&1 &
done
wait
grep -hE "^\[.*(targets|finished)" step4_diarize/logs/retranscribe_w*.log | cut -c1-160
echo "[$(date -u +%FT%TZ)] relabelling segments"
$PY step4_diarize/diarize.py --reassign-only 2>&1 | grep -E "^\[" | cut -c1-160
echo "[$(date -u +%FT%TZ)] host identification"
.venv/bin/python step4_diarize/identify_hosts.py --write-host-text 2>&1 | tail -6
echo "[$(date -u +%FT%TZ)] retranscribe chain done"
