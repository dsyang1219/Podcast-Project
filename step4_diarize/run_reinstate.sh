#!/usr/bin/env bash
# Reinstated shows (frame_decisions.md, 23 Nov 2026): after the re-transcription
# chain finishes, transcribe + archive their ladder manifest, then diarize and
# redo host identification, then refresh the sandbox.
set -u
cd "$(dirname "$0")/.."
M=data/output/sample_out/sample_manifest_reinstated15.csv
echo "[$(date -u +%FT%TZ)] waiting for run_retranscribe.sh to finish"
while pgrep -f "^bash step4_diarize/run_retranscribe.sh" >/dev/null; do sleep 900; done
echo "[$(date -u +%FT%TZ)] transcribing $(($(wc -l < $M)-1)) episodes of the 15 reinstated shows"
ARCHIVE=1 MANIFEST=$M WORKERS=6 MAX_QUEUED=120 START_ABOVE_FLOOR=2 PPC_HTTP_TIMEOUT=120 PPC_HTTP_RETRIES=4 LD_LIBRARY_PATH=/opt/ctranslate2/lib \
  bash step3_transcribe/run_ladder.sh > step3_transcribe/logs/run_reinstate15.log 2>&1
echo "[$(date -u +%FT%TZ)] ladder finished (state $(python3 -c "import json;print(json.load(open('data/output/ladder_status.json'))['state'])")); diarizing"
bash step4_diarize/run_corpus.sh > step4_diarize/logs/corpus_reinstate15.log 2>&1
echo "[$(date -u +%FT%TZ)] refreshing episode table and sandbox"
.venv/bin/python - <<'PY'
import subprocess
PY
.venv/bin/python step2_sample/build_sandbox.py --bands 5 > /dev/null 2>&1
echo "[$(date -u +%FT%TZ)] reinstatement done"
