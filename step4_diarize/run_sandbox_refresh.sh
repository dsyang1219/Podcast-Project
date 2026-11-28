#!/usr/bin/env bash
# Wait until every sandbox episode has an audio-aligned transcript, then relabel
# segments and rebuild the sandbox. Safe to run while the re-transcription chain works.
set -u
cd "$(dirname "$0")/.."
LIST=data/output/diarization/priority_episodes_sandbox.txt
remaining() { python3 -c "
import os,sys
n=0
for l in open('$LIST'):
    s,e=l.strip().split(',')[:2]
    if not os.path.exists(f'data/transcripts_v2/{s}/{e}.json'): n+=1
print(n)"; }
echo "[$(date -u +%FT%TZ)] waiting; sandbox episodes without v2: $(remaining)"
while [ "$(remaining)" != "0" ]; do sleep 300; done
echo "[$(date -u +%FT%TZ)] all sandbox episodes re-transcribed; relabelling and rebuilding"
../whisperx-env/bin/python step4_diarize/diarize.py --reassign-only 2>&1 | grep -E "^\[" | cut -c1-120
.venv/bin/python step4_diarize/identify_hosts.py 2>&1 | tail -3
.venv/bin/python step2_sample/build_sandbox.py --bands 5 > /dev/null 2>&1 && echo "[$(date -u +%FT%TZ)] sandbox rebuilt: $(python3 -c "import json; s=json.load(open('sandbox/stats.json')); print('aligned share', s['aligned_share'], '| sources', s['transcript_source'])")"
