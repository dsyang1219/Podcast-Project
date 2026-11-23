#!/usr/bin/env bash
# Corpus diarization: default variant (community-1, stride 3, active slots, fp16)
# over everything in the audio archive, repeated until the archive re-download
# has finished and nothing is left, then host identification + host-only text.
set -u
cd "$(dirname "$0")/.."
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
PY=../whisperx-env/bin/python
CAP="${GATE_BELOW_F:-145}"
echo "[$(date -u +%FT%TZ)] corpus run start, gate cap ${CAP}F"
while true; do
  $PY step4_diarize/diarize.py --gate-below-f "$CAP" 2>&1 | grep -E "^\[.*(targets|finished|FAIL|progress)" | cut -c1-180
  archive_running=$(pgrep -f "^\.venv/bin/python step4_diarize/archive_download" >/dev/null && echo 1 || echo 0)
  left=$($PY step4_diarize/diarize.py --dry-run 2>/dev/null | grep -oE "targets: [0-9]+" | grep -oE "[0-9]+")
  echo "[$(date -u +%FT%TZ)] pass done; archive_running=$archive_running remaining=${left:-?}"
  if [ "$archive_running" = "0" ] && [ "${left:-0}" = "0" ]; then break; fi
  sleep 900
done
echo "[$(date -u +%FT%TZ)] diarization complete; identifying hosts"
.venv/bin/python step4_diarize/identify_hosts.py --write-host-text 2>&1 | tail -8
echo "[$(date -u +%FT%TZ)] corpus run done"
