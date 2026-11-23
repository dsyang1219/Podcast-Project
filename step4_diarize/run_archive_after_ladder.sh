#!/usr/bin/env bash
# Wait for the transcription ladder to stop, then re-download the audio of every
# already-transcribed episode into the Opus archive (see archive_download.py).
#
# Why wait: the archive run shares origins with the ladder's own downloader and
# its ffmpeg encoders are CPU work on the same package as the GPU, which the
# ladder's thermal gate would pay for in throughput.
#
#   setsid nohup bash step4_diarize/run_archive_after_ladder.sh \
#       > step4_diarize/logs/run_archive.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
STATUS=data/output/ladder_status.json
WORKERS="${WORKERS:-12}"
ENCODERS="${ENCODERS:-8}"
mkdir -p step4_diarize/logs

state() { python3 -c "import json,sys; print(json.load(open('$STATUS')).get('state','?'))" 2>/dev/null || echo "?"; }

echo "[$(date -u +%FT%TZ)] waiting for ladder to stop (state=$(state))"
while [ "$(state)" = "running" ] && pgrep -f "^bash step3_transcribe/run_ladder.sh" >/dev/null; do
  sleep 600
done
echo "[$(date -u +%FT%TZ)] ladder state=$(state); starting archive download"
exec .venv/bin/python step4_diarize/archive_download.py --workers "$WORKERS" --encoders "$ENCODERS"
