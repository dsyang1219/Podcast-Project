#!/usr/bin/env bash
# Pause the archive encoders, profile pipeline variants on one episode, resume.
# The trap guarantees the resume even if this script is killed.
set -u
cd "$(dirname "$0")/.."
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
AP=$(pgrep -f "^\.venv/bin/python step4_diarize/archive_download" || true)
resume() { pkill -CONT -f "ffmpeg" 2>/dev/null; if [ -n "$AP" ]; then for c in $(pgrep -P $AP); do kill -CONT $c 2>/dev/null; done; kill -CONT $AP 2>/dev/null; fi; echo "[$(date -u +%T)] archive job resumed"; }
trap resume EXIT INT TERM
if [ -n "$AP" ]; then kill -STOP $AP; for c in $(pgrep -P $AP); do kill -STOP $c; done; pkill -STOP -f "nice -n 15 ffmpeg"; echo "[$(date -u +%T)] archive job paused"; fi
sleep 3; uptime | grep -oE "load average.*"
SHOW=${1:-1043245989}; EP=${2:-0f0caa45c5cf553c}
for v in "--stride 0.3" "--stock" "--model pyannote/speaker-diarization-3.1 --stride 0.3" "--model pyannote/speaker-diarization-3.1 --stock"; do
  echo "=== $v"
  ../whisperx-env/bin/python step4_diarize/profile_stages.py "$SHOW" "$EP" $v 2>&1 | grep -vE "OneLogger|exporters|Warning|warn|std\(|^\s*$|NeMo|correction|See https|^    "
done
echo "[$(date -u +%T)] profile done"
