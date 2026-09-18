#!/usr/bin/env bash
# Supervisor for the ladder transcription run (~27 GPU-days).
#
# Runs one downloader and $TRANSCRIBERS GPU workers, restarting any that die.
# Every half is
# safely resumable, which is what makes blind restart correct rather than
# reckless:
#   - download_audio.py rebuilds its skip sets from disk each start, so already
#     queued/transcribed episodes are skipped, and it sweeps orphaned .part files.
#   - transcribe.py compares audio_sha256 against any existing transcript before
#     working, and reclaims stale .meta.claimed files left by a killed worker.
#
# Without supervision a 27-day run stalls silently: if the downloader dies the
# transcriber drains the queue and then waits forever (no STOP was written); if
# the transcriber dies the downloader parks against backpressure. Neither exits,
# so nothing tells you it stopped making progress.
#
# Status lands in data/output/ladder_status.json on every state change.
set -uo pipefail
cd "$(dirname "$0")/.."   # step3_transcribe/ -> repository root; all paths below are root-relative
mkdir -p step3_transcribe/logs

MANIFEST="${MANIFEST:-data/output/sample_out/sample_manifest_ladder.csv}"
MODEL="${MODEL:-deepdml/faster-whisper-large-v3-turbo-ct2}"
# Back to 64 to cut sustained power draw (and so heat/fan noise) — 128 held the
# GPU busier in longer bursts, 51W vs 38W, and pushed a worker's unified-memory
# footprint from ~10 GB to ~15 GB.
#
# What 128 was worth, if you want it back: benchmarked 1.24x throughput at a
# 0.00% word error rate vs batch=64 — identical transcripts, so it costs no
# quality, only watts. 256 was not better (0.99x), so 128 is the sweet spot.
# int8_float16 is dominated: slower than fp16/128 (1.10x) *and* it shifts 0.76%
# of words, so precision stays float16 regardless.
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-16}"
MAX_QUEUED="${MAX_QUEUED:-250}"
# Concurrent GPU transcribers. A single worker leaves the GPU idle between
# episodes — audio decode, VAD, JSON write and gc all run GPU-idle, measured as
# a ~50% duty cycle — so a second process fills those gaps. Safe because the
# .meta -> .meta.claimed rename is atomic: two workers can never claim the same
# episode. Each worker cost ~10 GB of GPU memory at BATCH_SIZE=64; budget more
# at 128, against ~121 GB of unified memory on this box.
# Set to 1 deliberately: two workers measured ~1.20x throughput (105x -> 127x
# audio-realtime) but cost a second copy of GPU memory, ~15 GB at BATCH_SIZE=128
# on this box's unified memory. Raise to 2 to trade that memory back for speed.
TRANSCRIBERS="${TRANSCRIBERS:-1}"
# Gap between worker launches so N model loads don't collide.
TR_STAGGER="${TR_STAGGER:-20}"
# Cap the share of GPU SMs a worker may use, via CUDA MPS (needs the MPS control
# daemon running; see start_mps below). Empty = uncapped.
#
# This is the good way to run cooler. SIGSTOP duty-cycling throws away wall-clock
# at full power, so it costs ~1% throughput per 1% of heat. Capping occupancy
# instead keeps the GPU working continuously at lower power, and since power
# scales faster than linearly with utilisation, it sheds more heat than speed.
# GB10 exposes no power limit and clock locking needs root, so this is the only
# continuous lever available unprivileged.
GPU_THREAD_PCT="${GPU_THREAD_PCT:-}"
# Hold this average GPU temperature by idling between episodes (empty = uncapped).
# Preferred over gpu_thermal_governor.sh: that reacts to instantaneous readings
# from a 1Hz sensor on a die that heats ~13F/s, so it overshoots 10-20F and spends
# its pauses recovering — measured at 31% duty for a 129F average. This holds the
# average directly, and cannot leave a worker SIGSTOPped if it dies.
TARGET_TEMP_F="${TARGET_TEMP_F:-}"
[ -n "$TARGET_TEMP_F" ] && TR_TEMP_FLAG="--target-temp-f $TARGET_TEMP_F" || TR_TEMP_FLAG=""
# Wait for the die to fall to this temperature before STARTING each episode.
# Prefer this to TARGET_TEMP_F. Measured over 2,770 burst onsets on this box, the
# peak a burst reaches is set by where it starts (0.79F of peak per 1F of start,
# r=0.63) and barely by how long it runs (9F across the whole length range). So
# gating the start is the only thing that moves the peak; the duty controller
# holds an average and leaves peaks at 170F. It is also cheaper: reaching 125F
# after a burst took a median of 10s, giving ~74% duty against the duty
# controller's measured 5-14%.
START_BELOW_F="${START_BELOW_F:-}"
[ -n "$START_BELOW_F" ] && TR_GATE_FLAG="--start-below-f $START_BELOW_F" || TR_GATE_FLAG=""
# Preferred over START_BELOW_F: gate this many degrees above the *measured* idle
# floor rather than at an absolute temperature. The floor is not a constant —
# 118F to 136F across one day here as another tenant's CPU load ramped — so a
# fixed gate that is cheap in the morning is underwater by afternoon. Measured:
# fixed 125F ran 12-16% duty, fixed 130F ran 6%, both because they sat below the
# floor and every episode paid the full wait. A margin of 2 tracks it instead.
START_ABOVE_FLOOR="${START_ABOVE_FLOOR:-}"
[ -n "$START_ABOVE_FLOOR" ] && TR_GATE_FLAG="--start-above-floor $START_ABOVE_FLOOR"
# Pre-decode audio to 16 kHz mono WAV on the downloader's idle CPU rather than
# inside the GPU worker. Set PREDECODE=0 to disable (roughly a third the queue
# disk, but the GPU worker pays ~25s of container decode per episode).
# Off by default now. Measured: pre-decoding is worth ~3% throughput (127x -> 130x
# audio-realtime), but its up-to-6 concurrent ffmpeg transcodes are CPU work on the
# same package as the GPU. Transcription alone peaks ~135F; with the encoders it
# reached 145-162F. That is a bad trade when the goal is a cooler machine — the
# thermal cost is far larger than the 3% it buys.
# Keep a compressed copy of each episode's audio after transcription. Off means
# the audio is unlinked, so any later diarization pass needs a re-download from
# origins that already rate-limit us. On costs ~0.8 cores of continuous Opus
# encoding on the same package as the GPU — whether that is worth real degrees
# is what ARCHIVE=0 exists to measure.
ARCHIVE="${ARCHIVE:-1}"
[ "$ARCHIVE" = "0" ] && TR_ARCHIVE_FLAG="--no-archive-audio" || TR_ARCHIVE_FLAG=""
PREDECODE="${PREDECODE:-0}"
[ "$PREDECODE" = "1" ] && DL_PREDECODE_FLAG="" || DL_PREDECODE_FLAG="--no-predecode"

VENV="${VENV:-.venv/bin/python}"
# Relative default works on rma2, where the corpus sits directly in $HOME next to
# whisperx-env. Overridable because rma1 keeps the corpus one level deeper (under
# Podcast-Project/), so the relative path would resolve to a directory that does
# not exist there. Absolute is safest when driving a second box.
GPU_VENV="${GPU_VENV:-../whisperx-env/bin/python}"

DL_LOG="step3_transcribe/logs/ladder_download.log"
TR_LOG="step3_transcribe/logs/ladder_transcribe.log"
STATUS="data/output/ladder_status.json"
RESTART_SLEEP=30

write_status() {
  "$VENV" - "$1" "$2" "$3" <<'PY'
import datetime, json, os, subprocess, sys
state, dl_pid, tr_pids = sys.argv[1], sys.argv[2], sys.argv[3].split()
done = 0
for root, dirs, files in os.walk("data/transcripts"):
    done += sum(1 for f in files if f.endswith(".json"))
queued = len([f for f in os.listdir("data/audio_queue") if f.endswith(".meta")]) \
    if os.path.isdir("data/audio_queue") else 0
# .claimed files are episodes a worker is transcribing right now. Reported so a
# restart can tell "queue drained" from "workers still chewing on the tail".
claimed = len([f for f in os.listdir("data/audio_queue") if f.endswith(".claimed")]) \
    if os.path.isdir("data/audio_queue") else 0
json.dump({"state": state, "download_pid": dl_pid, "transcribe_pids": tr_pids,
           "n_transcribers": len([p for p in tr_pids if p != "-1"]),
           "transcripts_on_disk": done, "queue_depth": queued, "in_flight": claimed,
           "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()},
          open("data/output/ladder_status.json", "w"), indent=2)
PY
}

start_downloader() {
  "$VENV" step3_transcribe/download_audio.py \
      --manifest "$MANIFEST" --workers "$WORKERS" --max-queued "$MAX_QUEUED" \
      $DL_PREDECODE_FLAG \
      >> "$DL_LOG" 2>&1 &
  DL_PID=$!
  echo "[$(date -u +%H:%M:%S)] downloader started pid=$DL_PID" >> "$DL_LOG"
}

start_mps() {
  # Idempotent: the control daemon ignores a second -d if one is already up.
  [ -z "$GPU_THREAD_PCT" ] && return 0
  command -v nvidia-cuda-mps-control >/dev/null || {
    echo "[$(date -u +%H:%M:%S)] GPU_THREAD_PCT set but MPS not installed — ignoring" >> "$TR_LOG"
    GPU_THREAD_PCT=""; return 0; }
  export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps-$USER}"
  export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log-$USER}"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if ! echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
    nvidia-cuda-mps-control -d >> "$TR_LOG" 2>&1 || true
    sleep 2
  fi
  echo "[$(date -u +%H:%M:%S)] MPS active, capping workers to ${GPU_THREAD_PCT}% of SMs" >> "$TR_LOG"
}

# All workers share one log. Each line is a single flushed write, so O_APPEND
# keeps them intact and `grep -c 'Done:'` still counts the whole run.
start_transcriber() {
  local idx="$1"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="$GPU_THREAD_PCT" \
  "$GPU_VENV" step3_transcribe/transcribe.py \
      --model "$MODEL" --batch-size "$BATCH_SIZE" $TR_TEMP_FLAG $TR_GATE_FLAG $TR_ARCHIVE_FLAG \
      >> "$TR_LOG" 2>&1 &
  TR_PIDS[$idx]=$!
  echo "[$(date -u +%H:%M:%S)] transcriber $idx started pid=${TR_PIDS[$idx]}" >> "$TR_LOG"
}

# Space-joined for write_status; "${TR_PIDS[*]}" would need IFS juggling.
tr_pid_list() { echo "${TR_PIDS[@]}"; }

cleanup() {
  echo "shutting down..." >&2
  kill "$DL_PID" "${TR_PIDS[@]}" 2>/dev/null
  write_status "stopped" "$DL_PID" "$(tr_pid_list)"
  exit 0
}
trap cleanup INT TERM

TR_PIDS=()
start_mps
start_downloader
for ((i = 0; i < TRANSCRIBERS; i++)); do
  [ "$i" -gt 0 ] && sleep "$TR_STAGGER"
  start_transcriber "$i"
done
write_status "running" "$DL_PID" "$(tr_pid_list)"

while true; do
  sleep 30

  if ! kill -0 "$DL_PID" 2>/dev/null; then
    wait "$DL_PID"; rc=$?
    if [ "$rc" -eq 0 ]; then
      # dispatched every row and wrote STOP; let the transcriber drain.
      echo "[$(date -u +%H:%M:%S)] downloader finished cleanly" >> "$DL_LOG"
      DL_PID=-1
    else
      echo "[$(date -u +%H:%M:%S)] downloader died rc=$rc — restarting" >> "$DL_LOG"
      write_status "restarting-downloader" "$DL_PID" "$(tr_pid_list)"
      sleep "$RESTART_SLEEP"
      start_downloader
    fi
  fi

  # Each worker is supervised on its own: one dying doesn't disturb the others,
  # and a worker that exits 0 while the downloader is still running means the
  # queue momentarily ran dry, not that the run is over — so it gets restarted.
  for i in "${!TR_PIDS[@]}"; do
    pid="${TR_PIDS[$i]}"
    [ "$pid" -eq -1 ] && continue
    kill -0 "$pid" 2>/dev/null && continue

    wait "$pid"; rc=$?
    if [ "$rc" -eq 0 ] && [ "$DL_PID" -eq -1 ]; then
      echo "[$(date -u +%H:%M:%S)] transcriber $i drained queue" >> "$TR_LOG"
      TR_PIDS[$i]=-1
      continue
    fi
    echo "[$(date -u +%H:%M:%S)] transcriber $i exited rc=$rc — restarting" >> "$TR_LOG"
    write_status "restarting-transcriber-$i" "$DL_PID" "$(tr_pid_list)"
    sleep "$RESTART_SLEEP"
    start_transcriber "$i"
  done

  # Complete only once every worker has drained and the downloader is finished.
  live=0
  for pid in "${TR_PIDS[@]}"; do
    [ "$pid" -ne -1 ] && live=$((live + 1))
  done
  if [ "$live" -eq 0 ] && [ "$DL_PID" -eq -1 ]; then
    echo "[$(date -u +%H:%M:%S)] all transcribers drained — run complete" >> "$TR_LOG"
    write_status "complete" "-1" "-1"
    exit 0
  fi

  write_status "running" "$DL_PID" "$(tr_pid_list)"
done
