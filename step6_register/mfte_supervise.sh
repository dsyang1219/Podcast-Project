#!/usr/bin/env bash
# Supervise the MFTE shards (see run_mfte_corpus.sh for staging).
#
# Why this exists: MFTE's Stanza taggers run on the GPU, and their reserved memory creeps upward over hours even
# with bounded input pieces. On rma2 CPU and GPU share one memory pool, so the creep eventually trips the OOM
# killer. MFTE skips files whose tagged output already exists (both stages) and its statistics pass enumerates the
# whole folder, so a tagger can be killed and relaunched at any time and simply resumes. This script therefore:
#   1. for each shard, (re)launches a tagger until <shard>_MFTE/Statistics/counts_raw.csv exists;
#   2. every 2 minutes, if available memory is below MIN_AVAIL_GB, sends TERM to the tagger holding the most GPU
#      memory, which the loop in (1) relaunches lean.
# Taggers already running when this starts are adopted (waited for), not duplicated.
#
#   SHARDS=4 setsid nohup bash step6_register/mfte_supervise.sh > data/output/mfte_supervise.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
SHARDS="${SHARDS:-4}"
MIN_AVAIL_GB="${MIN_AVAIL_GB:-25}"
OUT=data/output/mfte_corpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ts() { date -u +%FT%TZ; }

tagger_pid() {  # pid of the tagger for shard $1, if any (exact command match)
  ps -eo pid,cmd | awk -v p="$OUT/shard$1/" '$2 ~ /python$/ && $4 == "MFTE" && $6 == p {print $1}' | head -1
}

shard_loop() {
  local i="$1" done_file="$OUT/shard${i}_MFTE/Statistics/counts_raw.csv"
  while [ ! -f "$done_file" ]; do
    local pid; pid=$(tagger_pid "$i")
    if [ -n "$pid" ]; then
      echo "[$(ts)] shard$i: adopting running tagger pid=$pid"
      while kill -0 "$pid" 2>/dev/null; do sleep 30; done
      echo "[$(ts)] shard$i: tagger pid=$pid exited"
    else
      echo "[$(ts)] shard$i: launching tagger"
      .venv/bin/python -m MFTE --path "$OUT/shard$i/" --parallel_md_tagging True >> "$OUT/shard$i.log" 2>&1
      echo "[$(ts)] shard$i: tagger exited rc=$?"
    fi
    [ -f "$done_file" ] || sleep 15
  done
  echo "[$(ts)] shard$i: FINISHED ($done_file)"
  touch "$OUT/shard$i.finished"
}

for i in $(seq 0 $((SHARDS-1))); do shard_loop "$i" & done

# Memory guard.
while true; do
  n_left=0
  for i in $(seq 0 $((SHARDS-1))); do [ -f "$OUT/shard${i}_MFTE/Statistics/counts_raw.csv" ] || n_left=$((n_left+1)); done
  [ "$n_left" -eq 0 ] && break
  avail=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$avail" -lt "$MIN_AVAIL_GB" ]; then
    # our tagger pids, ordered by GPU memory; kill the largest
    victim=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | sort -t, -k2 -n -r | awk -F, '{gsub(/ /,"",$1); print $1}' \
      | while read -r p; do ps -o cmd= -p "$p" 2>/dev/null | grep -q "MFTE --path $OUT/shard" && { echo "$p"; break; }; done)
    if [ -n "$victim" ]; then
      echo "[$(ts)] memory guard: ${avail} GB available < ${MIN_AVAIL_GB}; TERM to tagger pid=$victim ($(ps -o cmd= -p "$victim" | grep -oE 'shard[0-9]+'))"
      kill "$victim"
      sleep 120
    fi
  fi
  sleep 120
done
wait
echo "[$(ts)] all $SHARDS shards finished"
