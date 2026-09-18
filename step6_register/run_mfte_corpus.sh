#!/usr/bin/env bash
# Tag every episode transcript with MFTE (Multi-Feature Tagger of English, Le Foll 2021; Python port by Shakir)
# in N parallel shards. Output of interest: <shard>_MFTE/Statistics/counts_raw.csv, one row per episode with raw
# tag counts (VIMP = imperatives, PP2 = second-person pronouns, Words = token count). step6_register/mfte_address.py
# turns those into show-level rates and the published-tool address measure.
#
#   bash step6_register/run_mfte_corpus.sh            # 6 shards (default)
#   SHARDS=8 bash step6_register/run_mfte_corpus.sh
#
# Transcripts are written into data/output/mfte_corpus/shard<i>/ as <show_id>__<episode_id>__p<k>.txt pieces
# (stage_mfte_corpus.py) so both ids survive in MFTE's Filename column and the pieces can be summed per episode.
# Each shard writes shard<i>.log and touches shard<i>.done when finished.
set -uo pipefail
cd "$(dirname "$0")/.."
SHARDS="${SHARDS:-4}"
MAX_WORDS="${MAX_WORDS:-3000}"
OUT=data/output/mfte_corpus
mkdir -p "$OUT"
# MFTE's Stanza pipeline runs on the GPU. On rma2 CPU and GPU share one 121 GB memory pool, so: never run this
# while the transcription worker is up, keep SHARDS at 4 or fewer, and feed bounded pieces (stage_mfte_corpus.py)
# so no tagger's peak memory runs away on a five-hour episode.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -f "$OUT/staged.ok" ]; then
  echo "staging transcripts in pieces of <= $MAX_WORDS words into $SHARDS shards..."
  .venv/bin/python step6_register/stage_mfte_corpus.py --shards "$SHARDS" --max-words "$MAX_WORDS" --out "$OUT"
fi
for i in $(seq 0 $((SHARDS-1))); do
  [ -f "$OUT/shard$i.done" ] && { echo "shard$i already done"; continue; }
  ( .venv/bin/python -m MFTE --path "$OUT/shard$i/" --parallel_md_tagging True > "$OUT/shard$i.log" 2>&1; \
    echo "rc=$?" >> "$OUT/shard$i.log"; touch "$OUT/shard$i.done" ) &
  echo "shard$i started pid=$!"
done
wait
echo "all shards finished $(date -u +%FT%TZ)"
