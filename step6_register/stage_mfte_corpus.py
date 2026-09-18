"""Stage transcripts for MFTE in bounded pieces.

MFTE runs Stanza on the GPU, and Stanza's peak memory grows with the longest document it has processed. Whole
episode transcripts (up to ~50,000 words) pushed four taggers past the machine's shared CPU/GPU memory. Tag counts
are additive, so each transcript is written out in pieces of at most MAX_WORDS words, cut at sentence ends, named

    <show_id>__<episode_id>__p<k>.txt

and step6_register/mfte_address.py sums the pieces back to one row per episode. Pieces are dealt round-robin into
SHARDS folders under data/output/mfte_corpus/ so the shards have equal word counts.

    .venv/bin/python step6_register/stage_mfte_corpus.py --shards 4 --max-words 3000
"""

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENT_END = re.compile(r"(?<=[.!?])\s+")

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=4)
ap.add_argument("--max-words", type=int, default=3000)
ap.add_argument("--out", default=str(ROOT / "data/output/mfte_corpus"))
a = ap.parse_args()

out = Path(a.out)
for i in range(a.shards):
    (out / f"shard{i}").mkdir(parents=True, exist_ok=True)

files = sorted((ROOT / "data/transcripts").glob("*/*.txt"))
n_files = n_pieces = n_words = 0
k = 0
for f in files:
    show = f.parent.name
    episode = f.name.split("_", 1)[0]
    text = f.read_text(errors="replace")
    words = text.split()
    n_files += 1
    n_words += len(words)
    if not words:
        continue
    # cut at sentence ends into pieces of at most max_words words
    pieces, cur, cur_n = [], [], 0
    for sent in SENT_END.split(text):
        sw = len(sent.split())
        if cur_n + sw > a.max_words and cur:
            pieces.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(sent)
        cur_n += sw
    if cur:
        pieces.append(" ".join(cur))
    for j, p in enumerate(pieces):
        (out / f"shard{k % a.shards}" / f"{show}__{episode}__p{j}.txt").write_text(p)
        k += 1
        n_pieces += 1
(out / "staged.ok").write_text(f"{n_files} transcripts, {n_pieces} pieces, {n_words} words\n")
print(f"{n_files} transcripts -> {n_pieces} pieces ({n_words/1e6:.0f}M words) in {a.shards} shards")
