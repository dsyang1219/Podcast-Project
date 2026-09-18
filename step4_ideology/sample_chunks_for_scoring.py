#!/usr/bin/env python3
"""Subsample the full chunk file to N chunks per show-quarter.

    ./sample_chunks_for_scoring.py --cap 30 --out data/output/scoring_chunks_cap30.csv

WHY A CHUNK CAP AND NOT AN EPISODE CAP
--------------------------------------
build_scoring_chunks.py caps EPISODES per cell, but a single episode already
yields ~75 chunks at 750 chars, so even one episode overshoots what the estimate
needs. The binding quantity is chunks per show-quarter: the pilot ran a median
of 8 political chunks per cell and the within-show estimator -- the one the
whole panel design rests on -- came back empty at that density, because within
estimation uses quarter-to-quarter movement and that is where the noise lives.
30 clears the >=20 mark where the side-flip rate fell to 8%.

ROUND-ROBIN, NOT TOP-N
----------------------
Chunks within one episode share its topic, so 30 chunks from one episode is a
far thinner sample of a quarter than 5 chunks from each of 6 episodes. Episodes
are visited in `stable_rank_key` order and chunks drawn one per episode per pass
until the cap is met.

FROZEN AND MONOTONE
-------------------
Selection order is md5(SEED:chunk_id), fixed. Raising the cap later yields a
strict SUPERSET of a lower cap, so a partial or cheaper run is never wasted --
top-up scoring only pays for chunks it has not already scored. This mirrors the
guarantee build_scoring_chunks.py makes for episodes.
"""
import argparse, gzip, hashlib, csv, sys
from collections import defaultdict

SEED = "ladder-20260713"          # same frozen seed as sample_ladder.py
csv.field_size_limit(10_000_000)


def key(chunk_id: str) -> str:
    return hashlib.md5(f"{SEED}:{chunk_id}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=30, help="chunks per show-quarter")
    ap.add_argument("--src", default="data/output/scoring_chunks.csv.gz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # cell -> episode -> [chunk rows], each episode's list ordered by frozen key
    cells = defaultdict(lambda: defaultdict(list))
    hdr = None
    n = 0
    with gzip.open(a.src, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            hdr = hdr or list(r)
            cells[(r["collection_id"], r["q"])][r["episode_id"]].append(r)
            n += 1
    print(f"read {n:,} chunks in {len(cells):,} cells", file=sys.stderr)

    picked = []
    for cell, eps in cells.items():
        order = sorted(eps, key=key)                      # episodes, frozen order
        for e in order:
            eps[e].sort(key=lambda r: key(r["chunk_id"]))  # chunks, frozen order
        i, taken = 0, 0
        while taken < a.cap:                               # round-robin across episodes
            progressed = False
            for e in order:
                if i < len(eps[e]):
                    picked.append(eps[e][i]); taken += 1; progressed = True
                    if taken >= a.cap: break
            if not progressed: break
            i += 1

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader(); w.writerows(picked)
    per = len(picked) / max(1, len(cells))
    print(f"wrote {len(picked):,} chunks -> {a.out}  ({per:.1f}/cell)", file=sys.stderr)


if __name__ == "__main__":
    main()
