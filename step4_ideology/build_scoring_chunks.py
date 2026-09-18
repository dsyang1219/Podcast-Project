#!/usr/bin/env python3
"""Re-chunk transcripts to the granularity Much et al. (2026) score at.

    ./build_scoring_chunks.py --plan          # what would be built, write nothing
    ./build_scoring_chunks.py --build         # writes data/output/scoring_chunks.csv.gz
    ./build_scoring_chunks.py --pilot 50      # a small stratified set for validation

Why re-chunk at all
-------------------
The existing chunk files are far coarser than the unit the published pipeline
scores at:

    Much et al. unit      750 characters (hard cap on a speaker turn)
    chunks_300.csv      4,171 chars median
    chunks_500.csv      6,914 chars median
    chunks_800.csv     10,989 chars median

That gap is not cosmetic. Their episode measure is a share:

    net_ideology = share(conservative chunks) - share(liberal chunks)

so the chunk IS the unit of observation, and its size sets what "one
observation" means. A 7,000-character chunk spans several arguments, mixes
liberal and conservative content, and gets scored `moderate`. Feed the same
transcript through at 750 characters and the same material resolves into
separate directional chunks. The index systematically compresses toward zero as
chunk size grows: it is not scale-invariant, and their published validation
(kappa = 0.834 on ideology direction, against a human ceiling of 0.556) was
measured on 750-character units.

Their unit is a diarized SPEAKER TURN, capped at 750 chars and split if longer.
This corpus has no diarization — segments carry start/end/text only — so we cut
fixed windows at sentence boundaries instead. That is a real deviation, and it
is the one thing about this pipeline most in need of checking: the 3,581
episodes with archived audio can be diarized and scored both ways to test
whether their validation transfers. Until that is done, treat the transfer as
assumed rather than established.

Ads are deliberately KEPT
-------------------------
Much et al. score "the entirety of an episode" with no sponsor removal, and
measurement at their granularity says that is defensible here too:

    chunks containing ad markers          2.2%
    ...of those, politically themed        17%   (~0.4% of all chunks)

Their two-stage gate does incidental ad removal: a read for a mattress company
is not political, fails the politics detector, scores 0, and never reaches the
ideology scorer. Only politically-branded sponsors survive, and they are rare.
Note the earlier 13.8% figure for this corpus was measured on 500-WORD chunks,
where an ad read shares a chunk with ordinary show content — the same chunk-size
artifact this script exists to fix.

Keeping them also preserves comparability with their published show-level
scores. `has_ad_marker` is emitted per chunk so the whole analysis can be re-run
with ads dropped as a robustness check without re-scoring anything.

Sampling
--------
Scoring every transcript is unnecessary: the estimand is a show-quarter mean,
and a cell mean rests on ~61 chunks per episode, not on the episode count. Cells
are capped at --cap episodes, drawn in `stable_rank_key` order — the same frozen
hash the transcription ladder uses. That makes the draw reproducible, and makes
a larger cap a strict superset of a smaller one, so the cap can be raised later
without re-scoring or re-drawing anything already done.
"""
import argparse, csv, glob, gzip, hashlib, json, os, re, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repository root (this script lives one folder down)
TRANSCRIPTS = ROOT / "data/transcripts"
EPISODES = ROOT / "data/output/episodes.csv.gz"
OUT = ROOT / "data/output/scoring_chunks.csv.gz"
PILOT = ROOT / "data/output/scoring_chunks_pilot.csv"

SEED = "ladder-20260713"          # must match sample_ladder.py
CAP_CHARS = 750                   # Much et al. footnote 19
SENT_END = re.compile(r"(?<=[.!?])\s+")
AD_MARKER = re.compile(
    r"promo code|use code|dot com slash|percent off|sponsored by|brought to you by|"
    r"this episode is sponsored|free shipping|free trial|go to \w+\.com", re.I)


def stable_rank_key(episode_id: str) -> str:
    """Frozen per-episode sort key, identical to sample_ladder.py."""
    return hashlib.md5(f"{SEED}:{episode_id}".encode()).hexdigest()


def split_750(text: str, cap: int = CAP_CHARS):
    """Exhaustive, non-overlapping chunks of at most `cap` chars.

    Prefers sentence boundaries so a chunk is a coherent unit of argument rather
    than an arbitrary window; a sentence longer than the cap is hard-split so
    that no text is ever dropped. Much et al.: "All speaker chunks are split in
    such a way that no text is missed and the entirety of an episode is scored."
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    out, cur = [], ""
    for sent in SENT_END.split(text):
        if not sent:
            continue
        if len(cur) + len(sent) + 1 <= cap:
            cur = (cur + " " + sent).strip()
            continue
        if cur:
            out.append(cur)
            cur = ""
        while len(sent) > cap:                 # single sentence over the cap
            out.append(sent[:cap])
            sent = sent[cap:]
        cur = sent
    if cur:
        out.append(cur)
    return out


def episode_frame():
    """Episode metadata keyed on the same md5(audio_url)[:16] the corpus uses."""
    eps = pd.read_csv(EPISODES, compression="gzip", low_memory=False)
    eps["pub_date"] = pd.to_datetime(eps.pub_date, utc=True, errors="coerce", format="mixed")
    eps = eps[eps.pub_date.notna() & (eps.pub_date > "2000-01-01")]
    eps = eps[eps.audio_url.notna() & (eps.audio_url != "")].copy()
    eps["episode_id"] = eps.audio_url.map(
        lambda u: hashlib.md5(u.encode()).hexdigest()[:16])
    eps = eps.drop_duplicates(subset=["collection_id", "episode_id"])
    eps["q"] = eps.pub_date.dt.tz_localize(None).dt.to_period("Q").astype(str)
    return eps[["collection_id", "show_name", "episode_id", "episode_title", "pub_date", "q"]]


def select(args):
    """Episodes to score: on disk, in window, capped per show-quarter."""
    on_disk = {}
    for p in glob.glob(str(TRANSCRIPTS / "*/*.json")):
        on_disk[os.path.basename(p).split("_", 1)[0]] = p
    eps = episode_frame()
    eps = eps[eps.episode_id.isin(on_disk)].copy()
    if args.since:
        eps = eps[eps.q >= args.since]
    eps["_key"] = eps.episode_id.map(stable_rank_key)
    eps = eps.sort_values(["collection_id", "q", "_key"])
    if args.cap:
        eps = eps.groupby(["collection_id", "q"], group_keys=False).head(args.cap)
    eps["path"] = eps.episode_id.map(on_disk)
    return eps


def transcript_text(path):
    with open(path) as f:
        d = json.load(f)
    return " ".join(s.get("text", "") for s in d.get("segments", []))


def cmd_plan(args):
    eps = select(args)
    cells = eps.groupby(["collection_id", "q"]).size()
    print(f"episodes to score : {len(eps):,}")
    print(f"show-quarter cells: {len(cells):,}   shows: {eps.collection_id.nunique()}")
    print(f"  episodes/cell   : median {cells.median():.0f}  p90 {cells.quantile(.9):.0f}  max {cells.max()}")
    print(f"  window          : {eps.q.min()} .. {eps.q.max()}")
    print(f"\nestimated chunks at {CAP_CHARS} chars: ~{len(eps) * 61 / 1e6:.2f}M")
    print(f"  stage 1 calls   : that many")
    print(f"  stage 2 calls   : ~80% of them (this corpus is entirely political-genre;")
    print(f"                    Much et al. measured 81% of utterances political in that stratum)")


def cmd_build(args):
    eps = select(args)
    out_path = PILOT if args.pilot else OUT
    if args.pilot:
        eps = eps.sort_values("_key").head(args.pilot)
    print(f"chunking {len(eps):,} episodes -> {out_path}\n")
    cols = ["chunk_id", "episode_id", "collection_id", "show_name", "q", "pub_date",
            "episode_title", "chunk_ix", "n_chars", "n_words", "has_ad_marker",
            "context", "text"]
    opener = (lambda p: open(p, "w", newline="")) if args.pilot else \
             (lambda p: gzip.open(p, "wt", newline=""))
    n_chunks = n_ep = 0
    dropped = 0
    with opener(out_path) as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(eps.itertuples(), 1):
            try:
                text = transcript_text(r.path)
            except Exception as e:
                dropped += 1
                continue
            parts = split_750(text)
            # exhaustiveness check: chunking must not lose characters
            if parts:
                joined = len(re.sub(r"\s+", "", "".join(parts)))
                orig = len(re.sub(r"\s+", "", text))
                assert joined == orig, f"{r.episode_id}: lost {orig - joined} chars"
            for j, t in enumerate(parts):
                # Much et al. pass the PRECEDING speaker turn as {CONTEXT}. Without
                # it the model sees an isolated 750-char window and reads content
                # literally: it scored Posobiec MOCKING an anti-Trump position as
                # strongly_liberal, and "that's a very suburban liberal thing to do"
                # as liberal. Stance is not recoverable from the window alone.
                ctx = parts[j - 1][-400:] if j > 0 else ""
                w.writerow({
                    "context": ctx,
                    "chunk_id": f"{r.episode_id}_{j}", "episode_id": r.episode_id,
                    "collection_id": r.collection_id, "show_name": r.show_name,
                    "q": r.q, "pub_date": r.pub_date.date().isoformat(),
                    "episode_title": r.episode_title, "chunk_ix": j,
                    "n_chars": len(t), "n_words": len(t.split()),
                    "has_ad_marker": int(bool(AD_MARKER.search(t))), "text": t,
                })
            n_chunks += len(parts); n_ep += 1
            if i % 500 == 0:
                print(f"  {i:>6}/{len(eps)} episodes  {n_chunks:,} chunks", flush=True)
    print(f"\n{out_path}")
    print(f"  {n_ep:,} episodes -> {n_chunks:,} chunks  ({n_chunks/max(n_ep,1):.0f}/episode)")
    if dropped:
        print(f"  {dropped} transcripts unreadable and skipped")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--pilot", type=int, default=0,
                    help="build only N episodes, for validation")
    ap.add_argument("--since", default="2018Q1", help="earliest quarter (default 2018Q1)")
    ap.add_argument("--cap", type=int, default=10,
                    help="max episodes scored per show-quarter (default 10)")
    a = ap.parse_args()
    if a.plan: cmd_plan(a)
    elif a.build or a.pilot: cmd_build(a)
    else: ap.print_help()


if __name__ == "__main__":
    main()
