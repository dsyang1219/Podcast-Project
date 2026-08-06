"""Sample candidate windows for the hand-annotated gold span set.

This script does NOT label anything. It samples transcript windows across
strata and dumps them, IN FULL, for a human to read and annotate. Marker
regexes (nlp/adspan_common.AD_MARKERS etc.) are used only to make sure the
sample actually contains ads, meta, and ambiguous product-talk rather than the
~90% pure political content a uniform sample would return. They never assign a
label -- if they could, there would be no reason to run an LLM at all.

Reading the FULL window is the point. The previous chunk-level gold set
(data/output/adclf/gold_120.csv) was labeled from ~220-character excerpts and
had to be thrown away and redone, because ad copy routinely sits past where the
excerpt stopped.

    .venv/bin/python -m nlp.adspan_gold_candidates --n-episodes 400

Writes data/output/adspan/gold_candidates.json (all scored candidates) and
gold_candidates_readable.txt (the sampled ones, full text, for annotation).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import pandas as pd

from pipeline import config as pipeline_config
from .adspan_common import (
    build_windows,
    marker_counts,
    segments_to_text,
    sentences_for_texts,
)

TRANSCRIPTS_DIR = pipeline_config.DATA_DIR / "transcripts"
OUT_DIR = pipeline_config.OUTPUT_DIR / "adspan"

# How many windows to sample per stratum for hand annotation.
STRATUM_QUOTA = {
    "host_read_sponsor": 7,
    "network_dynamic_insert": 4,
    "meta_boilerplate": 4,
    "ambiguous_product": 6,
    "pure_content": 5,
}

# Phrases characteristic of programmatically inserted / network ad reads as
# opposed to a host improvising a sponsor read. Only used for stratification.
NETWORK_PHRASES = [
    "support for this podcast", "support for this show", "this message comes from",
    "is supported by", "sponsored by", "brought to you by", "paid partnership",
]

POLITICAL_TERMS = [
    "president", "congress", "senate", "democrat", "republican", "election",
    "vote", "policy", "court", "bill", "administration", "campaign",
    "immigration", "foreign", "war", "government", "law",
]


def stratify(window_text: str, offset: int, n_sentences_total: int) -> str | None:
    """Assign a sampling stratum, or None if the window is uninformative."""
    lower = window_text.lower()
    m = marker_counts(window_text)
    is_network = any(p in lower for p in NETWORK_PHRASES)
    pol_hits = sum(lower.count(t) for t in POLITICAL_TERMS)
    near_edge = offset < 40 or offset > n_sentences_total - 60

    if m["ad"] >= 3 and not is_network:
        return "host_read_sponsor"
    if is_network and m["ad"] >= 1:
        return "network_dynamic_insert"
    if m["meta"] >= 2 and m["ad"] == 0 and near_edge:
        return "meta_boilerplate"
    if m["ambiguous"] >= 3 and m["ad"] == 0 and m["meta"] == 0 and pol_hits >= 3:
        return "ambiguous_product"
    if m["ad"] == 0 and m["meta"] == 0 and pol_hits >= 12:
        return "pure_content"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-episodes", type=int, default=400,
                     help="how many episodes to scan for candidates")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-process", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    corpus = pd.read_csv(pipeline_config.OUTPUT_DIR / "corpus.csv",
                          usecols=["collection_id", "show_name"])
    show_names = {str(c): n for c, n in zip(corpus.collection_id, corpus.show_name)}

    paths = sorted(TRANSCRIPTS_DIR.glob("*/*.json"))
    # Spread the scan across shows rather than taking a contiguous block, so
    # candidates aren't dominated by whichever shows happen to sort first.
    by_show: dict[str, list] = defaultdict(list)
    for p in paths:
        by_show[p.parent.name].append(p)
    picked: list = []
    shows = sorted(by_show)
    rng.shuffle(shows)
    per_show = max(1, args.n_episodes // max(1, len(shows)))
    for s in shows:
        eps = by_show[s]
        rng.shuffle(eps)
        picked.extend(eps[:per_show])
    rng.shuffle(picked)
    picked = picked[:args.n_episodes]
    print(f"[scan] {len(picked)} episodes across {len({p.parent.name for p in picked})} shows")

    metas, texts = [], []
    for p in picked:
        d = json.loads(p.read_text())
        t = segments_to_text(d["segments"])
        if not t:
            continue
        metas.append({"episode_id": d["episode_id"], "show_id": str(d["show_id"]),
                       "episode_title": d.get("episode_title", "")})
        texts.append(t)

    print(f"[sentencize] {len(texts)} episodes (n_process={args.n_process})...")
    sent_lists = sentences_for_texts(texts, n_process=args.n_process)

    candidates = []
    for meta, sents in zip(metas, sent_lists):
        for w in build_windows(sents, meta["episode_id"], meta["show_id"]):
            wt = " ".join(w.sentences)
            stratum = stratify(wt, w.offset, len(sents))
            if stratum is None:
                continue
            candidates.append({
                "window_id": w.window_id,
                "episode_id": meta["episode_id"],
                "show_id": meta["show_id"],
                "show_name": show_names.get(meta["show_id"], ""),
                "episode_title": meta["episode_title"],
                "stratum": stratum,
                "offset": w.offset,
                "n_sentences": len(w.sentences),
                "sentences": w.sentences,
                "marker_counts": marker_counts(wt),
            })

    counts = defaultdict(int)
    for c in candidates:
        counts[c["stratum"]] += 1
    print(f"[candidates] {len(candidates)} total: {dict(counts)}")

    # Sample within stratum, at most one window per SHOW per stratum so no
    # single show's house style dominates a stratum.
    sampled = []
    for stratum, quota in STRATUM_QUOTA.items():
        pool = [c for c in candidates if c["stratum"] == stratum]
        rng.shuffle(pool)
        seen_shows = set()
        for c in pool:
            if len(seen_shows) >= quota:
                break
            if c["show_id"] in seen_shows:
                continue
            seen_shows.add(c["show_id"])
            sampled.append(c)
        got = sum(1 for s in sampled if s["stratum"] == stratum)
        print(f"  {stratum:<24} sampled {got}/{quota} (pool={len(pool)})")

    (OUT_DIR / "gold_candidates.json").write_text(json.dumps(sampled, indent=2))

    lines = []
    for i, c in enumerate(sampled):
        lines.append("=" * 100)
        lines.append(f"WINDOW {i}  id={c['window_id']}  stratum={c['stratum']}")
        lines.append(f"show={c['show_name']!r} ({c['show_id']})  episode={c['episode_title']!r}")
        lines.append(f"offset={c['offset']}  n_sentences={c['n_sentences']}  markers={c['marker_counts']}")
        lines.append("-" * 100)
        for j, s in enumerate(c["sentences"]):
            lines.append(f"[{j}] {s}")
        lines.append("")
    (OUT_DIR / "gold_candidates_readable.txt").write_text("\n".join(lines))
    print(f"\n[out] {len(sampled)} windows -> {OUT_DIR / 'gold_candidates.json'}")
    print(f"[out] full text for annotation -> {OUT_DIR / 'gold_candidates_readable.txt'}")


if __name__ == "__main__":
    main()
