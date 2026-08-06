"""Phase C step 1 -- re-segment the ad-cleaned transcripts into ~500-content-word
passages.

This is deliberately the SAME segmentation logic as nlp/run_chunks.py, reusing
the same functions (build_boilerplate_index, clean_episodes, chunk_sentences)
with the same parameters (target 500 content words, min stub 300, nolemma,
min_token_len 2). The ONLY difference is the input: episode text with the ad and
meta spans already excised. Nothing about how a passage is formed changes, so
any downstream difference is attributable to the excision and not to a second
change in segmentation.

The cross-show boilerplate n-gram filter is REBUILT on the cleaned corpus rather
than reused. It has to be: it flags n-grams recurring across >=4 shows, and the
Phase B excision already removed much of what fed it. Rebuilding keeps it doing
what it always did (catch residual cross-show ad copy) instead of silently
carrying over an index computed against text that no longer exists.

    .venv/bin/python -m nlp.adspan_rechunk
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from pipeline import config as pipeline_config
from . import config as nlp_config
from .adspan_phase_b import CLEANED_PATH, OUT_DIR
from .chunk import chunk_sentences
from .clean import CleanStats, build_boilerplate_index, clean_episodes, load_nlp

CHUNK_FIELDS = ["show", "collection_id", "episode_id", "episode_title",
                "chunk_id", "text", "n_words", "speakers", "clean_text"]
OUT_PATH = OUT_DIR / "chunks_500_nolemma_adclean.csv"


def load_cleaned() -> list[dict]:
    out = []
    with gzip.open(CLEANED_PATH, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d["text"].strip():
                out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-words", type=int, default=500)
    ap.add_argument("--min-stub-words", type=int, default=300)
    ap.add_argument("--n-process", type=int, default=10)
    args = ap.parse_args()

    corpus = pd.read_csv(pipeline_config.OUTPUT_DIR / "corpus.csv",
                          usecols=["collection_id", "show_name"])
    show_names = {str(c): n for c, n in zip(corpus.collection_id, corpus.show_name)}

    episodes = load_cleaned()
    print(f"[load] {len(episodes)} cleaned episodes across "
          f"{len({e['show_id'] for e in episodes})} shows")

    show_texts: dict[str, list[str]] = defaultdict(list)
    for e in episodes:
        show_texts[e["show_id"]].append(e["text"])
    boilerplate = build_boilerplate_index(show_texts)
    print(f"[boilerplate] {len(boilerplate)} cross-show n-grams flagged on the "
          f"CLEANED corpus (rebuilt, not reused)")

    nlp = load_nlp()
    print(f"[clean] parsing {len(episodes)} episodes (n_process={args.n_process})...")
    results = clean_episodes([e["text"] for e in episodes], nlp, boilerplate,
                              lemmatize=False,
                              min_token_len=nlp_config.DEFAULT_MIN_TOKEN_LEN,
                              n_process=args.n_process)

    show_stats: dict[str, CleanStats] = defaultdict(CleanStats)
    n_sentences = 0
    rows = []
    for e, (sentences, stats) in zip(episodes, results):
        show_stats[e["show_id"]] = show_stats[e["show_id"]] + stats
        n_sentences += len(sentences)
        show_name = show_names.get(e["show_id"], "")
        for ch in chunk_sentences(sentences, target_words=args.target_words,
                                   min_stub_words=args.min_stub_words):
            rows.append({
                "show": show_name,
                "collection_id": e["show_id"],
                "episode_id": e["episode_id"],
                "episode_title": e["episode_title"],
                "chunk_id": f"{e['episode_id']}_{ch.chunk_id}",
                "text": ch.text,
                "n_words": ch.n_words,
                "speakers": ch.speakers,
                "clean_text": ch.clean_text,
            })
    print(f"[clean] {n_sentences} sentences survived cleaning")

    out = pd.DataFrame(rows, columns=CHUNK_FIELDS)
    out.to_csv(OUT_PATH, index=False)
    w = out["n_words"]
    print(f"[chunk] {len(out)} passages | content words: min {w.min()} "
          f"p50 {w.median():.0f} mean {w.mean():.0f} max {w.max()} -> {OUT_PATH}")

    # Direct comparison against the contaminated segmentation this replaces.
    old_path = pipeline_config.OUTPUT_DIR / "chunks_500_nolemma.csv"
    n_old = None
    if old_path.exists():
        n_old = sum(1 for _ in open(old_path)) - 1
        print(f"[compare] contaminated corpus: {n_old} passages -> "
              f"cleaned: {len(out)} ({len(out) - n_old:+d}, "
              f"{100 * (len(out) - n_old) / n_old:+.1f}%)")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(CLEANED_PATH),
        "target_words": args.target_words,
        "min_stub_words": args.min_stub_words,
        "lemmatize": False,
        "min_token_len": nlp_config.DEFAULT_MIN_TOKEN_LEN,
        "boilerplate_ngrams_on_cleaned": len(boilerplate),
        "n_episodes": len(episodes),
        "n_shows": len({e["show_id"] for e in episodes}),
        "n_sentences_after_cleaning": n_sentences,
        "n_passages": len(out),
        "n_passages_contaminated_baseline": n_old,
        "mean_content_words": float(w.mean()),
    }
    (OUT_DIR / "rechunk_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] -> {OUT_DIR / 'rechunk_manifest.json'}")


if __name__ == "__main__":
    main()
