"""Build one curated regime's full-corpus token export for the LDA refit
(nlp/lda_regime_refit.py). Reads raw passage text (chunks_500_nolemma.csv's
`text` column -- boilerplate-sentence-stripped but otherwise unprocessed,
SAME passage boundaries as the existing pipeline, no re-segmentation),
applies nlp/preprocess_regimes.py's toggles, and writes:

    data/output/regimes/tokens_<regime>.csv         (chunk_id, collection_id, clean_text)
    data/output/regimes/vocab_accounting_<regime>.json

`aggressive` and `aggressive_pos` share ONE spaCy annotation pass (cached
to data/output/regimes/spacy_annotation_cache.pkl) since they differ only
in downstream filtering of the same tagged tokens, not in what gets
tagged -- this halves the (expensive) spaCy cost for the two POS-dependent
regimes.

Usage:
    .venv/bin/python -m nlp.export_regime_tokens --regime moderate
    .venv/bin/python -m nlp.export_regime_tokens --all
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from . import preprocess_regimes as pp

CHUNKS_PATH = Path("data/output/chunks_500_nolemma.csv")
OUT_DIR = Path("data/output/regimes")
SPACY_CACHE = OUT_DIR / "spacy_annotation_cache.pkl"
N_PROCESS = 5  # middle ground: fewer workers than the original 10 (~30GB peak) but faster than 3 (~10-15GB peak)


def load_raw(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(CHUNKS_PATH, usecols=["chunk_id", "collection_id", "text"],
                      dtype={"chunk_id": str, "collection_id": str}, nrows=limit)
    df["text"] = df["text"].fillna("")
    return df


def get_or_build_spacy_annotation(texts: list[str]) -> list:
    if SPACY_CACHE.exists():
        print(f"[cache] loading spaCy annotation from {SPACY_CACHE}")
        with open(SPACY_CACHE, "rb") as f:
            return pickle.load(f)
    print(f"[cache] no cache found, annotating {len(texts)} passages with spaCy "
          f"(n_process={N_PROCESS}) -- shared by aggressive + aggressive_pos...")
    ann = pp.annotate_spacy(texts, disfluency=True, n_process=N_PROCESS, batch_size=50, verbose=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPACY_CACHE, "wb") as f:
        pickle.dump(ann, f)
    print(f"[cache] wrote {SPACY_CACHE}")
    return ann


def export_regime(regime: str, raw: pd.DataFrame) -> None:
    toggles = pp.regime_toggles(regime)
    texts = raw["text"].tolist()

    needs_spacy = toggles.lemmatize or toggles.pos_filter
    if needs_spacy:
        ann = get_or_build_spacy_annotation(texts)
        token_lists = pp.tokens_from_annotation(ann, toggles)
        if toggles.disfluency:
            token_lists = [[t for t in toks if t not in pp.BACKCHANNEL] for toks in token_lists]
        if toggles.stopwords:
            sw = pp.extended_stopwords()
            token_lists = [[t for t in toks if t not in sw] for toks in token_lists]
        if toggles.ngrams:
            token_lists = pp.compound_collocations(token_lists, verbose=True)
    else:
        token_lists = pp.build_token_lists(texts, toggles, n_process=1, verbose=True)

    pruned, acc = pp.apply_docfreq_prune(token_lists, toggles, verbose=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "chunk_id": raw["chunk_id"],
        "collection_id": raw["collection_id"],
        "clean_text": [" ".join(t) for t in pruned],
    })
    out_path = OUT_DIR / f"tokens_{regime}.csv"
    out_df.to_csv(out_path, index=False)
    n_empty = (out_df["clean_text"] == "").sum()
    print(f"[{regime}] wrote {out_path} ({len(out_df)} rows, {n_empty} empty)")

    acc["regime"] = regime
    acc["n_empty_docs"] = int(n_empty)
    acc["toggles"] = vars(toggles)
    acc_path = OUT_DIR / f"vocab_accounting_{regime}.json"
    acc_path.write_text(json.dumps(acc, indent=2, default=str))
    print(f"[{regime}] vocab accounting -> {acc_path}")
    print(f"[{regime}] DONE: vocab {acc['vocab_before']} -> {acc['vocab_after']}, "
          f"tokens {acc['tokens_before']} -> {acc['tokens_after']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on first N rows")
    args = ap.parse_args()

    raw = load_raw(limit=args.limit)
    print(f"[data] {len(raw)} passages loaded from {CHUNKS_PATH}")

    if args.all:
        # spaCy-needing regimes first so the shared cache is built once
        regimes = ["aggressive", "aggressive_pos", "minimal", "moderate"]
    elif args.regime:
        regimes = [args.regime]
    else:
        ap.error("pass --regime NAME or --all")
        return

    for r in regimes:
        print(f"\n{'=' * 70}\n[export] regime = {r}\n{'=' * 70}")
        export_regime(r, raw)


if __name__ == "__main__":
    main()
