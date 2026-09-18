"""Re-embed the ad-span-CLEANED corpus and pool to show level.

Why this exists: every chunk_embeddings_*.npy in data/output/ was built on the
CONTAMINATED corpus (Jul 27, before Phase B excision). Using them as the
"embeddings" representation in the nested-CV ceiling would mean the embedding
arm sees ad copy that the topic arm does not, so any "embeddings add signal
beyond topics" result could be an artifact of the ads alone.

FIDELITY TO THE ORIGINAL PIPELINE
---------------------------------
The original path is: sentences -> embed each sentence -> length-weighted mean
to chunk (nlp/pool.py, weight = n_raw_words) -> plain mean to show
(embed_ideology_nonlinear.build_show_level_embeddings). This reproduces that
exactly, with one shortcut that is safe by construction: rather than re-running
clean_episodes() to recover the Sentence objects, it re-splits each cleaned
chunk's `text` with the SAME splitter the chunker used -- nlp/clean.load_nlp
disables the parser and adds a rule-based "sentencizer", so sentence boundaries
are pure punctuation rules and a blank English pipeline + sentencizer reproduces
them. `text` is the verbatim join of the surviving sentences, so splitting it
recovers the same sentences the chunker grouped, without a second spaCy parse.

Uses nlp/config.py EMBED_MODEL/EMBED_PREFIX so the model matches the original
run (all-mpnet-base-v2, prefix-free) rather than silently switching encoders.

    .venv/bin/python -m nlp.adspan_embed_clean [--batch-size 256]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import spacy

from . import config as nlp_config
from .adspan_phase_b import OUT_DIR

CHUNKS = OUT_DIR / "chunks_500_nolemma_adclean.csv"
OUT_NPY = OUT_DIR / "show_embeddings_adclean.npy"
OUT_IDS = OUT_DIR / "show_embeddings_adclean_ids.json"
OUT_META = OUT_DIR / "show_embeddings_adclean.meta.json"


def sentencize(texts: list[str], n_process: int) -> list[list[str]]:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    out = []
    for doc in nlp.pipe(texts, n_process=n_process, batch_size=200):
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        out.append(sents or [""])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-process", type=int, default=4)
    ap.add_argument("--model", default=nlp_config.EMBED_MODEL)
    ap.add_argument("--prefix", default=nlp_config.EMBED_PREFIX)
    args = ap.parse_args()

    ch = pd.read_csv(CHUNKS, usecols=["chunk_id", "collection_id", "text"],
                     dtype={"collection_id": str})
    ch["text"] = ch["text"].fillna("")
    print(f"[chunks] {len(ch)} cleaned chunks across {ch.collection_id.nunique()} shows")

    print(f"[sentencize] splitting with the chunker's own sentencizer "
          f"(n_process={args.n_process}) ...")
    sents_per_chunk = sentencize(ch["text"].tolist(), args.n_process)
    flat, owner = [], []
    for i, sents in enumerate(sents_per_chunk):
        for s in sents:
            flat.append(s)
            owner.append(i)
    owner = np.asarray(owner)
    weights = np.asarray([max(len(s.split()), 1) for s in flat], dtype=np.float64)
    print(f"[sentencize] {len(flat)} sentences "
          f"({len(flat) / max(len(ch), 1):.1f} per chunk)")

    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    print(f"[embed] model={args.model} prefix={args.prefix!r} device={device}")
    prefixed = [args.prefix + s for s in flat] if args.prefix else flat

    dim = model.get_sentence_embedding_dimension()
    n_chunks = len(ch)
    # Accumulate weighted sums per chunk as we stream, so the full
    # (n_sentences x 768) matrix never has to be held in memory at once.
    chunk_sum = np.zeros((n_chunks, dim), dtype=np.float64)
    chunk_w = np.zeros(n_chunks, dtype=np.float64)
    BLOCK = 100_000
    for start in range(0, len(prefixed), BLOCK):
        stop = min(start + BLOCK, len(prefixed))
        V = model.encode(prefixed[start:stop], batch_size=args.batch_size,
                         convert_to_numpy=True, normalize_embeddings=False,
                         show_progress_bar=False)
        o = owner[start:stop]
        w = weights[start:stop]
        np.add.at(chunk_sum, o, V * w[:, None])
        np.add.at(chunk_w, o, w)
        print(f"  ...{stop}/{len(prefixed)} sentences", flush=True)

    chunk_vecs = chunk_sum / np.maximum(chunk_w, 1e-12)[:, None]

    df = pd.DataFrame(chunk_vecs)
    df["show_id"] = ch["collection_id"].to_numpy()
    show_means = df.groupby("show_id").mean().sort_index()
    X = show_means.to_numpy(dtype=np.float32)
    np.save(OUT_NPY, X)
    OUT_IDS.write_text(json.dumps(list(show_means.index)))
    OUT_META.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "prefix": args.prefix, "device": device,
        "corpus": str(CHUNKS), "n_chunks": int(n_chunks),
        "n_sentences": int(len(flat)), "n_shows": int(X.shape[0]), "dim": int(X.shape[1]),
        "pooling": "sentence -> length-weighted mean to chunk -> plain mean to show",
    }, indent=2))
    print(f"[saved] {X.shape} -> {OUT_NPY}")


if __name__ == "__main__":
    main()
