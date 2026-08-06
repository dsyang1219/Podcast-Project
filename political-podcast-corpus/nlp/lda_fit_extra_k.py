"""Fit LDA at K=75/100/150 only (K=20/30/50 already fit and saved). No
coherence computation here -- Check 1 (K-curve) doesn't need it and it was
the dominant single-threaded cost in the original K=20/30/50 run; dropping
it is the main speed lever for this follow-up. Same vocab-pruning pipeline,
same 500 Gibbs iterations, same seed as the original run for consistency.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import tomotopy as tp

from pipeline import config as pipeline_config
from .lda_ideology import CHUNKS_PATH, SEED, build_pruned_vocab, load_chunks

KS = [75, 100, 150]


def main() -> None:
    OUT = pipeline_config.OUTPUT_DIR
    chunks = load_chunks()
    print(f"[chunks] {len(chunks)} chunks")

    token_lists_full = [t.split() for t in chunks["clean_text"]]
    vocab, prune_stats = build_pruned_vocab(token_lists_full)
    print(f"[vocab] {prune_stats['vocab_before_pruning']} -> {prune_stats['vocab_after_pruning']}")

    token_lists = [[w for w in toks if w in vocab] for toks in token_lists_full]
    keep_mask = np.array([len(t) > 0 for t in token_lists])
    chunks_kept = chunks.loc[keep_mask].reset_index(drop=True)
    token_lists_kept = [t for t, m in zip(token_lists, keep_mask) if m]
    chunk_ids = chunks_kept["chunk_id"].tolist()
    print(f"[dtm] {len(chunk_ids)} chunks x {len(vocab)} vocab")

    for k in KS:
        t0 = time.perf_counter()
        mdl = tp.LDAModel(k=k, seed=SEED, tw=tp.TermWeight.ONE)
        for toks in token_lists_kept:
            mdl.add_doc(toks)
        mdl.train(0)
        for i in range(0, 500, 50):
            mdl.train(50)
        doc_topic = np.array([d.get_topic_dist() for d in mdl.docs])
        elapsed = time.perf_counter() - t0
        print(f"[lda k={k}] fit in {elapsed:.1f}s")

        doc_topic_df = pd.DataFrame(doc_topic, columns=[f"T{i}" for i in range(k)])
        doc_topic_df.insert(0, "chunk_id", chunk_ids)
        out_path = OUT / f"lda_doctopic_k{k}_500.csv"
        doc_topic_df.to_csv(out_path, index=False)
        print(f"[save] -> {out_path.name}")


if __name__ == "__main__":
    main()
