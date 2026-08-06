"""Compare standard (variational) LDA against TensorLy TLDA on the same
document-term matrix built from the chunked political-podcast corpus.

Usage:
    .venv-tlda/bin/python nlp/compare_lda_tlda.py \
        --chunks data/output/chunks_500_nolemma.csv --n-topics 10

TLDA is the online, GPU-capable spectral/moment-based estimator from
Kangaslahti et al. 2026 ("Analyzing Political Text at Scale with Online
Tensor LDA", Political Analysis) -- github.com/TensorLy/tlda. Standard LDA
here is sklearn's variational-inference implementation, the common baseline
it's compared against in that paper.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from tlda.tlda_wrapper import TLDA


def top_words_per_topic(components: np.ndarray, vocab: np.ndarray, n_words: int = 12) -> list[list[str]]:
    topics = []
    for row in components:
        top_idx = np.argsort(row)[::-1][:n_words]
        topics.append([vocab[i] for i in top_idx])
    return topics


def topic_overlap(topics_a: list[list[str]], topics_b: list[list[str]]) -> np.ndarray:
    """Best-match Jaccard overlap: for each topic in A, its max overlap with any topic in B."""
    scores = np.zeros((len(topics_a), len(topics_b)))
    for i, a in enumerate(topics_a):
        set_a = set(a)
        for j, b in enumerate(topics_b):
            set_b = set(b)
            scores[i, j] = len(set_a & set_b) / len(set_a | set_b)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/output/chunks_500_nolemma.csv")
    ap.add_argument("--n-topics", type=int, default=10)
    ap.add_argument("--max-features", type=int, default=3000)
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--max-df", type=float, default=0.4)  # prune cross-topic discourse filler
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.chunks)
    docs = df["clean_text"].fillna("").tolist()
    print(f"Loaded {len(docs)} chunks from {args.chunks}")

    vectorizer = CountVectorizer(
        max_features=args.max_features, min_df=args.min_df, max_df=args.max_df
    )
    dtm = vectorizer.fit_transform(docs)
    vocab = np.array(vectorizer.get_feature_names_out())
    print(f"Document-term matrix: {dtm.shape[0]} docs x {dtm.shape[1]} vocab")

    # ---- standard LDA (variational inference) ----
    t0 = time.perf_counter()
    lda = LatentDirichletAllocation(
        n_components=args.n_topics, random_state=args.seed, max_iter=20, learning_method="batch"
    )
    lda.fit(dtm)
    lda_time = time.perf_counter() - t0
    lda_topics = top_words_per_topic(lda.components_, vocab)

    # ---- TLDA (online tensor / spectral) ----
    t0 = time.perf_counter()
    tlda = TLDA(
        n_topic=args.n_topics,
        alpha_0=0.01,
        n_iter_train=2000,
        n_iter_test=10,
        learning_rate=1e-5,
        pca_batch_size=min(10000, dtm.shape[0]),
        third_order_cumulant_batch=min(10, dtm.shape[0]),
        theta=5.005,
        ortho_loss_criterion=1,
        random_seed=args.seed,
        n_eigenvec=args.n_topics * 3,
    )
    dtm_dense = np.asarray(dtm.todense(), dtype=np.float32)
    tlda.fit(dtm_dense)
    tlda_time = time.perf_counter() - t0
    tlda_components = np.asarray(tlda.unwhitened_factors).T  # topics x vocab
    tlda_topics = top_words_per_topic(tlda_components, vocab)

    # ---- report ----
    print("\n=== Standard LDA (sklearn, variational) ===")
    print(f"fit time: {lda_time:.2f}s")
    for i, words in enumerate(lda_topics):
        print(f"  topic {i}: {', '.join(words)}")

    print("\n=== TLDA (tensor / spectral, online) ===")
    print(f"fit time: {tlda_time:.2f}s")
    for i, words in enumerate(tlda_topics):
        print(f"  topic {i}: {', '.join(words)}")

    overlap = topic_overlap(lda_topics, tlda_topics)
    best_match = overlap.max(axis=1)
    print("\n=== Topic overlap (best-match Jaccard, LDA topic -> closest TLDA topic) ===")
    for i, score in enumerate(best_match):
        j = int(np.argmax(overlap[i]))
        print(f"  LDA topic {i} <-> TLDA topic {j}: Jaccard={score:.2f}")
    print(f"\nmean best-match Jaccard: {best_match.mean():.3f}")
    print(f"speed ratio (LDA time / TLDA time): {lda_time / tlda_time:.2f}x")


if __name__ == "__main__":
    main()
