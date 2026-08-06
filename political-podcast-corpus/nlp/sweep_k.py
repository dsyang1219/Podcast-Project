"""Sweep n_topics for standard LDA and score with UMass coherence + perplexity,
to pick a k that isn't just eyeballed.

UMass coherence (Mimno et al. 2011): for each topic's top-N words, average
log((co-doc-freq(wi,wj) + eps) / doc-freq(wj)) over ordered pairs i>j. Higher
(closer to 0 / less negative) is better. Computed directly from the binarized
DTM so no extra dependency (gensim) is needed.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def umass_coherence(dtm_binary: np.ndarray, top_word_idx: np.ndarray, eps: float = 1e-12) -> float:
    doc_freq = dtm_binary.sum(axis=0)
    scores = []
    for i in range(1, len(top_word_idx)):
        for j in range(i):
            wi, wj = top_word_idx[i], top_word_idx[j]
            co_df = np.minimum(dtm_binary[:, wi], dtm_binary[:, wj]).sum()
            scores.append(np.log((co_df + eps) / (doc_freq[wj] + eps)))
    return float(np.mean(scores))


def topic_redundancy(components: np.ndarray, vocab: np.ndarray, n_words: int = 10) -> float:
    """Fraction of topic pairs whose top-N words overlap >= 50% (near-duplicate topics)."""
    top_sets = [set(vocab[np.argsort(row)[::-1][:n_words]]) for row in components]
    k = len(top_sets)
    dup = 0
    total = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += 1
            overlap = len(top_sets[i] & top_sets[j]) / n_words
            if overlap >= 0.5:
                dup += 1
    return dup / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/output/chunks_300_nolemma.csv")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 8, 10, 12, 15, 20, 25])
    ap.add_argument("--max-features", type=int, default=3000)
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--max-df", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.chunks)
    docs = df["clean_text"].fillna("").tolist()
    train_docs, test_docs = train_test_split(docs, test_size=0.2, random_state=args.seed)

    vectorizer = CountVectorizer(max_features=args.max_features, min_df=args.min_df, max_df=args.max_df)
    train_dtm = vectorizer.fit_transform(train_docs)
    test_dtm = vectorizer.transform(test_docs)
    vocab = np.array(vectorizer.get_feature_names_out())
    train_binary = (train_dtm > 0).toarray().astype(np.float32)

    print(f"{len(train_docs)} train / {len(test_docs)} test docs, vocab={len(vocab)}")
    print(f"{'k':>4} {'perplexity':>12} {'coherence':>12} {'redundancy':>12}")
    for k in args.ks:
        lda = LatentDirichletAllocation(n_components=k, random_state=args.seed, max_iter=20, learning_method="batch")
        lda.fit(train_dtm)
        perplexity = lda.perplexity(test_dtm)

        coherences = []
        for row in lda.components_:
            top_idx = np.argsort(row)[::-1][:10]
            coherences.append(umass_coherence(train_binary, top_idx))
        mean_coherence = np.mean(coherences)
        redundancy = topic_redundancy(lda.components_, vocab)

        print(f"{k:>4} {perplexity:>12.1f} {mean_coherence:>12.2f} {redundancy:>12.2f}")


if __name__ == "__main__":
    main()
