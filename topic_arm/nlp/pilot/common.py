"""Shared utilities for the LDA vs TLDA pilot bake-off.

Locked config for this pilot: data/output/chunks_500_nolemma.csv only.
Nothing here writes to the raw chunk files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

from tlda.tlda_wrapper import TLDA

import tensorly as tl


def _patch_tlda_eigval_bug():
    """Upstream bug in tlda/tlda_wrapper.py TLDA._unwhiten_factors: it computes
    eig_vals by iterating over rows of third_order.factors_ (shape
    n_eigenvec x n_topic), giving a length-n_eigenvec vector. That gets stored
    as self.weights_ and later broadcast against gammad (n_docs x n_topic) in
    third_order_cumulant.predict, which ValueErrors whenever n_eigenvec !=
    n_topic (the library's own example uses n_eigenvec = 5*n_topic). Fix:
    iterate over columns (one eigenvalue per topic) instead.
    """
    def _unwhiten_factors_fixed(self):
        factors_unwhitened = self.second_order.reverse_transform(self.third_order.factors_.T).T
        factors_unwhitened += tl.reshape(self.mean, (self.vocab, 1))
        factors_unwhitened[factors_unwhitened < 0.] = 0.
        self.unwhitened_factors_raw_ = tl.copy(factors_unwhitened)
        factors_unwhitened *= (1. - self.smoothing)
        factors_unwhitened += (self.smoothing / factors_unwhitened.shape[1])

        # --- fixed: one eigenvalue per topic (column), not per eigenvector (row) ---
        eig_vals = tl.tensor([
            tl.norm(self.third_order.factors_[:, t]) ** 3
            for t in range(self.third_order.factors_.shape[1])
        ])
        alpha = eig_vals ** (-2)
        alpha_norm = (alpha / alpha.sum()) * self.alpha_0
        self.weights_ = tl.tensor(alpha_norm)

        factors_unwhitened /= factors_unwhitened.sum(axis=0)
        return factors_unwhitened

    TLDA._unwhiten_factors = _unwhiten_factors_fixed


_patch_tlda_eigval_bug()

CHUNKS_PATH = "data/output/chunks_500_nolemma.csv"
NO_ABOVE = 0.99
NO_BELOW = 3


def load_tokenized_docs(chunks_path: str = CHUNKS_PATH) -> tuple[pd.DataFrame, list[list[str]]]:
    df = pd.read_csv(chunks_path)
    texts = [str(t).split() for t in df["clean_text"].fillna("")]
    return df, texts


def build_dictionary_and_dtm(texts: list[list[str]], no_above: float = NO_ABOVE, no_below: int = NO_BELOW):
    """Gensim Dictionary + BoW corpus + a matching scipy DTM (same vocab/order), so LDA and TLDA see identical input."""
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=None)
    dictionary.compactify()

    bow_corpus = [dictionary.doc2bow(t) for t in texts]

    n_docs = len(texts)
    n_vocab = len(dictionary)
    rows, cols, vals = [], [], []
    for d, bow in enumerate(bow_corpus):
        for term_id, count in bow:
            rows.append(d)
            cols.append(term_id)
            vals.append(count)
    dtm = csr_matrix((vals, (rows, cols)), shape=(n_docs, n_vocab))
    vocab = np.array([dictionary[i] for i in range(n_vocab)])
    return dictionary, bow_corpus, dtm, vocab


def fit_lda(dtm: csr_matrix, k: int, seed: int) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(n_components=k, random_state=seed, max_iter=20, learning_method="batch")
    lda.fit(dtm)
    return lda


def lda_topic_word(lda: LatentDirichletAllocation) -> np.ndarray:
    """topics x vocab, rows normalized to sum to 1."""
    comp = lda.components_
    return comp / comp.sum(axis=1, keepdims=True)


def lda_doc_topic(lda: LatentDirichletAllocation, dtm: csr_matrix) -> np.ndarray:
    theta = lda.transform(dtm)
    return theta / theta.sum(axis=1, keepdims=True)


def fit_tlda(dtm: csr_matrix, k: int, seed: int, n_eigenvec_mult: int = 3) -> TLDA:
    dtm_dense = np.asarray(dtm.todense(), dtype=np.float64)
    tlda = TLDA(
        n_topic=k,
        alpha_0=0.01,
        n_iter_train=2000,
        n_iter_test=10,
        learning_rate=1e-5,
        pca_batch_size=min(10000, dtm.shape[0]),
        third_order_cumulant_batch=min(10, dtm.shape[0]),
        theta=5.005,
        ortho_loss_criterion=1,
        random_seed=seed,
        n_eigenvec=k * n_eigenvec_mult,
    )
    tlda.fit(dtm_dense)
    return tlda


def tlda_topic_word(tlda: TLDA) -> np.ndarray:
    """topics x vocab, rows normalized to sum to 1. unwhitened_factors is vocab x topics, cols sum to 1."""
    comp = np.asarray(tlda.unwhitened_factors).T
    return comp / comp.sum(axis=1, keepdims=True)


def tlda_doc_topic(tlda: TLDA, dtm: csr_matrix) -> np.ndarray:
    dtm_dense = np.asarray(dtm.todense(), dtype=np.float64)
    gammad = tlda.transform(dtm_dense, predict=True)
    gammad = np.asarray(gammad)
    return gammad / gammad.sum(axis=1, keepdims=True)


def top_words(topic_word: np.ndarray, vocab: np.ndarray, n_words: int = 15) -> list[list[str]]:
    out = []
    for row in topic_word:
        idx = np.argsort(row)[::-1][:n_words]
        out.append([vocab[i] for i in idx])
    return out


def k_eff_and_entropy(doc_topic: np.ndarray) -> tuple[float, float]:
    """K_eff = exp(entropy(mean topic prevalence)); mean per-chunk entropy of theta_d."""
    q = doc_topic.mean(axis=0)
    q = q[q > 0]
    k_eff = float(np.exp(-np.sum(q * np.log(q))))

    theta = np.clip(doc_topic, 1e-12, 1.0)
    per_doc_entropy = -np.sum(theta * np.log(theta), axis=1)
    return k_eff, float(per_doc_entropy.mean())


def flag_degenerate_topics(doc_topic: np.ndarray, topic_word: np.ndarray, vocab: np.ndarray,
                            prevalence_floor: float = 0.005, dup_jaccard: float = 0.5, n_words: int = 10) -> dict:
    q = doc_topic.mean(axis=0)
    near_empty = [i for i, v in enumerate(q) if v < prevalence_floor]

    top_sets = [set(vocab[np.argsort(row)[::-1][:n_words]]) for row in topic_word]
    k = len(top_sets)
    dup_pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            overlap = len(top_sets[i] & top_sets[j]) / n_words
            if overlap >= dup_jaccard:
                dup_pairs.append((i, j, overlap))
    return {"near_empty_topics": near_empty, "near_duplicate_pairs": dup_pairs}


def best_match_alignment(vecs_a: np.ndarray, vecs_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine-similarity best matching between two topic sets (Hungarian, maximize).
    Returns (row_ind, col_ind) pairing indices of A to B."""
    a_norm = vecs_a / np.linalg.norm(vecs_a, axis=1, keepdims=True)
    b_norm = vecs_b / np.linalg.norm(vecs_b, axis=1, keepdims=True)
    sim = a_norm @ b_norm.T
    row_ind, col_ind = linear_sum_assignment(-sim)
    return row_ind, col_ind, sim


def jaccard_best_match(topics_a: list[list[str]], topics_b: list[list[str]]) -> float:
    k = len(topics_a)
    sets_a = [set(t) for t in topics_a]
    sets_b = [set(t) for t in topics_b]
    overlap = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            overlap[i, j] = len(sets_a[i] & sets_b[j]) / len(sets_a[i] | sets_b[j])
    row_ind, col_ind = linear_sum_assignment(-overlap)
    return float(overlap[row_ind, col_ind].mean())


def correspondence_analysis_d_eff(chunk_topic_matrix: np.ndarray) -> float:
    """CA on a (n_chunks x n_topics) non-negative matrix -> participation-ratio effective
    dimensionality of the principal inertia spectrum: (sum(lambda))^2 / sum(lambda^2)."""
    P = chunk_topic_matrix / chunk_topic_matrix.sum()
    r = P.sum(axis=1, keepdims=True)
    c = P.sum(axis=0, keepdims=True)
    E = r @ c
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(E > 0, (P - E) / np.sqrt(E), 0.0)
    _, sv, _ = np.linalg.svd(S, full_matrices=False)
    lam = sv ** 2
    lam = lam[lam > 1e-12]
    if lam.sum() <= 0:
        return float("nan")
    d_eff = (lam.sum() ** 2) / (lam ** 2).sum()
    return float(d_eff)
