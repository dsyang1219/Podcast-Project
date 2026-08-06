"""Minimal LDA arm: does TOPIC predict ideology better than STYLE?

    .venv/bin/python -m nlp.lda_ideology [--ks 20 30 50]

The embedding arm (nlp/embed_ideology.py, nlp/embed_ideology_nonlinear.py)
found DIME ideology is linearly recoverable from the FULL 768-dim show-level
embedding at LOSO-CV R^2~=0.21 (95% CI from that arm's bootstrap: see
embed_ideology_nonlinear.py's ridge_linear_baseline entry) on N=120 shows
with genuine (coverage in {full, partial}) DIME CFscore. That is a STYLE
representation -- how a show talks. This module asks whether WHAT a show
talks about (LDA topic proportions) predicts the same CFscore better, worse,
or about the same, using the IDENTICAL ideology-probe harness (Ridge,
leave-one-show-out CV, alpha selected inside each fold via GridSearchCV,
permutation negative control, bootstrap 95% CI) so the two R^2 numbers are
directly comparable.

Minimal version: LDA -> document-topic matrix -> the same probe. NOT
correspondence analysis, NOT a rank-1 null -- those are a separate, later
decision that depends on this result (see task brief).

Step 0 (DTM): tokens are already cleaned/filtered (nlp/clean.py -- boilerplate
stripped, filler words dropped, stopwords dropped, min_token_len=2). This is
Stage 3 (nlp/config.py: STAGE3_DF_NO_ABOVE=0.99, documented but not yet run
until now) -- prune near-universal terms (DF > 0.99) and add a floor (DF < 5
chunks) to remove ASR junk / one-off misspellings that a pure DF-ceiling
prune wouldn't touch.

Step 1 (LDA): tomotopy LDAModel (C++, fast) at each K in {20, 30, 50}, one
fixed pruned vocab shared across all K so the head-to-head isn't confounded
by different vocabularies at different K. c_v coherence reported per K as a
quality signal only -- K is NOT selected by coherence here, the ideology
probe is the selection criterion per the task brief.

Step 3 (probe): reuses nlp/embed_ideology_nonlinear.py's exact harness
(ridge_fit_predict / loso_cv / r2_from_predictions / bootstrap_r2_ci) and
nlp/embed_ideology.py's load_cfscore -- same N=120 shows, same alpha grid,
same inner-CV discipline, same permutation-null construction -- imported
directly rather than reimplemented, so there is no risk of the two harnesses
silently drifting apart.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tomotopy as tp
from tomotopy.coherence import Coherence

from pipeline import config as pipeline_config
from .embed_ideology import load_cfscore
from .embed_ideology_nonlinear import (
    bootstrap_r2_ci,
    loso_cv,
    r2_from_predictions,
    ridge_fit_predict,
)

CHUNKS_PATH = Path("data/output/chunks_500_nolemma.csv")
DF_NO_ABOVE = 0.99   # nlp/config.py STAGE3_DF_NO_ABOVE
DF_FLOOR_CHUNKS = 5  # ASR junk / one-off misspelling floor, new for Stage 3
KS = [20, 30, 50]
SEED = 0
PERM_SEED = 0


def load_chunks() -> pd.DataFrame:
    df = pd.read_csv(CHUNKS_PATH, usecols=["chunk_id", "collection_id", "clean_text"])
    df["collection_id"] = df["collection_id"].astype(str)
    df["clean_text"] = df["clean_text"].fillna("")
    return df


def build_pruned_vocab(token_lists: list[list[str]]) -> tuple[set[str], dict]:
    """DF pruning: drop terms with DF > DF_NO_ABOVE (near-universal) or
    DF < DF_FLOOR_CHUNKS chunks (ASR junk / one-off misspellings)."""
    n_docs = len(token_lists)
    df_counts: Counter = Counter()
    for toks in token_lists:
        df_counts.update(set(toks))

    vocab_before = set(df_counts)
    ceiling = DF_NO_ABOVE * n_docs
    kept = {w for w, c in df_counts.items() if DF_FLOOR_CHUNKS <= c <= ceiling}

    n_dropped_ceiling = sum(1 for w, c in df_counts.items() if c > ceiling)
    n_dropped_floor = sum(1 for w, c in df_counts.items() if c < DF_FLOOR_CHUNKS)
    stats = {
        "vocab_before_pruning": len(vocab_before),
        "vocab_after_pruning": len(kept),
        "n_dropped_df_ceiling": n_dropped_ceiling,
        "n_dropped_df_floor": n_dropped_floor,
        "n_docs": n_docs,
        "df_no_above_frac": DF_NO_ABOVE,
        "df_floor_chunks": DF_FLOOR_CHUNKS,
        "top_dropped_ceiling_terms": sorted(
            [w for w, c in df_counts.items() if c > ceiling], key=lambda w: -df_counts[w]
        )[:15],
    }
    return kept, stats


def fit_lda(token_lists: list[list[str]], chunk_ids: list[str], k: int, seed: int = SEED):
    # workers=0 (all cores) for speed; tomotopy's own warning notes exact Gibbs
    # sampling path isn't seed-reproducible across worker counts, only the
    # single-worker path is bit-identical. Doesn't affect the substantive
    # result here (one run per K, not a stability sweep).
    mdl = tp.LDAModel(k=k, seed=seed, tw=tp.TermWeight.ONE)
    for toks in token_lists:
        mdl.add_doc(toks)
    mdl.train(0)  # initialize
    n_iter = 500
    for i in range(0, n_iter, 20):
        mdl.train(20)
    coh = Coherence(mdl, coherence="c_v", top_n=10).get_score()
    # mdl.docs supports iteration but its __getitem__ is unreliable beyond
    # index 0 in this tomotopy build (0.14.0) -- iterate, don't index.
    doc_topic = np.array([d.get_topic_dist() for d in mdl.docs])
    return mdl, coh, doc_topic


def show_level_topics(doc_topic: np.ndarray, show_ids: np.ndarray, k: int) -> pd.DataFrame:
    cols = [f"T{i}" for i in range(k)]
    df = pd.DataFrame(doc_topic, columns=cols)
    df["show_id"] = show_ids
    agg = df.groupby("show_id")[cols].mean()
    agg["n_chunks"] = df.groupby("show_id").size()
    return agg


def permutation_null(X: np.ndarray, y: np.ndarray, seed: int = PERM_SEED) -> float:
    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y)
    preds = loso_cv(X, y_shuf, ridge_fit_predict)
    return r2_from_predictions(y_shuf, preds)


def top_terms(mdl, topic_idx: int, n: int = 12) -> list[str]:
    return [w for w, _ in mdl.get_topic_words(topic_idx, top_n=n)]


def run(ks: list[int]) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    cfscore_df = load_cfscore()
    print(f"[cfscore] N genuine (coverage full/partial) = {len(cfscore_df)}")

    chunks = load_chunks()
    print(f"[chunks] loaded {CHUNKS_PATH}: {len(chunks)} chunks, "
          f"{chunks['collection_id'].nunique()} shows")

    token_lists_full = [t.split() for t in chunks["clean_text"]]
    vocab, prune_stats = build_pruned_vocab(token_lists_full)
    print(f"[dtm] vocab before pruning={prune_stats['vocab_before_pruning']}, "
          f"after={prune_stats['vocab_after_pruning']} "
          f"(dropped {prune_stats['n_dropped_df_ceiling']} above DF={DF_NO_ABOVE}, "
          f"{prune_stats['n_dropped_df_floor']} below {DF_FLOOR_CHUNKS} chunks)")

    token_lists = [[w for w in toks if w in vocab] for toks in token_lists_full]
    n_empty = sum(1 for t in token_lists if len(t) == 0)
    keep_mask = np.array([len(t) > 0 for t in token_lists])
    print(f"[dtm] {n_empty} chunks empty after pruning, dropped")

    chunks_kept = chunks.loc[keep_mask].reset_index(drop=True)
    token_lists_kept = [t for t, m in zip(token_lists, keep_mask) if m]
    chunk_ids = chunks_kept["chunk_id"].tolist()
    show_ids_chunks = chunks_kept["collection_id"].to_numpy()

    n_chunks, vocab_size = len(chunk_ids), len(vocab)
    print(f"[dtm] final matrix shape = ({n_chunks}, {vocab_size})")

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(CHUNKS_PATH),
        "n_matched_shows": len(cfscore_df),
        "dtm_shape_final": [n_chunks, vocab_size],
        "n_chunks_dropped_empty_after_pruning": n_empty,
        "pruning": prune_stats,
        "ks": ks,
        "per_k": {},
        "embedding_baseline_reference": {
            "representation": "embedding (768d, full mean-pooled)",
            "unit": "show", "model": "Ridge",
            "loso_cv_r2": 0.21,
            "note": "cited from record (nlp/embed_ideology.py verdict_correction / "
                    "nlp/embed_ideology_nonlinear.py ridge_linear_baseline); "
                    "NOT rerun here per task scope.",
        },
    }

    head_to_head = [{
        "representation": "embedding (768d)", "unit": "show", "model": "Ridge",
        "loso_cv_r2": 0.21, "ci_95": None, "permutation_r2": None,
        "source": "record (embed_ideology_nonlinear.py)",
    }]

    for k in ks:
        print(f"\n=== K={k} ===")
        mdl, coh, doc_topic = fit_lda(token_lists_kept, chunk_ids, k)
        print(f"[lda k={k}] c_v coherence = {coh:.4f}")

        doc_topic_df = pd.DataFrame(doc_topic, columns=[f"T{i}" for i in range(k)])
        doc_topic_df.insert(0, "chunk_id", chunk_ids)
        doc_topic_df.to_csv(OUT / f"lda_doctopic_k{k}_500.csv", index=False)

        show_topics = show_level_topics(doc_topic, show_ids_chunks, k)
        joined = show_topics.join(cfscore_df, how="inner")
        n_matched = len(joined)
        print(f"[join k={k}] N matched shows = {n_matched}")

        cols = [f"T{i}" for i in range(k)]
        X = joined[cols].to_numpy()
        y = joined["avg_host_cfscore"].to_numpy()

        preds = loso_cv(X, y, ridge_fit_predict)
        cv_r2 = r2_from_predictions(y, preds)
        lo, hi = bootstrap_r2_ci(y, preds)
        perm_r2 = permutation_null(X, y)
        print(f"[probe k={k}] LOSO-CV R2={cv_r2:.4f} (95% CI [{lo:.4f}, {hi:.4f}])  "
              f"permutation R2={perm_r2:.4f}")

        ridge_fit = ridge_fit_predict(X, y, X)
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV
        pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
        grid = GridSearchCV(pipe, {"ridge__alpha": [1.0, 10.0, 100.0, 1000.0, 10000.0]}, cv=5, scoring="r2")
        grid.fit(X, y)
        coefs = grid.best_estimator_.named_steps["ridge"].coef_
        top_pos = np.argsort(coefs)[::-1][:5]
        top_neg = np.argsort(coefs)[:5]
        top_topics = {
            "most_positive_(conservative-leaning_cfscore)": [
                {"topic": int(i), "coef": float(coefs[i]), "terms": top_terms(mdl, int(i))} for i in top_pos
            ],
            "most_negative_(liberal-leaning_cfscore)": [
                {"topic": int(i), "coef": float(coefs[i]), "terms": top_terms(mdl, int(i))} for i in top_neg
            ],
        }

        report["per_k"][str(k)] = {
            "k": k,
            "coherence_cv": float(coh),
            "n_matched_shows": n_matched,
            "loso_cv_r2": float(cv_r2),
            "bootstrap_95ci": [lo, hi],
            "permutation_r2": float(perm_r2),
            "top_ideology_topics": top_topics,
            "doc_topic_csv": f"lda_doctopic_k{k}_500.csv",
        }
        head_to_head.append({
            "representation": f"LDA topics (K={k})", "unit": "show", "model": "Ridge",
            "loso_cv_r2": float(cv_r2), "ci_95": [lo, hi], "permutation_r2": float(perm_r2),
            "source": "computed",
        })

    report["head_to_head"] = head_to_head

    best = max(report["per_k"].values(), key=lambda r: r["loso_cv_r2"])
    embed_r2 = 0.21
    best_r2 = best["loso_cv_r2"]
    lo_b, hi_b = best["bootstrap_95ci"]
    # Judged on the POINT ESTIMATE gap to the embedding baseline, not raw CI
    # overlap: at N=120 the LDA-probe bootstrap CIs are wide enough (e.g.
    # [-0.45, 0.27]) to overlap almost any embedding CI by construction, which
    # would make "CI overlaps 0.21" a near-meaningless equivalence test here.
    if best_r2 > embed_r2 + 0.05 and lo_b > 0.05:
        verdict = ("topic >> style: LDA topic proportions predict CFscore meaningfully better than "
                   f"the full embedding (best K={best['k']}, R2={best_r2:.3f} vs embedding R2={embed_r2:.2f}). "
                   "Ideology in this corpus is more thematic than stylistic -- worth building the full "
                   "CA / rank-1-null arm.")
    elif abs(best_r2 - embed_r2) <= 0.08:
        verdict = ("topic ~= style: LDA topic proportions (best K="
                   f"{best['k']}, R2={best_r2:.3f}, CI [{lo_b:.3f},{hi_b:.3f}]) land in the same "
                   f"range as the embedding baseline (R2={embed_r2:.2f}). Ideology is weakly encoded in "
                   "both representations -- a robust 'it's just weak' finding across two very different "
                   "methods. CA is optional; the topic axis is still worth reporting for interpretability.")
    else:
        verdict = ("topic << style: LDA topic proportions (best K="
                   f"{best['k']}, R2={best_r2:.3f}, CI [{lo_b:.3f},{hi_b:.3f}]) fall meaningfully short of "
                   f"the embedding baseline (R2={embed_r2:.2f}). Ideology looks more encoded in HOW shows "
                   "talk than WHAT they talk about -- worth checking whether topic granularity at these K "
                   "is too compressed before concluding CA isn't worth it.")
    report["verdict"] = verdict
    print(f"\n=== VERDICT ===\n{verdict}")

    report_path = OUT / "lda_ideology_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path.name}")

    print("\n=== HEAD-TO-HEAD ===")
    print(f"{'representation':<22}{'unit':<7}{'model':<8}{'LOSO-CV R2':<13}{'95% CI':<20}{'perm R2':<10}")
    for row in head_to_head:
        ci = f"[{row['ci_95'][0]:.3f},{row['ci_95'][1]:.3f}]" if row["ci_95"] else "record"
        perm = f"{row['permutation_r2']:.3f}" if row["permutation_r2"] is not None else "n/a"
        print(f"{row['representation']:<22}{row['unit']:<7}{row['model']:<8}{row['loso_cv_r2']:<13.3f}{ci:<20}{perm:<10}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    args = ap.parse_args()
    run(args.ks)


if __name__ == "__main__":
    main()
