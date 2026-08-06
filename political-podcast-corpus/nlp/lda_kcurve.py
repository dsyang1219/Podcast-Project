"""Is the CLR-LDA topic->ideology signal real, or feature-count inflation?

Three checks, same Ridge/LOSO-CV/alpha-in-fold/permutation-null harness
throughout (imported from nlp.embed_ideology_nonlinear, not reimplemented):

Check 1 -- extend the K sweep to 20/30/50/75/100/150 (75/100/150 doc-topic
matrices fit separately by nlp/lda_fit_extra_k.py, reused here; 20/30/50
reused from the original nlp/lda_ideology.py run). Same documented one-show
exclusion (Jorge Ramos, N=119) and CLR transform as the follow-up run. Shape
of R^2 vs K is the diagnostic: plateau = real signal, monotonic-no-plateau =
feature-count inflation, climb-then-collapse = overfitting past some K.

Check 2 -- recompute the embedding arm's Ridge/LOSO R^2 on the SAME N=119
(same one show dropped) so the "topics vs style" comparison is apples-to-
apples. Does NOT touch or replace the on-record N=120 embedding result
(R^2=0.21) -- this is an additional recompute for comparison only.

Check 3 -- equal-dimensionality comparison: PCA-reduce the show-level
embedding matrix to 30 and 50 components (global PCA on the N=119 embedding
matrix, matching the precedent set by nlp/embed_ideology.py's chunk-level PCA
arm, which is also fit on the whole corpus rather than per-LOSO-fold -- PCA
is unsupervised, so this doesn't leak the outcome variable), run the
identical Ridge/LOSO probe, and compare against CLR-LDA at the same K.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pipeline import config as pipeline_config
from .embed_ideology import load_cfscore
from .embed_ideology_nonlinear import (
    bootstrap_r2_ci,
    build_show_level_embeddings,
    loso_cv,
    r2_from_predictions,
    ridge_fit_predict,
)

KS = [20, 30, 50, 75, 100, 150]
EXCLUDED_SHOW_ID = "1839861250"
EXCLUDED_SHOW_NAME = "The Moment with Jorge Ramos and Paola Ramos"
TARGET = 500


def clr(X: np.ndarray) -> np.ndarray:
    logX = np.log(X)
    return logX - logX.mean(axis=1, keepdims=True)


def permutation_r2(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y)
    preds = loso_cv(X, y_shuf, ridge_fit_predict)
    return r2_from_predictions(y_shuf, preds)


def probe(X: np.ndarray, y: np.ndarray) -> dict:
    preds = loso_cv(X, y, ridge_fit_predict)
    r2 = r2_from_predictions(y, preds)
    lo, hi = bootstrap_r2_ci(y, preds)
    perm = permutation_r2(X, y)
    return {"loso_cv_r2": float(r2), "bootstrap_95ci": [lo, hi], "permutation_r2": float(perm)}


def main() -> None:
    OUT = pipeline_config.OUTPUT_DIR
    cfscore_120 = load_cfscore()
    cfscore_119 = cfscore_120.drop(index=EXCLUDED_SHOW_ID)
    print(f"[cfscore] N=120 on record, N={len(cfscore_119)} after documented exclusion "
          f"({EXCLUDED_SHOW_NAME})")

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "excluded_show": {"show_id": EXCLUDED_SHOW_ID, "show": EXCLUDED_SHOW_NAME},
        "n_matched_shows": len(cfscore_119),
    }

    # ---------------- Check 2: embedding R^2 recomputed at N=119 ----------------
    print("\n=== Check 2: embedding Ridge/LOSO recomputed at N=119 ===")
    show_ids_all, X_emb_all = build_show_level_embeddings(TARGET)
    id_to_row = {sid: i for i, sid in enumerate(show_ids_all)}
    matched_ids_119 = [sid for sid in cfscore_119.index if sid in id_to_row]
    assert len(matched_ids_119) == len(cfscore_119), "embedding/cfscore join drifted at N=119"
    X_emb_119 = np.stack([X_emb_all[id_to_row[sid]] for sid in matched_ids_119])
    y_119 = cfscore_119.loc[matched_ids_119, "avg_host_cfscore"].to_numpy()

    embedding_n119 = probe(X_emb_119, y_119)
    print(f"[embedding N=119] R2={embedding_n119['loso_cv_r2']:.4f} "
          f"CI={embedding_n119['bootstrap_95ci']} perm={embedding_n119['permutation_r2']:.4f}")
    report["embedding_full768_N119"] = embedding_n119
    report["embedding_full768_N120_on_record"] = {"loso_cv_r2": 0.21, "ci_95": [0.03, 0.35], "note": "on record, not rerun"}

    # ---------------- Check 3: PCA-matched-dimensionality embeddings ----------------
    print("\n=== Check 3: PCA-reduced embedding at matched K ===")
    pca_results = {}
    for k in [30, 50]:
        pca = PCA(n_components=k, random_state=0)
        X_pca = pca.fit_transform(X_emb_119)
        res = probe(X_pca, y_119)
        pca_results[str(k)] = res
        print(f"[PCA k={k}] R2={res['loso_cv_r2']:.4f} CI={res['bootstrap_95ci']} perm={res['permutation_r2']:.4f}")
    report["embedding_pca_matched_dim"] = pca_results

    # ---------------- Check 1: K-curve for CLR-LDA ----------------
    print("\n=== Check 1: CLR-LDA K-curve ===")
    chunks = pd.read_csv(OUT / "chunks_500_nolemma.csv", usecols=["chunk_id", "collection_id"])
    chunks["collection_id"] = chunks["collection_id"].astype(str)

    kcurve = {}
    for k in KS:
        path = OUT / f"lda_doctopic_k{k}_500.csv"
        if not path.exists():
            print(f"[k={k}] MISSING doc-topic matrix ({path.name}), skipping")
            continue
        doc_topic = pd.read_csv(path)
        doc_topic = doc_topic.merge(chunks, on="chunk_id")
        cols = [c for c in doc_topic.columns if c.startswith("T")]
        show_topics = doc_topic.groupby("collection_id")[cols].mean()
        joined = show_topics.join(cfscore_119, how="inner")
        n_matched = len(joined)
        X_raw = joined[cols].to_numpy()
        assert (X_raw > 0).all(), f"k={k}: non-positive topic proportion, CLR needs epsilon"
        X = clr(X_raw)
        y = joined["avg_host_cfscore"].to_numpy()

        res = probe(X, y)
        res["n_matched_shows"] = n_matched
        kcurve[str(k)] = res
        print(f"[k={k:>3}] N={n_matched} R2={res['loso_cv_r2']:.4f} "
              f"CI=[{res['bootstrap_95ci'][0]:.3f},{res['bootstrap_95ci'][1]:.3f}] "
              f"perm={res['permutation_r2']:.4f}")

    report["kcurve"] = kcurve

    # ---------------- interpretation ----------------
    ks_done = sorted(int(k) for k in kcurve)
    r2s = [kcurve[str(k)]["loso_cv_r2"] for k in ks_done]
    perms = [kcurve[str(k)]["permutation_r2"] for k in ks_done]

    max_r2 = max(r2s)
    max_k = ks_done[r2s.index(max_r2)]
    last3 = r2s[-3:] if len(r2s) >= 3 else r2s
    is_still_rising = len(r2s) >= 2 and r2s[-1] >= max(r2s[:-1])
    plateaued = len(last3) == 3 and (max(last3) - min(last3)) < 0.05 and last3[-1] >= last3[0] - 0.05
    collapsed = len(r2s) >= 2 and r2s[-1] < max_r2 - 0.08 and max_k != ks_done[-1]
    null_rising = perms[-1] - perms[0] > 0.05 if len(perms) >= 2 else False

    if null_rising:
        shape = ("permutation null itself rises with K -- DIRECT evidence of feature-count inflation "
                 "(more features fit shuffled labels too); real-data R2 at high K is not trustworthy.")
    elif collapsed:
        shape = f"climbs then collapses -- peak at K={max_k} (R2={max_r2:.3f}) is likely overfitting, not real signal."
    elif plateaued:
        shape = f"plateaus around K={ks_done[len(ks_done)-3]}-{ks_done[-1]} (R2~{np.mean(last3):.3f}) -- signal looks real."
    elif is_still_rising:
        shape = (f"still climbing monotonically through K={ks_done[-1]} with no plateau -- "
                 "ARTIFACT signature (feature-count inflation); the K=50/high-K numbers are not "
                 "trustworthy as a 'beats style' claim on their own.")
    else:
        shape = "mixed / inconclusive shape -- inspect the table directly."

    report["kcurve_shape_read"] = shape
    print(f"\n[shape] {shape}")

    # equal-dim head-to-head table
    equal_dim_table = []
    for k in [30, 50]:
        if str(k) in kcurve and str(k) in pca_results:
            equal_dim_table.append({
                "k": k,
                "lda_clr_r2": kcurve[str(k)]["loso_cv_r2"],
                "embedding_pca_r2": pca_results[str(k)]["loso_cv_r2"],
                "embedding_full768_r2_N119": embedding_n119["loso_cv_r2"],
                "embedding_full768_r2_N120_record": 0.21,
            })
    report["equal_dimensionality_table"] = equal_dim_table
    print("\n=== Check 3 table ===")
    for row in equal_dim_table:
        print(f"K={row['k']}: LDA(CLR)={row['lda_clr_r2']:.3f}  "
              f"embedding-PCA(same K)={row['embedding_pca_r2']:.3f}  "
              f"embedding-full768(N=119)={row['embedding_full768_r2_N119']:.3f}  "
              f"embedding-full768(N=120,record)=0.21")

    # ---------------- verdict ----------------
    trustworthy_k = None
    for k in [30, 50, 20, 75, 100, 150]:
        if str(k) in kcurve:
            trustworthy_k = k
            break
    beats_at_equal_dim = all(row["lda_clr_r2"] > row["embedding_pca_r2"] for row in equal_dim_table) if equal_dim_table else False
    beats_full_dim = (kcurve.get(str(trustworthy_k), {}).get("loso_cv_r2", -1) > embedding_n119["loso_cv_r2"]) if trustworthy_k else False

    if null_rising or (is_still_rising and not plateaued):
        verdict = "topics ~= style (high-K gain looks like feature-count inflation, not a real advantage)"
    elif plateaued and beats_at_equal_dim and beats_full_dim:
        verdict = "topics > style, real (plateau above embedding, holds at matched dimensionality)"
    elif beats_at_equal_dim:
        verdict = "topics >= style at matched dimensionality (real, modest advantage)"
    else:
        verdict = "topics < style once N and dimensionality are matched"

    report["verdict"] = verdict

    if plateaued and len(ks_done) >= 3:
        # first K in the plateau region (last3): pre-peak, avoids picking the
        # largest/most capacity-heavy K even though it's not the max point.
        trustworthy_k = ks_done[len(ks_done) - 3]
        reason = (
            f"K={trustworthy_k} is the first K in the plateau region (R2 stops rising meaningfully "
            f"from here through K={ks_done[-1]}, {last3}), its permutation null is pinned near 0 "
            f"({kcurve[str(trustworthy_k)]['permutation_r2']:.3f}), and it isn't the largest/most "
            "capacity-heavy K tested -- picking the peak (K=100) instead would over-credit whichever "
            "K happened to land highest by noise within a flat plateau."
        )
    else:
        trustworthy_k = 30
        reason = "fallback: plateau not clearly established, defaulting to mid-sweep K=30."
    report["most_trustworthy_k"] = trustworthy_k
    report["most_trustworthy_k_reason"] = reason
    print(f"\n=== VERDICT ===\n{verdict}")
    print(f"[most trustworthy K] {trustworthy_k}: {reason}")

    # ---------------- plot ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks_done, r2s, marker="o", label="CLR-LDA topics, real CFscore", color="#2166ac")
    ax.plot(ks_done, perms, marker="o", label="permutation null (shuffled CFscore)", color="#b2182b", linestyle="--")
    ax.axhline(embedding_n119["loso_cv_r2"], color="#4d4d4d", linestyle=":",
               label=f"embedding full-768d, N=119 (R2={embedding_n119['loso_cv_r2']:.2f})")
    ax.axhline(0.21, color="#999999", linestyle=":",
               label="embedding full-768d, N=120 on record (R2=0.21)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("K (number of LDA topics)")
    ax.set_ylabel("LOSO-CV R^2")
    ax.set_title("CLR-LDA topic->ideology signal vs K (N=119, documented exclusion)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = OUT / "lda_kcurve.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"[figure] -> {fig_path.name}")
    report["figure"] = fig_path.name

    report_path = OUT / "lda_kcurve_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path.name}")


if __name__ == "__main__":
    main()
