"""Follow-up to nlp/lda_ideology.py: CLR-transform + one documented exclusion.

Reuses the ALREADY-FIT doc-topic matrices (data/output/lda_doctopic_k{20,30,50}
_500.csv) -- no LDA refit, this is a probe-only rerun so it's fast.

Two changes from the original minimal-arm run:

1. CLR (centered log-ratio) transform on the show-level topic-proportion
   vector before Ridge. Topic proportions are compositional (each row sums to
   1), which is the textbook case for CLR (Aitchison 1982): clr(x)_i =
   log(x_i) - mean_j(log(x_j)). tomotopy's doc-topic distributions are
   Dirichlet-smoothed (alpha=0.1 symmetric prior), so every component is
   strictly positive -- verified on the saved matrices before writing this,
   no epsilon/pseudocount needed. CLR is applied to the SHOW-level mean
   proportions (Step 2 of the original arm is unchanged: raw per-chunk
   proportions are still arithmetic-meaned per show, matching the embedding
   arm's show-level-primary unit) -- CLR is a Step-3 input transform, not a
   change to aggregation. Note this means the show-level CLR centroid is not
   exactly the Aitchison-geometry centroid (which would CLR each chunk first,
   then arithmetic-mean); that stricter version is a possible further
   refinement, not done here.

2. One documented exclusion: "The Moment with Jorge Ramos and Paola Ramos"
   (show_id 1839861250) is dropped from the CFscore join for this arm only.
   Diagnosis (see data/output/lda_ideology_report.json's per-K
   loso_outlier_diagnostic from the original run): this show was the single
   dominant driver of the K=20/K=50 LOSO blowups (e.g. true CFscore -1.02,
   predicted +8.5 at K=20, +19.0 at K=50) at EVERY tested K, consistent with
   its topic-proportion vector being a genuine outlier -- plausibly because
   its bilingual Spanish/English content sits far outside the mostly-English
   corpus vocabulary the LDA model and DF-pruned vocab were built from, not
   random noise. This is a one-show, documented, diagnosis-driven exclusion,
   not a general outlier-trimming policy -- N drops from 120 to 119 for this
   arm ONLY; the embedding-arm baseline (R2=0.21, N=120) is unchanged and
   still cited from record, so this comparison has a 1-show N mismatch,
   flagged explicitly in the output rather than silently glossed over.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config as pipeline_config
from .embed_ideology import load_cfscore
from .embed_ideology_nonlinear import bootstrap_r2_ci, loso_cv, r2_from_predictions, ridge_fit_predict

KS = [20, 30, 50]
EXCLUDED_SHOW_ID = "1839861250"
EXCLUDED_SHOW_NAME = "The Moment with Jorge Ramos and Paola Ramos"
EXCLUDED_REASON = (
    "Single dominant driver of LOSO-CV blowups at every tested K in the original "
    "raw-proportions run (K=20: pred=+8.49 vs true=-1.02; K=30: pred=-6.06; K=50: "
    "pred=+18.95) -- plausibly bilingual Spanish/English content producing a "
    "topic-proportion vector far outside the mostly-English corpus vocabulary. "
    "One-show, documented, diagnosis-driven exclusion for this arm only."
)


def clr(X: np.ndarray) -> np.ndarray:
    logX = np.log(X)
    return logX - logX.mean(axis=1, keepdims=True)


def run() -> None:
    OUT = pipeline_config.OUTPUT_DIR
    cfscore_df = load_cfscore()
    cfscore_df = cfscore_df.drop(index=EXCLUDED_SHOW_ID)
    print(f"[cfscore] N genuine minus 1 documented exclusion ({EXCLUDED_SHOW_NAME}) = {len(cfscore_df)}")

    chunks = pd.read_csv(OUT / "chunks_500_nolemma.csv", usecols=["chunk_id", "collection_id"])
    chunks["collection_id"] = chunks["collection_id"].astype(str)

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "change_from_original_arm": [
            "CLR transform on show-level topic proportions before Ridge (compositional-data-correct)",
            f"documented exclusion: {EXCLUDED_SHOW_NAME} (show_id {EXCLUDED_SHOW_ID}): {EXCLUDED_REASON}",
        ],
        "n_matched_shows": len(cfscore_df),
        "note_n_mismatch": (
            "N=119 for this arm (120 minus 1 documented exclusion) vs embedding baseline's N=120 -- "
            "embedding arm NOT rerun, cited from record per task scope."
        ),
        "embedding_baseline_reference": {
            "representation": "embedding (768d, full mean-pooled)", "unit": "show", "model": "Ridge",
            "loso_cv_r2": 0.21, "n": 120, "source": "record",
        },
        "ks": KS, "per_k": {},
    }

    head_to_head = [{
        "representation": "embedding (768d)", "unit": "show", "model": "Ridge", "n": 120,
        "loso_cv_r2": 0.21, "ci_95": None, "permutation_r2": None, "source": "record",
    }]

    for k in KS:
        doc_topic = pd.read_csv(OUT / f"lda_doctopic_k{k}_500.csv")
        doc_topic = doc_topic.merge(chunks, on="chunk_id")
        cols = [c for c in doc_topic.columns if c.startswith("T")]
        show_topics = doc_topic.groupby("collection_id")[cols].mean()

        joined = show_topics.join(cfscore_df, how="inner")
        n_matched = len(joined)
        X_raw = joined[cols].to_numpy()
        assert (X_raw > 0).all(), "non-positive topic proportion found, CLR needs epsilon"
        X = clr(X_raw)
        y = joined["avg_host_cfscore"].to_numpy()
        print(f"\n=== K={k} ===  N matched={n_matched}")

        preds = loso_cv(X, y, ridge_fit_predict)
        cv_r2 = r2_from_predictions(y, preds)
        lo, hi = bootstrap_r2_ci(y, preds)

        rng = np.random.default_rng(0)
        y_shuf = rng.permutation(y)
        perm_preds = loso_cv(X, y_shuf, ridge_fit_predict)
        perm_r2 = r2_from_predictions(y_shuf, perm_preds)

        print(f"[probe k={k}] CLR+Ridge LOSO-CV R2={cv_r2:.4f} (95% CI [{lo:.4f}, {hi:.4f}])  "
              f"permutation R2={perm_r2:.4f}")

        report["per_k"][str(k)] = {
            "k": k, "n_matched_shows": n_matched,
            "loso_cv_r2": float(cv_r2), "bootstrap_95ci": [lo, hi], "permutation_r2": float(perm_r2),
        }
        head_to_head.append({
            "representation": f"LDA topics CLR (K={k}, -1 show)", "unit": "show", "model": "Ridge",
            "n": n_matched, "loso_cv_r2": float(cv_r2), "ci_95": [lo, hi],
            "permutation_r2": float(perm_r2), "source": "computed",
        })

    report["head_to_head"] = head_to_head
    best = max(report["per_k"].values(), key=lambda r: r["loso_cv_r2"])
    embed_r2 = 0.21
    best_r2 = best["loso_cv_r2"]
    if best_r2 > embed_r2 + 0.05:
        verdict = f"topic >> style: best K={best['k']}, R2={best_r2:.3f} vs embedding R2={embed_r2:.2f}."
    elif abs(best_r2 - embed_r2) <= 0.08:
        verdict = f"topic ~= style: best K={best['k']}, R2={best_r2:.3f} vs embedding R2={embed_r2:.2f}."
    else:
        verdict = f"topic << style: best K={best['k']}, R2={best_r2:.3f} vs embedding R2={embed_r2:.2f}."
    report["verdict"] = verdict
    print(f"\n=== VERDICT ===\n{verdict}")

    report_path = OUT / "lda_ideology_clr_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path.name}")

    print("\n=== HEAD-TO-HEAD (CLR + documented exclusion) ===")
    print(f"{'representation':<32}{'unit':<7}{'n':<5}{'LOSO-CV R2':<13}{'95% CI':<20}{'perm R2':<10}")
    for row in head_to_head:
        ci = f"[{row['ci_95'][0]:.3f},{row['ci_95'][1]:.3f}]" if row["ci_95"] else "record"
        perm = f"{row['permutation_r2']:.3f}" if row["permutation_r2"] is not None else "n/a"
        print(f"{row['representation']:<32}{row['unit']:<7}{row['n']:<5}{row['loso_cv_r2']:<13.3f}{ci:<20}{perm:<10}")


if __name__ == "__main__":
    run()
