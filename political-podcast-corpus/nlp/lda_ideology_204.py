"""Stratify-and-compare re-run of the LDA ideology probe at N=204 coverage.

Reuses the ALREADY-FIT K=75 doc-topic matrix (data/output/lda_doctopic_k75_500.csv)
and the CLR + Ridge + LOSO-CV + permutation machinery from nlp/lda_ideology_clr.py /
nlp/embed_ideology_nonlinear.py -- no LDA refit, no K re-sweep. Only the target
(host cfscore -> ideology_targets_204.csv columns) and the show-set change.

Four conditions (all target the SAME K=75 CLR topic vectors):

  host_clean      host_coverage == "full" only (~84 shows), target = avg_host_cfscore.
                  This is the old N~120 analysis's cleanest subset -- the trustworthy
                  anchor, unchanged in spirit from nlp/lda_ideology_clr.py.
  extended_all    every show with ideology_primary (~204): host cfscore where
                  host_coverage == "full", else guest weighted_mean_cfscore.
                  The full-coverage headline number.
  well_supported  extended_all, but guest-primary rows additionally require
                  n_dime_guests >= 20 (the point where the guest-vs-host correlation
                  sensitivity table peaked). Tests whether thin guest estimates are
                  propping up extended_all.
  guest_uniform   every show with a guest score (ideology_guest_only), using ONLY
                  guest ideology as target even for host_coverage=="full" shows.
                  Tests whether the finding is an artifact of switching measurement
                  instruments across the sample (host for some shows, guest for
                  others) rather than the coverage itself.

Plus one robustness-only column:

  extended_weighted   same shows/target as extended_all, but inverse-variance
                       weighted Ridge: weight=1.0 for host_full rows, weight=
                       min(n_dime_guests, 50)/50 for guest rows (saturating cap,
                       since raw guest count is a noisy reliability proxy and
                       weighting a point estimate on a noisy proxy can do more harm
                       than good -- WLS literature). Reported as confirmation only,
                       never as the headline.

No new K sweep, no new embedding/LDA fit, no measurement-error model -- per task
scope.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import config as pipeline_config
from .embed_ideology_nonlinear import (
    RIDGE_ALPHAS,
    INNER_CV_FOLDS,
    bootstrap_r2_ci,
    loso_cv,
    r2_from_predictions,
    ridge_fit_predict,
)
from .lda_ideology_clr import clr

K = 75
CAP = 50  # saturating cap for inverse-variance guest weight


def loso_cv_weighted(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Same outer show-holdout contract as loso_cv, but forwards sample_weight
    to the Ridge step only (StandardScaler ignores it) -- weights come from
    the training fold only, never touch the held-out show."""
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        idx = np.array([j for j in range(n) if j != i])
        pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
        grid = GridSearchCV(pipe, {"ridge__alpha": RIDGE_ALPHAS}, cv=INNER_CV_FOLDS, scoring="r2")
        grid.fit(X[idx], y[idx], ridge__sample_weight=w[idx])
        preds[i] = grid.predict(X[i:i + 1])[0]
    return preds


def load_show_topics() -> pd.DataFrame:
    OUT = pipeline_config.OUTPUT_DIR
    chunks = pd.read_csv(OUT / "chunks_500_nolemma.csv", usecols=["chunk_id", "collection_id"])
    chunks["collection_id"] = chunks["collection_id"].astype(str)
    doc_topic = pd.read_csv(OUT / f"lda_doctopic_k{K}_500.csv")
    doc_topic = doc_topic.merge(chunks, on="chunk_id")
    cols = [c for c in doc_topic.columns if c.startswith("T")]
    show_topics = doc_topic.groupby("collection_id")[cols].mean()
    return show_topics, cols


def probe(joined: pd.DataFrame, cols: list[str], target_col: str, label: str,
          weight_col: str | None = None) -> dict:
    X_raw = joined[cols].to_numpy()
    assert (X_raw > 0).all(), f"[{label}] non-positive topic proportion found, CLR needs epsilon"
    X = clr(X_raw)
    y = joined[target_col].to_numpy(dtype=float)
    n = len(y)

    if weight_col is None:
        preds = loso_cv(X, y, ridge_fit_predict)
    else:
        w = joined[weight_col].to_numpy(dtype=float)
        preds = loso_cv_weighted(X, y, w)

    cv_r2 = r2_from_predictions(y, preds)
    lo, hi = bootstrap_r2_ci(y, preds)
    pear_r, pear_p = stats.pearsonr(y, preds)

    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    if weight_col is None:
        perm_preds = loso_cv(X, y_shuf, ridge_fit_predict)
    else:
        perm_preds = loso_cv_weighted(X, y_shuf, w)
    perm_r2 = r2_from_predictions(y_shuf, perm_preds)

    print(f"[{label}] N={n}  LOSO-CV R2={cv_r2:.4f}  95%CI=[{lo:.4f},{hi:.4f}]  "
          f"pred-actual r={pear_r:.4f}  perm R2={perm_r2:.4f}")

    return {
        "condition": label, "n": n, "target": target_col,
        "loso_cv_r2": float(cv_r2), "ci_95": [lo, hi],
        "pearson_r": float(pear_r), "pearson_p": float(pear_p),
        "permutation_r2": float(perm_r2),
    }


def run() -> None:
    OUT = pipeline_config.OUTPUT_DIR
    targets = pd.read_csv(OUT / "ideology_targets_204.csv", dtype={"show_id": str}).set_index("show_id")
    show_topics, cols = load_show_topics()

    joined_all = show_topics.join(targets, how="inner")
    print(f"[join] doc-topic matrix rows matched to targets: {len(joined_all)} / {len(targets)} canonical shows")
    missing = set(targets.index) - set(joined_all.index)
    if missing:
        print(f"[join] {len(missing)} canonical show(s) missing from LDA doc-topic matrix "
              f"(no sampled chunks / not in corpus.csv chunking): "
              f"{[targets.loc[m, 'show'] for m in missing]}")

    results = []

    # host_clean
    hc = joined_all[joined_all["ideology_primary_source"] == "host_full"].dropna(subset=["avg_host_cfscore"])
    results.append(probe(hc, cols, "avg_host_cfscore", "host_clean"))

    # extended_all
    ea = joined_all.dropna(subset=["ideology_primary"])
    results.append(probe(ea, cols, "ideology_primary", "extended_all"))

    # well_supported: host_full as-is, guest rows require n_dime_guests >= 20
    ws_mask = (joined_all["ideology_primary_source"] == "host_full") | (
        (joined_all["ideology_primary_source"] == "guest") & (joined_all["n_dime_guests"] >= 20)
    )
    ws = joined_all[ws_mask].dropna(subset=["ideology_primary"])
    results.append(probe(ws, cols, "ideology_primary", "well_supported"))

    # guest_uniform: guest score for everyone who has one, ignore host entirely
    gu = joined_all.dropna(subset=["ideology_guest_only"])
    results.append(probe(gu, cols, "ideology_guest_only", "guest_uniform"))

    # extended_weighted: robustness-only, same shows/target as extended_all
    ea_w = ea.copy()
    ea_w["ivw_weight"] = np.where(
        ea_w["ideology_primary_source"] == "host_full",
        1.0,
        np.minimum(ea_w["n_dime_guests"], CAP) / CAP,
    )
    results.append(probe(ea_w, cols, "ideology_primary", "extended_weighted", weight_col="ivw_weight"))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arm": "LDA topics (CLR, K=75)",
        "k": K,
        "note": "Reused already-fit K=75 doc-topic matrix; no K re-sweep, no LDA refit.",
        "results": results,
    }
    report_path = OUT / "lda_ideology_204_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path.name}")

    print("\n=== LDA (CLR, K=75) -- host_clean vs extended_all vs well_supported vs guest_uniform vs extended_weighted ===")
    print(f"{'condition':<20}{'N':<6}{'LOSO-CV R2':<13}{'95% CI':<20}{'pred-actual r':<15}{'perm R2':<10}")
    for r in results:
        ci = f"[{r['ci_95'][0]:.3f},{r['ci_95'][1]:.3f}]"
        print(f"{r['condition']:<20}{r['n']:<6}{r['loso_cv_r2']:<13.3f}{ci:<20}{r['pearson_r']:<15.3f}{r['permutation_r2']:<10.3f}")


if __name__ == "__main__":
    run()
