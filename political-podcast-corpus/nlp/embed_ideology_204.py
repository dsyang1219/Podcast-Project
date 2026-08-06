"""Stratify-and-compare re-run of the EMBEDDING ideology probe at N=204 coverage.

Mirrors nlp/lda_ideology_204.py exactly (same 5 conditions, same target file,
same Ridge/LOSO-CV/permutation/IVW machinery) but swaps the representation:
show-level full 768d mean-pooled embeddings (target_words=500, the length-
weighted pooling already on record as the embedding baseline in
lda_ideology_clr_report.json's embedding_baseline_reference, R2=0.21 @ N=120)
instead of K=75 CLR topic vectors. No re-embedding, no re-pooling -- reuses
data/output/chunk_embeddings_500_weighted.npy as-is, per the out-of-scope note
(a long-context embedding re-run is a separate future task).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from pipeline import config as pipeline_config
from .embed_ideology_nonlinear import (
    bootstrap_r2_ci,
    build_show_level_embeddings,
    loso_cv,
    r2_from_predictions,
    ridge_fit_predict,
)
from .lda_ideology_204 import loso_cv_weighted

TARGET_WORDS = 500


def probe(joined: pd.DataFrame, X_all: np.ndarray, id_to_row: dict, target_col: str,
          label: str, weight_col: str | None = None) -> dict:
    ids = list(joined.index)
    X = np.stack([X_all[id_to_row[sid]] for sid in ids])
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

    show_ids_all, X_all = build_show_level_embeddings(TARGET_WORDS)
    id_to_row = {sid: i for i, sid in enumerate(show_ids_all)}

    available = [sid for sid in targets.index if sid in id_to_row]
    joined_all = targets.loc[available]
    print(f"[join] embedding matrix rows matched to targets: {len(joined_all)} / {len(targets)} canonical shows")
    missing = set(targets.index) - set(available)
    if missing:
        print(f"[join] {len(missing)} canonical show(s) missing from embedding matrix: "
              f"{[targets.loc[m, 'show'] for m in missing]}")

    results = []

    hc = joined_all[joined_all["ideology_primary_source"] == "host_full"].dropna(subset=["avg_host_cfscore"])
    results.append(probe(hc, X_all, id_to_row, "avg_host_cfscore", "host_clean"))

    ea = joined_all.dropna(subset=["ideology_primary"])
    results.append(probe(ea, X_all, id_to_row, "ideology_primary", "extended_all"))

    ws_mask = (joined_all["ideology_primary_source"] == "host_full") | (
        (joined_all["ideology_primary_source"] == "guest") & (joined_all["n_dime_guests"] >= 20)
    )
    ws = joined_all[ws_mask].dropna(subset=["ideology_primary"])
    results.append(probe(ws, X_all, id_to_row, "ideology_primary", "well_supported"))

    gu = joined_all.dropna(subset=["ideology_guest_only"])
    results.append(probe(gu, X_all, id_to_row, "ideology_guest_only", "guest_uniform"))

    ea_w = ea.copy()
    ea_w["ivw_weight"] = np.where(
        ea_w["ideology_primary_source"] == "host_full",
        1.0,
        np.minimum(ea_w["n_dime_guests"], 50) / 50,
    )
    results.append(probe(ea_w, X_all, id_to_row, "ideology_primary", "extended_weighted", weight_col="ivw_weight"))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "arm": "embeddings (768d full mean-pooled, weighted pooling, target_words=500)",
        "note": "Reused existing chunk_embeddings_500_weighted.npy; no re-embedding, no re-pooling.",
        "results": results,
    }
    report_path = OUT / "embed_ideology_204_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path.name}")

    print("\n=== Embeddings (768d full) -- host_clean vs extended_all vs well_supported vs guest_uniform vs extended_weighted ===")
    print(f"{'condition':<20}{'N':<6}{'LOSO-CV R2':<13}{'95% CI':<20}{'pred-actual r':<15}{'perm R2':<10}")
    for r in results:
        ci = f"[{r['ci_95'][0]:.3f},{r['ci_95'][1]:.3f}]"
        print(f"{r['condition']:<20}{r['n']:<6}{r['loso_cv_r2']:<13.3f}{ci:<20}{r['pearson_r']:<15.3f}{r['permutation_r2']:<10.3f}")


if __name__ == "__main__":
    run()
