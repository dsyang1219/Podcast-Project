"""Step 2 of the register-quantification task: regress each of the top ~10
chunk-level PCs on the 6 a-priori register features (nlp/register_features.py),
report per-PC CV R^2 (the core result) and standardized coefficients (which
features drive each PC).

Reuses PC1..PC10 from the ALREADY-COMPUTED cache
data/output/embed_dimensions_feature_table_500.pkl (same primary-anisotropy-
variant chunk-level PCA nlp/embed_dimensions.py / nlp/embed_rank.py /
nlp/embed_ideology.py all share) -- no PCA refit here.

Linear regression (plain OLS via Ridge with a small fixed alpha=1.0 for
numerical stability only, not for meaningful shrinkage -- 6 features on
40k+ rows has essentially zero overfit risk) with K-fold CV (chunk-level
folds, not LOSO -- there is no natural show-level target here, this is a
chunk-level structural question) to report held-out R^2 rather than
in-sample R^2, per task requirement.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from pipeline import config as pipeline_config

TARGET = 500
N_PCS = 10
N_FOLDS = 10
RIDGE_ALPHA = 1.0  # numerical-stability only; n>>p makes this ~unregularized OLS
REGISTER_HIGH_R2 = 0.3

FEATURES = [
    "citation_density", "abstraction", "present_tense_news_markers",
    "entity_mix", "sentence_complexity", "first_vs_third_person",
]


def r2_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def run() -> None:
    OUT = pipeline_config.OUTPUT_DIR
    pc_df = pd.read_pickle(OUT / f"embed_dimensions_feature_table_{TARGET}.pkl")
    pc_cols = [f"PC{i}" for i in range(1, N_PCS + 1)]
    pc_df = pc_df[["chunk_id"] + pc_cols]
    print(f"[pcs] loaded cached PC1..PC{N_PCS} for {len(pc_df)} chunks (no PCA refit)")

    reg_df = pd.read_csv(OUT / f"register_features_{TARGET}.csv")
    print(f"[register] loaded {len(reg_df)} chunks x {len(FEATURES)} a-priori features")

    df = pc_df.merge(reg_df[["chunk_id"] + FEATURES], on="chunk_id", how="inner")
    before = len(df)
    df = df.dropna(subset=FEATURES + pc_cols)
    print(f"[join] N={len(df)} (dropped {before - len(df)} rows with any missing feature/PC)")

    X_raw = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    rng = np.random.default_rng(0)

    results = []
    for pc in pc_cols:
        y = df[pc].to_numpy(dtype=float)
        model = Ridge(alpha=RIDGE_ALPHA)
        cv_preds = cross_val_predict(model, X, y, cv=kf)
        cv_r2 = r2_from_predictions(y, cv_preds)

        y_shuf = rng.permutation(y)
        perm_preds = cross_val_predict(model, X, y_shuf, cv=kf)
        perm_r2 = r2_from_predictions(y_shuf, perm_preds)

        model.fit(X, y)
        coefs = dict(zip(FEATURES, model.coef_.tolist()))

        register_explained = cv_r2 > REGISTER_HIGH_R2
        results.append({
            "pc": pc, "n": len(y), "cv_r2": float(cv_r2), "permutation_r2": float(perm_r2),
            "standardized_coefficients": coefs, "register_explained": bool(register_explained),
        })
        top_feats = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_str = ", ".join(f"{f}={c:+.3f}" for f, c in top_feats[:3])
        print(f"[{pc}] CV R2={cv_r2:.4f}  perm R2={perm_r2:.4f}  "
              f"{'REGISTER-EXPLAINED' if register_explained else 'not register-explained'}  "
              f"top coefs: {top_str}")

    n_register_explained = sum(1 for r in results if r["register_explained"])
    messy = n_register_explained == 0
    print(f"\n[summary] {n_register_explained}/{N_PCS} PCs register-explained (CV R2 > {REGISTER_HIGH_R2})")
    print(f"[ICA trigger] Step 2 is {'MESSY -- ICA robustness check should run' if messy else 'clean -- ICA not needed'}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_words": TARGET, "n_pcs": N_PCS, "n_folds": N_FOLDS,
        "features": FEATURES, "register_high_r2_threshold": REGISTER_HIGH_R2,
        "n_matched_chunks": len(df),
        "per_pc": results,
        "n_register_explained": n_register_explained,
        "ica_triggered": messy,
    }
    report_path = OUT / "register_quant_step2_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path}")

    print(f"\n{'PC':<6}{'N':<8}{'CV R2':<10}{'perm R2':<10}{'explained?':<12}")
    for r in results:
        print(f"{r['pc']:<6}{r['n']:<8}{r['cv_r2']:<10.4f}{r['permutation_r2']:<10.4f}"
              f"{'YES' if r['register_explained'] else 'no':<12}")


if __name__ == "__main__":
    run()
