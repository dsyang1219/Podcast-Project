"""Step 5 (CONDITIONAL) of the register-quantification task: ICA robustness
check, run because Step 2 came out messy -- 0/10 PCs cleared the R^2>0.3
register-explained bar, and per-PC standardized coefficients were smeared
thin across all 6 features rather than concentrated (see
data/output/register_quant_step2_report.json, ica_triggered=true).

Runs sklearn FastICA on the SAME embedding matrix the PCA was computed on
(centered-only anisotropy-correction variant: X - X.mean(axis=0), the
"primary anisotropy variant" nlp/embed_dimensions.py / nlp/embed_rank.py /
nlp/embed_ideology.py all share), same n_components=10 as the retained PCA
dims. Then repeats the EXACT Step 2 regression (register features -> each
component, Ridge alpha=1.0 for numerical stability only, 10-fold CV,
permutation null) on the independent components instead of the principal
components, and reports whether ICA maps onto register features more
cleanly (higher per-axis R^2, sparser/more concentrated coefficients).

No new embedding fit -- reuses data/output/chunk_embeddings_500_weighted.npy
as-is; only the post-hoc linear decomposition (ICA instead of PCA) changes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from pipeline import config as pipeline_config
from .register_pc_regression import FEATURES, N_FOLDS, REGISTER_HIGH_R2, RIDGE_ALPHA, r2_from_predictions

TARGET = 500
N_COMPONENTS = 10  # matches N_PCS in register_pc_regression.py (retained PCA dims)
ICA_SEED = 0


def load_centered_embeddings() -> tuple[np.ndarray, list[str]]:
    OUT = pipeline_config.OUTPUT_DIR
    meta = json.loads((OUT / f"chunk_embeddings_{TARGET}_weighted.meta.json").read_text())
    X = np.load(OUT / f"chunk_embeddings_{TARGET}_weighted.npy")
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc, meta["chunk_id_order"]


def sparsity_gini(coefs: np.ndarray) -> float:
    """Gini coefficient of |standardized coefficients| across the 6 features
    for one component -- 0 = perfectly spread evenly across all features,
    1 = all weight on a single feature. Used to compare "sparser" mapping
    between PCA and ICA per the task's acceptance criterion."""
    a = np.sort(np.abs(coefs))
    n = len(a)
    if a.sum() == 0:
        return 0.0
    cum = np.cumsum(a)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def run() -> None:
    OUT = pipeline_config.OUTPUT_DIR
    print("[embeddings] loading + centering (same anisotropy-correction variant PCA used)...")
    Xc, chunk_id_order = load_centered_embeddings()
    print(f"[embeddings] shape={Xc.shape}")

    print(f"[ica] fitting FastICA n_components={N_COMPONENTS} (seed={ICA_SEED})...")
    t0 = datetime.now()
    ica = FastICA(n_components=N_COMPONENTS, random_state=ICA_SEED, whiten="unit-variance", max_iter=1000)
    S = ica.fit_transform(Xc)  # (n_chunks, N_COMPONENTS) independent-component scores
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[ica] done in {elapsed:.1f}s, n_iter_={getattr(ica, 'n_iter_', 'n/a')}")

    ic_cols = [f"IC{i + 1}" for i in range(N_COMPONENTS)]
    ic_df = pd.DataFrame(S, columns=ic_cols)
    ic_df["chunk_id"] = chunk_id_order

    reg_df = pd.read_csv(OUT / f"register_features_{TARGET}.csv")
    df = ic_df.merge(reg_df[["chunk_id"] + FEATURES], on="chunk_id", how="inner")
    df = df.dropna(subset=FEATURES + ic_cols)
    print(f"[join] N={len(df)}")

    X_raw = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xf = scaler.fit_transform(X_raw)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    rng = np.random.default_rng(0)

    results = []
    for ic in ic_cols:
        y = df[ic].to_numpy(dtype=float)
        model = Ridge(alpha=RIDGE_ALPHA)
        cv_preds = cross_val_predict(model, Xf, y, cv=kf)
        cv_r2 = r2_from_predictions(y, cv_preds)

        y_shuf = rng.permutation(y)
        perm_preds = cross_val_predict(model, Xf, y_shuf, cv=kf)
        perm_r2 = r2_from_predictions(y_shuf, perm_preds)

        model.fit(Xf, y)
        coefs = dict(zip(FEATURES, model.coef_.tolist()))
        gini = sparsity_gini(model.coef_)

        register_explained = cv_r2 > REGISTER_HIGH_R2
        results.append({
            "component": ic, "n": len(y), "cv_r2": float(cv_r2), "permutation_r2": float(perm_r2),
            "standardized_coefficients": coefs, "coefficient_gini": gini,
            "register_explained": bool(register_explained),
        })
        top_feats = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_str = ", ".join(f"{f}={c:+.3f}" for f, c in top_feats[:3])
        print(f"[{ic}] CV R2={cv_r2:.4f}  perm R2={perm_r2:.4f}  gini={gini:.3f}  "
              f"{'REGISTER-EXPLAINED' if register_explained else 'not register-explained'}  "
              f"top coefs: {top_str}")

    n_register_explained = sum(1 for r in results if r["register_explained"])
    mean_r2_ica = float(np.mean([r["cv_r2"] for r in results]))
    mean_gini_ica = float(np.mean([r["coefficient_gini"] for r in results]))

    pca_report = json.loads((OUT / "register_quant_step2_report.json").read_text())
    pca_r2s = [r["cv_r2"] for r in pca_report["per_pc"]]
    mean_r2_pca = float(np.mean(pca_r2s))
    pca_ginis = [sparsity_gini(np.array(list(r["standardized_coefficients"].values())))
                 for r in pca_report["per_pc"]]
    mean_gini_pca = float(np.mean(pca_ginis))

    cleaner = (mean_r2_ica > mean_r2_pca + 0.02) or (mean_gini_ica > mean_gini_pca + 0.02)
    comparison_verdict = (
        f"ICA {'maps MORE cleanly onto register features' if cleaner else 'does NOT map more cleanly onto register features'} "
        f"than PCA: mean CV R2 PCA={mean_r2_pca:.3f} vs ICA={mean_r2_ica:.3f}; "
        f"mean coefficient-sparsity (Gini) PCA={mean_gini_pca:.3f} vs ICA={mean_gini_ica:.3f}."
    )
    print(f"\n[comparison] {comparison_verdict}")
    print(f"[summary] {n_register_explained}/{N_COMPONENTS} ICs register-explained (CV R2 > {REGISTER_HIGH_R2}) "
          f"vs {pca_report['n_register_explained']}/{len(pca_r2s)} PCs")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reason_triggered": "Step 2 messy: 0/10 PCs register-explained, coefficients smeared thin across features",
        "target_words": TARGET, "n_components": N_COMPONENTS, "n_folds": N_FOLDS,
        "n_matched_chunks": len(df),
        "per_component": results,
        "n_register_explained": n_register_explained,
        "mean_cv_r2_ica": mean_r2_ica, "mean_cv_r2_pca": mean_r2_pca,
        "mean_coefficient_gini_ica": mean_gini_ica, "mean_coefficient_gini_pca": mean_gini_pca,
        "comparison_verdict": comparison_verdict,
    }
    report_path = OUT / "register_quant_step5_ica_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path}")

    print(f"\n{'IC':<6}{'N':<8}{'CV R2':<10}{'perm R2':<10}{'gini':<8}{'explained?':<12}")
    for r in results:
        print(f"{r['component']:<6}{r['n']:<8}{r['cv_r2']:<10.4f}{r['permutation_r2']:<10.4f}"
              f"{r['coefficient_gini']:<8.3f}{'YES' if r['register_explained'] else 'no':<12}")


if __name__ == "__main__":
    run()
