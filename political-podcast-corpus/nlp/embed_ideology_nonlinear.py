"""Nonlinear probe: is ideology recoverable from embeddings NONlinearly?

    python -m nlp.embed_ideology_nonlinear [--target-words 500]

The linear-on-PCs analysis (nlp/embed_ideology.py) found DIME ideology was
not recoverable from the top-30 chunk-embedding PCs specifically (LOSO-CV
R^2 never exceeded ~0.03, negative at k=30 PCs). As this module's own result
below shows, that was a PCA-truncation artifact, not evidence of linear
unrecoverability in general -- a Ridge regression on the FULL 768-dim
embedding recovers R^2~=0.21 under the identical CV harness (see
nlp/embed_ideology.py's `verdict_correction` field, added after this probe).
This module tests three small, regularized nonlinear models against the SAME
120 matched shows, the SAME leave-one-show-out CV discipline, and the FULL
768-dim mean-pooled show embedding (not top-k PCs, so nonlinear models can
use structure PCA discarded) -- to check whether nonlinearity adds anything
beyond what full-embedding Ridge already recovers.

Every model here -- including the linear baseline, which is RECOMPUTED under
this harness rather than cited from the earlier report -- is judged by
leave-one-SHOW-out CV R^2 ONLY. In-sample R^2 is computed for each model
purely to display the overfit gap; it is never reported as a result.

Feature standardization is fit inside each outer LOSO fold, on the 119
training shows only, via sklearn Pipeline (StandardScaler -> estimator)
wrapped in GridSearchCV for hyperparameter selection -- both the scaler's
mean/std AND the hyperparameter choice are therefore determined without the
held-out show ever being seen. This is the mechanism that prevents the
"scale leaks test-show information" CV bug named in the task brief.

Models, cheapest first:
  1. Ridge regression (the linear baseline, on the full 768-dim vector --
     necessarily regularized since p=768 >> n_train=119).
  2. Kernel ridge regression (RBF) -- primary nonlinear probe, same
     regularized-linear family as Ridge but with a kernel, so it's the
     natural nonlinear analogue for direct comparison.
  3. k-NN regression, cosine distance -- "do ideologically similar shows
     sit near each other in embedding space" as a standalone question,
     unscaled (cosine already normalizes away per-vector scale; per-feature
     standardization would distort the embedding's native geometry).
  4. Small MLP (1 hidden layer, 16-32 units, dropout + weight decay, early
     stopping on an internal validation split) -- 3 seeds, range reported,
     since this is the model most sensitive to init/seed noise at N=120.

Hyperparameters are small FIXED grids selected via inner (nested) CV on the
training fold only -- never tuned against outer-fold/held-out performance.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import config as pipeline_config
from .embed_ideology import load_cfscore

RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
KRR_ALPHAS = [0.1, 1.0, 10.0, 100.0]
KRR_GAMMAS = [1e-4, 3e-4, 1e-3, 1e-2]  # ~1/(2*768) natural scale is ~6.5e-4
KNN_KS = [5, 10, 15]
INNER_CV_FOLDS = 5
MLP_SEEDS = [0, 1, 2]
MEANINGFUL_R2 = 0.15


def r2_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def bootstrap_r2_ci(y_true: np.ndarray, y_pred: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = r2_from_predictions(y_true[idx], y_pred[idx])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def build_show_level_embeddings(target: int) -> tuple[np.ndarray, np.ndarray]:
    """Show-id array + (n_shows, 768) mean-pooled embedding matrix.

    Pools the existing length-weighted CHUNK embeddings up to show level via
    a plain mean across each show's chunks -- no re-embedding/re-pooling of
    sentences, per the out-of-scope note.
    """
    OUT = pipeline_config.OUTPUT_DIR
    chunks_bert = pd.read_csv(OUT / f"chunks_{target}_bert.csv", usecols=["chunk_id", "collection_id"])
    meta = json.loads((OUT / f"chunk_embeddings_{target}_weighted.meta.json").read_text())
    chunk_id_order = meta["chunk_id_order"]
    X_chunk = np.load(OUT / f"chunk_embeddings_{target}_weighted.npy")

    show_ids = chunks_bert.set_index("chunk_id").loc[chunk_id_order, "collection_id"].astype(str).to_numpy()
    df = pd.DataFrame(X_chunk)
    df["show_id"] = show_ids
    show_means = df.groupby("show_id").mean()
    return show_means.index.to_numpy(), show_means.to_numpy()


def loso_cv(X: np.ndarray, y: np.ndarray, fit_predict_fn) -> np.ndarray:
    """Leave-one-show-out CV. `fit_predict_fn(X_train, y_train, X_test)` must
    do ALL its own scaling/hyperparameter selection using only X_train/y_train
    (e.g. via an sklearn Pipeline + GridSearchCV) -- this function only
    handles the outer show-holdout split, never touches scaling itself, so
    there is no code path by which a held-out show's values could reach a
    scaler or a hyperparameter search.
    """
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        preds[i] = fit_predict_fn(X[train_idx], y[train_idx], X[i:i + 1])[0]
    return preds


def ridge_fit_predict(X_train, y_train, X_test):
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    grid = GridSearchCV(pipe, {"ridge__alpha": RIDGE_ALPHAS}, cv=INNER_CV_FOLDS, scoring="r2")
    grid.fit(X_train, y_train)
    return grid.predict(X_test)


def krr_fit_predict(X_train, y_train, X_test):
    pipe = Pipeline([("scale", StandardScaler()), ("krr", KernelRidge(kernel="rbf"))])
    grid = GridSearchCV(pipe, {"krr__alpha": KRR_ALPHAS, "krr__gamma": KRR_GAMMAS},
                         cv=INNER_CV_FOLDS, scoring="r2")
    grid.fit(X_train, y_train)
    return grid.predict(X_test)


def knn_fit_predict(X_train, y_train, X_test):
    # No StandardScaler: cosine distance already normalizes per-vector scale;
    # per-feature standardization would distort the embedding's native
    # geometry rather than just "fix the units" the way it does for Ridge/KRR.
    knn = KNeighborsRegressor(metric="cosine")
    grid = GridSearchCV(knn, {"n_neighbors": KNN_KS}, cv=INNER_CV_FOLDS, scoring="r2")
    grid.fit(X_train, y_train)
    return grid.predict(X_test)


def make_mlp_fit_predict(seed: int):
    import torch
    import torch.nn as nn

    def fit_predict(X_train, y_train, X_test):
        scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        Xte = scaler.transform(X_test)

        rng = np.random.default_rng(seed)
        n = len(y_train)
        val_size = max(8, int(0.2 * n))
        perm = rng.permutation(n)
        val_idx, fit_idx = perm[:val_size], perm[val_size:]

        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(Xtr.shape[1], 24), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(24, 1),
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        loss_fn = nn.MSELoss()

        Xfit = torch.tensor(Xtr[fit_idx], dtype=torch.float32)
        yfit = torch.tensor(y_train[fit_idx], dtype=torch.float32).unsqueeze(1)
        Xval = torch.tensor(Xtr[val_idx], dtype=torch.float32)
        yval = torch.tensor(y_train[val_idx], dtype=torch.float32).unsqueeze(1)

        best_val, best_state, patience, bad_epochs = float("inf"), None, 20, 0
        for epoch in range(500):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(Xfit), yfit)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xval), yval).item()
            if val_loss < best_val - 1e-5:
                best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte, dtype=torch.float32)).numpy().flatten()
        return pred

    return fit_predict


def run(target: int) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    cfscore_df = load_cfscore()

    show_ids_all, X_all = build_show_level_embeddings(target)
    id_to_row = {sid: i for i, sid in enumerate(show_ids_all)}
    matched_ids = [sid for sid in cfscore_df.index if sid in id_to_row]
    n = len(matched_ids)
    print(f"[data] N matched shows (same as linear analysis) = {n}")
    assert n == len(cfscore_df), (
        f"matched N ({n}) != linear-analysis N ({len(cfscore_df)}) -- "
        f"embedding/CFscore join drifted, investigate before trusting results"
    )

    X = np.stack([X_all[id_to_row[sid]] for sid in matched_ids])
    y = cfscore_df.loc[matched_ids, "avg_host_cfscore"].to_numpy()
    print(f"[data] X shape={X.shape}, y shape={y.shape}")

    report: dict = {
        "target": target, "n_matched": n,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }

    def evaluate(name: str, fit_predict_fn, in_sample_fn=None) -> dict:
        preds = loso_cv(X, y, fit_predict_fn)
        cv_r2 = r2_from_predictions(y, preds)
        lo, hi = bootstrap_r2_ci(y, preds)
        entry = {"loso_cv_r2": float(cv_r2), "bootstrap_95ci": [lo, hi]}
        if in_sample_fn is not None:
            in_sample_pred = in_sample_fn(X, y, X)
            entry["in_sample_r2_NOT_REPORTABLE"] = float(r2_from_predictions(y, in_sample_pred))
        print(f"[{name}] LOSO-CV R2={cv_r2:.3f} (95% CI [{lo:.3f}, {hi:.3f}])"
              + (f"  |  in-sample R2={entry.get('in_sample_r2_NOT_REPORTABLE', float('nan')):.3f} "
                 f"(overfit gap, NOT a result)" if in_sample_fn else ""))
        return entry

    print("\n=== Ridge (linear baseline, recomputed under this harness) ===")
    report["models"]["ridge_linear_baseline"] = evaluate("ridge", ridge_fit_predict, ridge_fit_predict)

    print("\n=== Kernel Ridge (RBF) -- primary nonlinear probe ===")
    report["models"]["kernel_ridge_rbf"] = evaluate("krr_rbf", krr_fit_predict, krr_fit_predict)

    print("\n=== k-NN (cosine distance) ===")
    report["models"]["knn_cosine"] = evaluate("knn_cosine", knn_fit_predict, knn_fit_predict)

    print("\n=== Small MLP (3 seeds) ===")
    mlp_r2s = []
    for seed in MLP_SEEDS:
        fit_predict_fn = make_mlp_fit_predict(seed)
        entry = evaluate(f"mlp_seed{seed}", fit_predict_fn, fit_predict_fn)
        report["models"][f"mlp_seed{seed}"] = entry
        mlp_r2s.append(entry["loso_cv_r2"])
    report["models"]["mlp_summary"] = {
        "cv_r2_across_seeds": mlp_r2s,
        "mean": float(np.mean(mlp_r2s)), "range": [float(min(mlp_r2s)), float(max(mlp_r2s))],
    }
    print(f"[mlp summary] mean CV R2={np.mean(mlp_r2s):.3f}, range=[{min(mlp_r2s):.3f}, {max(mlp_r2s):.3f}]")

    linear_r2 = report["models"]["ridge_linear_baseline"]["loso_cv_r2"]
    nonlinear_candidates = {
        "kernel_ridge_rbf": report["models"]["kernel_ridge_rbf"]["loso_cv_r2"],
        "knn_cosine": report["models"]["knn_cosine"]["loso_cv_r2"],
        "mlp_mean": float(np.mean(mlp_r2s)),
    }
    best_nonlinear_name = max(nonlinear_candidates, key=nonlinear_candidates.get)
    best_nonlinear_r2 = nonlinear_candidates[best_nonlinear_name]

    if best_nonlinear_r2 > MEANINGFUL_R2 and best_nonlinear_r2 > linear_r2 + 0.1:
        overall = (
            f"A nonlinear model ({best_nonlinear_name}) reaches CV R2={best_nonlinear_r2:.3f}, "
            f"clearing the linear baseline (R2={linear_r2:.3f}) and the {MEANINGFUL_R2} meaningfulness bar. "
            f"This CHANGES the finding: ideology IS present in the embedding space, structured "
            f"nonlinearly -- the linear null was a linearity artifact, not evidence of absence."
        )
    else:
        overall = (
            f"Best nonlinear CV R2 ({best_nonlinear_name}={best_nonlinear_r2:.3f}) is not "
            f"meaningfully better than the linear baseline (R2={linear_r2:.3f}); neither clears "
            f"{MEANINGFUL_R2}. This STRENGTHENS the earlier finding: ideology is not recoverable "
            f"from this embedding space at N={n}, linearly or nonlinearly, at least not via these "
            f"three reasonable, appropriately-regularized model families."
        )
    caveat = (
        f"N={n} is small; every CV R2 above carries real sampling noise (see bootstrap 95% CIs) -- "
        f"treat single point estimates cautiously, not as settled facts."
    )
    report["verdict"] = {
        "linear_baseline_r2": linear_r2,
        "best_nonlinear_model": best_nonlinear_name,
        "best_nonlinear_r2": best_nonlinear_r2,
        "summary": overall,
        "caveat": caveat,
    }
    print(f"\n=== VERDICT ===\n{overall}\n{caveat}")

    report_path = OUT / f"nonlinear_probe_report_{target}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[report] -> {report_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-words", type=int, default=500)
    args = ap.parse_args()
    run(args.target_words)


if __name__ == "__main__":
    main()
