"""How high can the K=50 ideology R2 be pushed by tuning? (exploratory)

READ THIS BEFORE QUOTING ANY NUMBER FROM HERE
----------------------------------------------
Every result in this file is OUTCOME-SELECTED. Each variant is scored on the
same LOSO-CV R2 used to rank them, so the winner is chosen by the metric it is
then reported on. That is precisely the circularity this project removed from K
selection (K now comes from coherence, not R2) and the same failure mode that
made post-hoc topic dropping read 0.451 -> 0.376 when the true ad contribution
was +0.001.

The maximum here is an UPPER BOUND on what tuning can extract, not an estimate
of the signal. The honest version requires nested CV -- variant selection inside
each LOSO fold -- which would cost a factor of ~N more compute and would land
LOWER than anything printed here.

What IS legitimate: the per-fold hyperparameter search inside each estimator
(ridge alpha, KRR alpha/gamma) already happens within the training split, so the
individual variants are honest estimates OF THEMSELVES. Only the act of picking
the best across variants is contaminated.

Outputs are written with selected_by_outcome=true so they cannot be mistaken for
a reportable result.

    .venv/bin/python -m nlp.adspan_tuning_ceiling --k 50
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .adspan_phase_c import paths_for
from .adspan_target_robustness import build_targets
from .embed_ideology_nonlinear import (
    bootstrap_r2_ci,
    knn_fit_predict,
    krr_fit_predict,
    loso_cv,
    r2_from_predictions,
    ridge_fit_predict,
)
from .lda_ideology_clr import clr

warnings.filterwarnings("ignore")


# ------------------------------------------------------------- transforms --
def t_clr(P):   return clr(np.where(P <= 0, 1e-12, P))          # the baseline
def t_raw(P):   return P
def t_sqrt(P):  return np.sqrt(P)                                # Hellinger-ish
def t_log(P):   return np.log(np.where(P <= 0, 1e-12, P))
def t_logit(P):
    q = np.clip(P, 1e-9, 1 - 1e-9)
    return np.log(q / (1 - q))


TRANSFORMS = {"clr": t_clr, "raw": t_raw, "sqrt": t_sqrt, "log": t_log,
              "logit": t_logit}


# ------------------------------------------------------------- estimators --
def enet_fit_predict(Xtr, ytr, Xte):
    p = make_pipeline(StandardScaler(),
                       ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5, max_iter=5000, random_state=0))
    p.fit(Xtr, ytr)
    return p.predict(Xte)


def rf_fit_predict(Xtr, ytr, Xte):
    m = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=4)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def gb_fit_predict(Xtr, ytr, Xte):
    m = GradientBoostingRegressor(random_state=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def svr_fit_predict(Xtr, ytr, Xte):
    p = make_pipeline(StandardScaler(), SVR(kernel="rbf"))
    g = GridSearchCV(p, {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 1e-3, 1e-2]},
                      cv=5, scoring="r2")
    g.fit(Xtr, ytr)
    return g.predict(Xte)


def pls_fit_predict(Xtr, ytr, Xte):
    n = min(10, Xtr.shape[1])
    g = GridSearchCV(PLSRegression(), {"n_components": list(range(2, n + 1, 2))},
                      cv=5, scoring="r2")
    g.fit(Xtr, ytr)
    return np.asarray(g.predict(Xte)).ravel()


ESTIMATORS = {
    "ridge": ridge_fit_predict,       # the baseline
    "krr_rbf": krr_fit_predict,
    "elasticnet": enet_fit_predict,
    "svr_rbf": svr_fit_predict,
    "pls": pls_fit_predict,
    "knn": knn_fit_predict,
    "rf": rf_fit_predict,
    "gbr": gb_fit_predict,
}

AGGREGATIONS = {"mean": "mean", "median": "median"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--corpus", default="clean")
    ap.add_argument("--target", default="extended_all")
    args = ap.parse_args()

    P = paths_for(args.corpus)
    K = args.k
    dt = pd.read_csv(P["doctopic_fmt"].format(k=K),
                      dtype={"collection_id": str, "chunk_id": str})
    cols = [f"T{i}" for i in range(K)]
    y_series = build_targets()[args.target]

    print(f"[ceiling] EXPLORATORY -- every number below is OUTCOME-SELECTED")
    print(f"[ceiling] corpus={args.corpus} K={K} target={args.target}")

    rows = []
    for agg_name, agg in AGGREGATIONS.items():
        show = dt.groupby("collection_id")[cols].agg(agg)
        j = show.join(y_series.rename("y"), how="inner").dropna(subset=["y"])
        Praw = j[cols].to_numpy(dtype=float)
        y = j["y"].to_numpy(dtype=float)
        for tname, tf in TRANSFORMS.items():
            X = tf(Praw)
            if not np.all(np.isfinite(X)):
                continue
            for ename, fn in ESTIMATORS.items():
                try:
                    preds = loso_cv(X, y, fn)
                    r2 = r2_from_predictions(y, preds)
                except Exception as e:
                    print(f"  [skip] {agg_name}/{tname}/{ename}: {type(e).__name__}")
                    continue
                rows.append({"agg": agg_name, "transform": tname,
                              "estimator": ename, "r2": float(r2), "n": int(len(y))})
                print(f"  {agg_name:<7}{tname:<7}{ename:<12} R2={r2:+.4f}", flush=True)

    df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    base = df[(df["agg"] == "mean") & (df["transform"] == "clr")
               & (df["estimator"] == "ridge")]["r2"]
    baseline = float(base.iloc[0]) if len(base) else float("nan")

    print(f"\n[ceiling] BASELINE (mean/clr/ridge, the reported config): {baseline:+.4f}")
    print(f"[ceiling] TOP 12 of {len(df)} variants:")
    for _, r in df.head(12).iterrows():
        print(f"  {r['r2']:+.4f}  {r['agg']}/{r['transform']}/{r['estimator']}"
              f"   (+{r['r2'] - baseline:.4f} vs baseline)")

    # A CI on the winner, so its width is visible next to its point estimate.
    best = df.iloc[0]
    show = dt.groupby("collection_id")[cols].agg(AGGREGATIONS[best["agg"]])
    j = show.join(y_series.rename("y"), how="inner").dropna(subset=["y"])
    X = TRANSFORMS[best["transform"]](j[cols].to_numpy(dtype=float))
    y = j["y"].to_numpy(dtype=float)
    preds = loso_cv(X, y, ESTIMATORS[best["estimator"]])
    lo, hi = bootstrap_r2_ci(y, preds)
    rng = np.random.default_rng(0)
    ysh = rng.permutation(y)
    perm = r2_from_predictions(ysh, loso_cv(X, ysh, ESTIMATORS[best["estimator"]]))
    print(f"\n[ceiling] winner {best['agg']}/{best['transform']}/{best['estimator']}: "
          f"R2={best['r2']:+.4f} CI=[{lo:+.3f},{hi:+.3f}] perm={perm:+.4f}")

    out = P["sweep"].parent / f"tuning_ceiling_k{K}.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "WARNING": "EXPLORATORY. Every variant was scored on the same LOSO-CV R2 "
                    "used to rank them. The maximum is an upper bound produced by "
                    "outcome selection, NOT a reportable result. Nested CV would "
                    "land lower.",
        "selected_by_outcome": True,
        "corpus": args.corpus, "k": K, "target": args.target,
        "baseline_mean_clr_ridge": baseline,
        "n_variants": len(df),
        "winner": {**best.to_dict(), "ci_95": [float(lo), float(hi)],
                    "permutation_r2": float(perm)},
        "all_variants": rows,
    }, indent=2, default=str))
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"[ceiling] -> {out}")


if __name__ == "__main__":
    main()
