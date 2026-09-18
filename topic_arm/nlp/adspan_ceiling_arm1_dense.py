"""Arm 1 re-run on grids dense enough that the estimator comparison is fair.

WHY THIS EXISTS
---------------
Arm 1 asks "is ridge leaving signal on the table, or is the representation the
bottleneck?" That question is only answerable if every estimator was tuned with
comparable care. It was not. Two follow-up audits showed the published Arm 1
table is partly a GRID artifact rather than an estimator ranking:

  * nlp/enet_grid_probe    -- elasticnet's optimum (alpha ~ 0.0032) falls in the
                              GAP between Arm 1's adjacent grid points 0.001 and
                              0.01. Its published +0.357 rises to +0.478 on a
                              25-point grid. That is ~0.12 R2 of pure tuning
                              resolution, and it is the difference between
                              "elasticnet is much worse than ridge" and
                              "elasticnet is level with ridge".
  * nlp/arm1_grid_audit    -- ridge selects Arm 1's LOWEST alpha in 204/204
                              folds, i.e. the optimum may sit outside the
                              searched range entirely. It does: the dense
                              optimum is 0.562, below the grid floor of 1.0.
                              Ridge is nearly insensitive to this (+0.460 ->
                              +0.465) but it was searched on the wrong range.

So the published table compares a well-tuned ridge against an under-tuned
elasticnet and an unaudited MLP/RF/GBM. This module re-runs all of them on grids
chosen so that no estimator is handicapped by grid resolution, and records the
selected hyperparameter in every fold so the claim is checkable rather than
asserted.

WHAT "DENSE ENOUGH" MEANS HERE (the fairness rule, applied uniformly)
--------------------------------------------------------------------
For every estimator with a continuous penalty:
  1. log-spaced grid at <= 1/4 decade resolution (1/2 decade for the MLP, whose
     fits cost ~1000x a ridge fit), and
  2. a range wide enough that the modal selection is INTERIOR -- verified after
     the fact and flagged in the output if violated.
Structural grids (RF depth/features, GBM rate/iters/leaves) have no continuous
penalty to resolve, so they are WIDENED rather than densified and audited for
edge selection.

SCALER NOTE -- why this is not a second free parameter for ridge
----------------------------------------------------------------
On a single feature block, BlockNormalizer is exactly StandardScaler followed by
dividing every column by sqrt(p). For ridge that is a pure reparameterization:
scaling X by c rescales the optimal alpha by c^2 and leaves the fitted function
unchanged. With p=75 the two should differ by a factor of ~75, and they do --
the audit's BlockNormalizer optimum 0.562 vs. its StandardScaler optima 31.6/56.2
(geometric mean 42.2 ~ 75 x 0.562). So ridge's scaler choice is NOT a modelling
degree of freedom, only grid alignment, and BlockNormalizer is kept for
consistency with Arm 2.

Elastic net is different: the L1 term scales as 1/c and the L2 term as 1/c^2, so
they cannot co-scale and the scaler genuinely changes the model. Both are
therefore run, and BOTH are reported -- `elasticnet` (BlockNormalizer, matching
Arm 1/2 convention) is the headline row and `elasticnet_stdscaler` is a
robustness row. Neither is selected on its outer R2.

HONESTY CONTROLS -- unchanged from nlp/adspan_ceiling_nested
------------------------------------------------------------
Every hyperparameter is still chosen by GridSearchCV strictly inside the outer
training split; only outer-loop R2 is reported; per-show squared errors are kept
so estimator comparisons use a PAIRED test on identical folds. Densifying a grid
does not relax any of this -- the inner search still never sees the held-out
show. The reported quantity remains a RANGE with the ridge baseline highlighted;
the max is never the headline, because picking the best estimator by the same
outer R2 used to report it is the outcome-selection this whole harness exists to
remove. That is exactly why the elasticnet correction below does NOT promote
elasticnet to the baseline carried into Arm 2.

PARALLELISM -- outer loop, not inner
-------------------------------------
adspan_ceiling_nested pins inner GridSearchCV to n_jobs=1 for a measured reason:
the inner fits are ~1ms, so joblib's per-search process-pool spawn costs more
than the work, and the outer loop pays it 204 times. The fix is to parallelize
the OUTER loop instead -- one pool, 204 tasks, each doing a whole serial fold.
Inner searches stay n_jobs=1 here, and RandomForest drops to n_jobs=1 too since
its trees would otherwise contend with the outer workers.

    .venv/bin/python -u -m nlp.adspan_ceiling_arm1_dense --estimators ridge,elasticnet
    .venv/bin/python -u -m nlp.adspan_ceiling_arm1_dense --estimators rf,gbm,mlp
    .venv/bin/python -u -m nlp.adspan_ceiling_arm1_dense --merge
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .adspan_ceiling_nested import (
    ADSPAN_DIR,
    BlockNormalizer,
    INNER_CV_FOLDS,
    K_PRIMARY,
    SEED,
    align,
    build_topics,
    paired_test,
)
from .adspan_target_robustness import build_targets
from .embed_ideology_nonlinear import RIDGE_ALPHAS, bootstrap_r2_ci, r2_from_predictions

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGETS = ["extended_all", "host_only"]

# ------------------------------------------------------------------ grids ----
# Arm 1's published (coarse) grids, kept only to compute EDGE/GAP flags.
COARSE = {
    "ridge":               {"m__alpha": RIDGE_ALPHAS},
    "elasticnet":          {"m__alpha": [0.001, 0.01, 0.1, 1.0],
                            "m__l1_ratio": [0.1, 0.5, 0.9, 1.0]},
    "elasticnet_stdscaler": {"m__alpha": [0.001, 0.01, 0.1, 1.0],
                             "m__l1_ratio": [0.1, 0.5, 0.9, 1.0]},
    "mlp":                 {"m__alpha": [1e-3, 1e-1, 1.0],
                            "m__hidden_layer_sizes": [(32,), (64, 32)]},
    "random_forest":       {"max_depth": [None, 6], "max_features": ["sqrt", 0.3]},
    "grad_boost":          {"learning_rate": [0.03, 0.1], "max_iter": [200, 400],
                            "max_leaf_nodes": [7, 31]},
}

# 1/4-decade log grids for the continuous penalties; widened structural grids.
DENSE = {
    "ridge":               {"m__alpha": list(np.logspace(-3, 5, 33))},
    "elasticnet":          {"m__alpha": list(np.logspace(-5, 1, 25)),
                            "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
    "elasticnet_stdscaler": {"m__alpha": list(np.logspace(-5, 1, 25)),
                             "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
    # 1/2 decade: an MLP fit costs ~1000x a ridge fit, so 15 alphas x 2
    # architectures x 5 inner x 204 outer is already ~31k network fits.
    "mlp":                 {"m__alpha": list(np.logspace(-5, 2, 15)),
                            "m__hidden_layer_sizes": [(32,), (64, 32)]},
    "random_forest":       {"max_depth": [None, 6, 12],
                            "max_features": ["sqrt", 0.3, 0.6]},
    "grad_boost":          {"learning_rate": [0.01, 0.03, 0.1],
                            "max_iter": [200, 400], "max_leaf_nodes": [7, 31]},
}

# The parameter whose selection is audited for EDGE/GAP (the continuous one
# where there is one; the most structural otherwise).
AUDIT_KEY = {
    "ridge": "m__alpha", "elasticnet": "m__alpha",
    "elasticnet_stdscaler": "m__alpha", "mlp": "m__alpha",
    "random_forest": "max_depth", "grad_boost": "learning_rate",
}

ALIASES = {"rf": "random_forest", "gbm": "grad_boost",
           "enet": "elasticnet", "enet_std": "elasticnet_stdscaler"}


def make_pipe(name: str, p: int):
    """Estimator pipeline. Scalers match the Arm 1/2 convention (BlockNormalizer)
    except for the explicit elasticnet StandardScaler robustness row."""
    if name == "ridge":
        return Pipeline([("blk", BlockNormalizer((p,))), ("m", Ridge())])
    if name == "elasticnet":
        return Pipeline([("blk", BlockNormalizer((p,))),
                         ("m", ElasticNet(max_iter=20000))])
    if name == "elasticnet_stdscaler":
        return Pipeline([("s", StandardScaler()),
                         ("m", ElasticNet(max_iter=20000))])
    if name == "mlp":
        return Pipeline([("blk", BlockNormalizer((p,))),
                         ("m", MLPRegressor(random_state=SEED, max_iter=2000,
                                            early_stopping=True,
                                            n_iter_no_change=20))])
    if name == "random_forest":
        # n_jobs=1: the OUTER loop owns the workers here, so per-fit tree
        # parallelism would oversubscribe rather than help.
        return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1)
    if name == "grad_boost":
        return HistGradientBoostingRegressor(random_state=SEED)
    raise ValueError(f"unknown estimator {name!r}")


# ------------------------------------------------------------- outer loop ----
def _fold(X, y, i, name, grid):
    tr = np.ones(len(y), dtype=bool)
    tr[i] = False
    g = GridSearchCV(make_pipe(name, X.shape[1]), grid,
                     cv=INNER_CV_FOLDS, scoring="r2", n_jobs=1)
    g.fit(X[tr], y[tr])
    return float(g.predict(X[i:i + 1])[0]), g.best_params_


def loso_dense(X, y, name, grid, workers):
    out = Parallel(n_jobs=workers, backend="loky")(
        delayed(_fold)(X, y, i, name, grid) for i in range(len(y)))
    return np.array([p for p, _ in out]), [b for _, b in out]


def audit_flags(name, picks):
    """EDGE (selections pile on a grid end) / GAP (dense optimum unreachable on
    the coarse grid). These are the two ways a coarse grid silently mis-tunes."""
    key = AUDIT_KEY[name]
    vals = [b[key] for b in picks]
    numeric = [v for v in vals if isinstance(v, (int, float))]
    flags = []
    coarse = COARSE[name].get(key)
    dense = DENSE[name].get(key)
    if numeric and dense:
        dnum = [v for v in dense if isinstance(v, (int, float))]
        at_lo = sum(1 for v in numeric if v <= min(dnum) + 1e-12)
        at_hi = sum(1 for v in numeric if v >= max(dnum) - 1e-12)
        if at_lo:
            flags.append(f"DENSE-EDGE-low {at_lo}/{len(numeric)} -- widen range")
        if at_hi:
            flags.append(f"DENSE-EDGE-high {at_hi}/{len(numeric)} -- widen range")
    modal = Counter(str(v) for v in vals).most_common(1)[0][0]
    if numeric and coarse:
        cnum = [v for v in coarse if isinstance(v, (int, float))]
        m = Counter(numeric).most_common(1)[0][0]
        if m < min(cnum):
            flags.append(f"coarse grid MISSED: modal {m:.6g} below its floor {min(cnum)}")
        elif m > max(cnum):
            flags.append(f"coarse grid MISSED: modal {m:.6g} above its ceiling {max(cnum)}")
        elif m not in cnum:
            below = [v for v in cnum if v < m][-1:] or [None]
            above = [v for v in cnum if v > m][:1] or [None]
            flags.append(f"coarse grid GAP: modal {m:.6g} lies between "
                         f"{below[0]} and {above[0]}")
    return modal, flags


def run_estimator(name, workers):
    grid = DENSE[name]
    ncomb = int(np.prod([len(v) for v in grid.values()]))
    res = {}
    for tname in TARGETS:
        y_series = build_targets()[tname]
        idx, (X,), y = align([build_topics(K_PRIMARY)], y_series)
        print(f"[dense] {name} / {tname}: n={len(y)} dim={X.shape[1]} "
              f"grid={ncomb} combos x {INNER_CV_FOLDS} inner x {len(y)} outer",
              flush=True)
        preds, picks = loso_dense(X, y, name, grid, workers)
        r2 = r2_from_predictions(y, preds)
        lo, hi = bootstrap_r2_ci(y, preds)
        modal, flags = audit_flags(name, picks)
        counts = Counter(json.dumps({k: str(v) for k, v in b.items()},
                                    sort_keys=True) for b in picks)
        print(f"    outer R2 = {r2:+.4f} [{lo:+.4f}, {hi:+.4f}]  modal "
              f"{AUDIT_KEY[name]}={modal}", flush=True)
        for f in flags:
            print(f"    FLAG: {f}", flush=True)
        res[tname] = {
            "label": f"{tname}/{name}", "n": int(len(y)),
            "outer_r2": float(r2), "ci_95": [float(lo), float(hi)],
            "sq_err": ((y - preds) ** 2).tolist(), "preds": preds.tolist(),
            "show_ids": list(idx),
            "grid": {k: [str(v) for v in vals] for k, vals in grid.items()},
            "n_grid_combos": ncomb,
            "modal_param": modal, "flags": flags,
            "selection_counts": [[json.loads(k), c]
                                 for k, c in counts.most_common(5)],
        }
    p = ADSPAN_DIR / f"arm1_dense_{name}.json"
    p.write_text(json.dumps({"estimator": name, "k": K_PRIMARY,
                             "results": res,
                             "generated_at": datetime.now(timezone.utc).isoformat()},
                            indent=2))
    print(f"[dense] -> {p}", flush=True)


# ----------------------------------------------------------------- merge -----
def merge():
    """Assemble the per-estimator files into an Arm-1-shaped JSON.

    Same schema as ceiling_nested_arm1_k75.json so the report layer and any
    paired test can consume it unchanged. `elasticnet_stdscaler` is carried as a
    robustness row and kept OUT of the range/median so the headline summary
    compares one row per estimator, not two for elasticnet."""
    order = ["ridge", "elasticnet", "random_forest", "grad_boost", "mlp",
             "elasticnet_stdscaler"]
    loaded = {}
    for n in order:
        p = ADSPAN_DIR / f"arm1_dense_{n}.json"
        if p.exists():
            loaded[n] = json.loads(p.read_text())["results"]
    if "ridge" not in loaded:
        raise SystemExit("[merge] ridge is the baseline and is missing -- run it first")

    coarse = json.loads((ADSPAN_DIR / "ceiling_nested_arm1_k75.json").read_text())
    out = {"k": K_PRIMARY, "arm": 1, "grids": "dense",
           "question": "recoverability, not dominance",
           "supersedes": "ceiling_nested_arm1_k75.json",
           "robustness_rows": ["elasticnet_stdscaler"],
           "results": {}}
    for t in TARGETS:
        models = {n: loaded[n][t] for n in order
                  if n in loaded and t in loaded[n]}
        if not models:
            continue
        headline = {n: m for n, m in models.items()
                    if n not in out["robustness_rows"]}
        r2s = [m["outer_r2"] for m in headline.values()]
        prev = coarse["results"].get(t, {}).get("models", {})
        out["results"][t] = {
            "show_ids": models["ridge"]["show_ids"],
            "models": models,
            "range": [float(min(r2s)), float(max(r2s))],
            "median": float(np.median(r2s)),
            "ridge_baseline": models["ridge"]["outer_r2"],
            "vs_ridge_paired": {n: paired_test(models["ridge"], models[n])
                                for n in models if n != "ridge"},
            "delta_vs_coarse": {n: float(m["outer_r2"] - prev[n]["outer_r2"])
                                for n, m in models.items() if n in prev},
            "coarse_r2": {n: float(m["outer_r2"]) for n, m in prev.items()},
        }
    p = ADSPAN_DIR / f"ceiling_nested_arm1_k{K_PRIMARY}_dense.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[merge] {len(loaded)} estimators -> {p}")
    for t, r in out["results"].items():
        print(f"\n{t}: range [{r['range'][0]:+.3f}, {r['range'][1]:+.3f}], "
              f"median {r['median']:+.3f}, ridge baseline {r['ridge_baseline']:+.3f}")
        for n, m in r["models"].items():
            d = r["delta_vs_coarse"].get(n)
            tag = " [robustness row]" if n in out["robustness_rows"] else ""
            print(f"  {n:<22} {m['outer_r2']:+.4f}"
                  + (f"  (coarse {r['coarse_r2'][n]:+.4f}, d {d:+.4f})" if d is not None else "")
                  + tag)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--estimators", default="",
                    help="comma list: ridge,elasticnet,elasticnet_stdscaler,rf,gbm,mlp")
    ap.add_argument("--workers", type=int, default=4,
                    help="OUTER-loop workers; inner GridSearchCV stays n_jobs=1")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if the estimator's JSON exists")
    args = ap.parse_args()

    for raw in [s.strip() for s in args.estimators.split(",") if s.strip()]:
        name = ALIASES.get(raw, raw)
        if name not in DENSE:
            raise SystemExit(f"unknown estimator {raw!r}")
        p = ADSPAN_DIR / f"arm1_dense_{name}.json"
        if p.exists() and not args.force:
            print(f"[skip] {name} -- {p.name} exists")
            continue
        run_estimator(name, args.workers)

    if args.merge:
        merge()


if __name__ == "__main__":
    main()
