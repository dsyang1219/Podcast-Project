"""Diagnostic bundle: WHY does ideology sit low in the topic PCA?

The unsupervised CLR PCA puts ideology on PC5-6 at K=75 with PC1 ~ 0. The
advisor expected PC1/PC2. This varies the four choices that could be
responsible, ONE AT A TIME from a common default, and reports every cell.

THE BRIGHT LINE
---------------
This is not a search for the configuration that puts ideology on PC1. If some
cell does put it there, that is a finding about THAT CHOICE and is reported with
its conditions -- it does not become the headline. The deliverable is "ideology's
position depends on X, and here is how".

Default cell: CLR + unsupervised PCA + no register control + extended_all.

Driver A -- transform:      CLR (default) | ILR | z-score (non-compositional)
Driver B -- method:         unsupervised PCA (default) | supervised PLS-1 | supervised ridge
Driver C -- register:       none (default) | drop core | drop broad | residualize
Driver D -- target:         extended_all (default) | host_only | guest_only

DOMINANCE vs RECOVERABILITY -- keep these separate
--------------------------------------------------
Unsupervised PCA answers DOMINANCE: is ideology a high-variance axis of topical
variation? Supervised projection answers RECOVERABILITY: does a direction that
predicts ideology exist at all? A supervised direction is NOT "PC1" and is never
reported as one. The interesting quantity is how much variance the supervised
ideology-direction carries, and where it WOULD rank if inserted among the PCs --
if that rank is low, then ideology is recoverable but not dominant, which is
precisely the answer to the advisor's expectation.

ILR is a mathematical control, not an independent finding
---------------------------------------------------------
ILR coordinates are CLR composed with an orthonormal basis of the sum-zero
hyperplane -- an isometry. PCA is invariant under orthogonal transformation of
the input, so ILR MUST reproduce CLR's eigenvalues and ideology correlations
exactly. The cell is included because agreeing to numerical precision verifies
the implementation; a disagreement would mean a bug. It is not evidence that the
result is robust to the choice of compositional basis, because there is no real
choice being made.

CLR remains the correct default. If the non-compositional z-score cell places
ideology higher, that is informative about the simplex geometry -- it does NOT
license abandoning CLR, because raw-proportion PCA reads simplex-induced
negative correlation as real structure.

    .venv/bin/python -m nlp.adspan_pca_diagnostics [--b 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV

from .adspan_phase_c import OUT_DIR, paths_for
from .adspan_target_robustness import build_targets
from .adspan_topic_pca import (
    K_PRIMARY,
    K_ROBUST,
    N_TOP_LOADINGS,
    ZERO_GUARD,
    fisher_ci,
    load_topic_meta,
)
from .lda_ideology_clr import clr

# Register topics. The CORE set is rule-selected and reproducible: a topic
# qualifies if >=2 of its top-10 words are spoken-stopwords/profanity, or any is
# profanity. That rule picks exactly one topic per K (T66 at K=75, T9 at K=30) --
# the same ones that anchor leading PCs by inspection. Note the sweep's own
# `is_contaminant` flag misses both: its threshold is n_filler>=5.
#
# The BROAD set adds hedging / discourse-marker / news-format topics identified
# by READING the top words. PROVENANCE: agent-proposed, NOT user-confirmed --
# unlike nlp/adspan_register_dropout.py's sets, which are the user's. Treat the
# broad set as a sensitivity check, not an authoritative register definition.
REGISTER_SETS = {
    75: {"core": [66], "broad": [66, 32]},
    30: {"core": [9], "broad": [9, 12, 25]},
}
REGISTER_WHY = {
    75: {66: "profanity/banter (rule-selected)",
         32: "hedging + discourse markers: bit guess blah i've thought feel"},
    30: {9: "profanity/banter (rule-selected)",
         12: "hedging/abstract-discussion markers: different sense bit ways we've",
         25: "news-bulletin format: news week quote today morning day days"},
}

DEFAULT_CELL = {"transform": "clr", "method": "unsupervised",
                "register": "none", "target": "extended_all"}


# ------------------------------------------------------------ transforms ----
def ilr_basis(d: int) -> np.ndarray:
    """Orthonormal (d x d-1) basis of the sum-zero hyperplane."""
    h = np.eye(d) - np.ones((d, d)) / d
    u, s, _ = np.linalg.svd(h)
    return u[:, : d - 1]


def apply_transform(raw: np.ndarray, transform: str) -> np.ndarray:
    pos = np.where(raw <= 0, ZERO_GUARD, raw)
    if transform == "clr":
        return clr(pos)
    if transform == "ilr":
        return clr(pos) @ ilr_basis(pos.shape[1])
    if transform == "zscore":
        # Deliberately NON-compositional: z-score the raw proportions.
        sd = raw.std(axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
        return (raw - raw.mean(axis=0)) / sd
    raise ValueError(transform)


def is_compositional(transform: str) -> bool:
    return transform in ("clr", "ilr")


# -------------------------------------------------------- register control ----
def apply_register(raw: np.ndarray, k: int, control: str) -> tuple[np.ndarray, list[int], str]:
    """Return (matrix, kept topic indices, stage) where stage says whether the
    control acts on raw proportions (before transform) or on the transformed
    matrix (handled by the caller)."""
    all_idx = list(range(raw.shape[1]))
    if control == "none":
        return raw, all_idx, "raw"
    if control in ("drop_core", "drop_broad"):
        drop = set(REGISTER_SETS[k]["core" if control == "drop_core" else "broad"])
        keep = [i for i in all_idx if i not in drop]
        sub = raw[:, keep]
        # Re-close the composition so the kept topics still sum to 1 -- without
        # this the CLR geometric mean is taken over a sub-composition that no
        # longer spans the simplex, which is the standard sub-composition step.
        sub = sub / sub.sum(axis=1, keepdims=True)
        return sub, keep, "raw"
    if control == "residualize":
        return raw, all_idx, "transformed"
    raise ValueError(control)


def residualize_on_register(X: np.ndarray, k: int) -> tuple[np.ndarray, list[int]]:
    """Project the register topics' subspace out of the remaining columns.

    Keeps every show but removes the linear component of each content topic that
    register explains -- a weaker intervention than dropping, and it answers a
    slightly different question: not 'what if these topics did not exist' but
    'what is left once register variance is partialled out'.
    """
    reg = REGISTER_SETS[k]["core"]
    keep = [i for i in range(X.shape[1]) if i not in set(reg)]
    R = X[:, reg]
    Xo = X[:, keep]
    Rc = R - R.mean(axis=0, keepdims=True)
    Xoc = Xo - Xo.mean(axis=0, keepdims=True)
    resid = Xoc - Rc @ np.linalg.pinv(Rc) @ Xoc
    return resid, keep


def build_matrix(raw: np.ndarray, k: int, transform: str, control: str) -> tuple[np.ndarray, list[int]]:
    sub, keep, stage = apply_register(raw, k, control)
    X = apply_transform(sub, transform)
    if stage == "transformed":
        X, keep = residualize_on_register(X, k)
    return X, keep


def numerical_rank(X: np.ndarray) -> int:
    """Components with real variance. Computed rather than derived from a
    formula because the cap differs per cell: CLR loses one dimension to the
    zero-sum constraint, ILR has already absorbed it, z-score loses none, and
    residualizing on register drops a further dimension. Getting this wrong
    would pad the scree with structurally-zero components and corrupt Horn's.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    tol = max(Xc.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    return int(min((s > tol).sum(), Xc.shape[0] - 1))


# ---------------------------------------------------------------- PCA ----
def eig_and_scores(X: np.ndarray, n_comp: int):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    eig = (s ** 2) / (Xc.shape[0] - 1)
    return eig[:n_comp], U[:, :n_comp] * s[:n_comp], Vt[:n_comp].T


def horns(raw: np.ndarray, k: int, transform: str, control: str,
          n_comp: int, b: int, seed: int) -> int:
    """Permute raw proportion columns, then run the identical pipeline."""
    rng = np.random.default_rng(seed)
    X, _ = build_matrix(raw, k, transform, control)
    observed, _, _ = eig_and_scores(X, n_comp)

    null = np.empty((b, n_comp))
    perm = np.empty_like(raw)
    for i in range(b):
        for j in range(raw.shape[1]):
            perm[:, j] = rng.permutation(raw[:, j])
        Xp, _ = build_matrix(perm, k, transform, control)
        e, _, _ = eig_and_scores(Xp, n_comp)
        null[i] = e

    null95 = np.percentile(null, 95, axis=0)
    retained = 0
    for i in range(n_comp):
        if observed[i] > null95[i]:
            retained += 1
        else:
            break
    return max(retained, 1)


# --------------------------------------------------------- supervised ----
def loso_cv_corr(X: np.ndarray, y: np.ndarray, fit_direction) -> float:
    """Leave-one-show-out correlation for a supervised direction -- the honest
    counterpart to the in-sample correlation, which is optimistic because the
    direction was chosen to maximise it."""
    preds = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), dtype=bool)
        m[i] = False
        w, mu = fit_direction(X[m], y[m])
        preds[i] = float((X[i] - mu) @ w)
    return float(np.corrcoef(preds, y)[0, 1])


def pls1_direction(X: np.ndarray, y: np.ndarray):
    mu = X.mean(axis=0)
    pls = PLSRegression(n_components=1, scale=False)
    pls.fit(X - mu, y - y.mean())
    w = pls.x_weights_[:, 0]
    return w / np.linalg.norm(w), mu


def ridge_direction(X: np.ndarray, y: np.ndarray):
    mu = X.mean(axis=0)
    r = RidgeCV(alphas=np.logspace(-3, 5, 40))
    r.fit(X - mu, y - y.mean())
    w = r.coef_.ravel()
    n = np.linalg.norm(w)
    return (w / n if n > 0 else w), mu


# -------------------------------------------------------------- cells ----
def register_anchored(loading: np.ndarray, kept: list[int], k: int,
                      transform: str) -> bool | None:
    """Does this direction's top-|loading| set include a register topic?

    Returns None for ILR: its coordinates are an orthonormal ROTATION of the
    CLR space, so a loading index is a basis vector, not a topic, and mapping it
    back to a topic id would be meaningless. The CLR row answers this question
    for both, since ILR is the same PCA.
    """
    if transform == "ilr":
        return None
    reg = set(REGISTER_SETS[k]["core"])
    top = np.argsort(np.abs(loading))[::-1][:N_TOP_LOADINGS]
    return any(kept[i] in reg for i in top)


def run_cell(raw: np.ndarray, show_index: pd.Index, k: int, y_series: pd.Series,
             transform: str, method: str, control: str, target: str,
             b: int, seed: int) -> dict:
    common = show_index.intersection(y_series.index)
    pos = show_index.get_indexer(common)
    sub_raw = raw[pos]
    y = y_series.loc[common].to_numpy(dtype=float)

    X, kept = build_matrix(sub_raw, k, transform, control)
    n_comp = numerical_rank(X)
    eig, scores, loadings = eig_and_scores(X, n_comp)
    total_var = float(eig.sum())

    base = {"k": k, "transform": transform, "method": method,
            "register_control": control, "target": target, "n": int(len(y)),
            "n_topics_used": len(kept), "n_components": n_comp}

    if method == "unsupervised":
        retained = horns(sub_raw, k, transform, control, n_comp, b, seed)
        rows = []
        for pc in range(retained):
            x = scores[:, pc]
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
            rows.append({"pc": pc + 1, "r": r, "abs_r": abs(r),
                         "ci_95": fisher_ci(r, len(y)),
                         "pct_var": float(eig[pc] / total_var * 100)})
        best = max(rows, key=lambda d: d["abs_r"])
        return {**base, "retained_dimensions": retained,
                "ideology_rank": best["pc"], "rank_label": f"PC{best['pc']}",
                "r": best["r"], "abs_r": best["abs_r"], "ci_95": best["ci_95"],
                "pct_var_of_that_axis": best["pct_var"],
                "register_anchored": (
                    register_anchored(loadings[:, best["pc"] - 1], kept, k, transform)
                    if control in ("none", "residualize") else False),
                "register_topics_present": control in ("none", "residualize"),
                "on_pc1_or_pc2": best["pc"] <= 2,
                "pc1_abs_r": rows[0]["abs_r"],
                "per_pc": rows}

    fit = pls1_direction if method == "supervised_pls1" else ridge_direction
    w, mu = fit(X, y)
    t = (X - mu) @ w
    var_dir = float(t.var(ddof=1))
    # Where the supervised direction WOULD rank if inserted among the PCs.
    insert_rank = int((eig > var_dir).sum() + 1)
    r_in = float(np.corrcoef(t, y)[0, 1])
    r_cv = loso_cv_corr(X, y, fit)
    return {**base,
            "retained_dimensions": None,
            "ideology_rank": insert_rank,
            "rank_label": f"supervised (would insert at #{insert_rank})",
            "r": r_in, "abs_r": abs(r_in), "ci_95": fisher_ci(r_in, len(y)),
            "r_in_sample": r_in, "r_loso_cv": r_cv,
            "r2_loso_cv": r_cv ** 2 if np.isfinite(r_cv) else float("nan"),
            "pct_var_of_that_axis": float(var_dir / total_var * 100),
            "register_anchored": register_anchored(w, kept, k, transform),
            "register_topics_present": control in ("none", "residualize"),
            "on_pc1_or_pc2": False,
            "pc1_abs_r": None,
            "note": "SUPERVISED -- answers recoverability, NOT dominance; "
                    "this is not a PC and must not be reported as PC1"}


def grid_cells() -> list[dict]:
    """One factor at a time from DEFAULT_CELL."""
    cells = [dict(DEFAULT_CELL, driver="default")]
    for t in ("ilr", "zscore"):
        cells.append(dict(DEFAULT_CELL, transform=t, driver="A_transform"))
    for m in ("supervised_pls1", "supervised_ridge"):
        cells.append(dict(DEFAULT_CELL, method=m, driver="B_supervised"))
    for c in ("drop_core", "drop_broad", "residualize"):
        cells.append(dict(DEFAULT_CELL, register=c, driver="C_register"))
    for tg in ("host_only", "guest_only"):
        cells.append(dict(DEFAULT_CELL, target=tg, driver="D_target"))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    targets = build_targets()
    P = paths_for("clean")
    results = []

    for k in (K_PRIMARY, K_ROBUST):
        dt = pd.read_csv(P["doctopic_fmt"].format(k=k), dtype={"collection_id": str})
        cols = [f"T{i}" for i in range(k)]
        show = dt.groupby("collection_id")[cols].mean()
        raw = show.to_numpy(dtype=float)
        print(f"\n{'=' * 78}\n[grid] K={k}  ({raw.shape[0]} shows x {raw.shape[1]} topics)\n{'=' * 78}")
        for cell in grid_cells():
            res = run_cell(raw, show.index, k, targets[cell["target"]],
                           cell["transform"], cell["method"], cell["register"],
                           cell["target"], args.b, args.seed)
            res["driver"] = cell["driver"]
            results.append(res)
            print(f"  [{cell['driver']:<13}] {cell['transform']:<7} "
                  f"{cell['method']:<17} {cell['register']:<11} "
                  f"{cell['target']:<13} -> {res['rank_label']:<34} "
                  f"|r|={res['abs_r']:.3f}  var={res['pct_var_of_that_axis']:.1f}%"
                  f"  reg={'Y' if res['register_anchored'] else 'n'}")

    out = OUT_DIR / "pca_diagnostics.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "default_cell": DEFAULT_CELL,
        "register_sets": {str(k): v for k, v in REGISTER_SETS.items()},
        "register_provenance": (
            "core = rule-selected (>=2 spoken-stopword/profanity words in top-10, "
            "or any profanity); broad = core + hand-read hedging/format topics, "
            "AGENT-PROPOSED and not user-confirmed"),
        "register_why": {str(k): {str(t): w for t, w in v.items()}
                          for k, v in REGISTER_WHY.items()},
        "guardrails": {
            "supervised": "answers recoverability, not dominance; never a PC1 claim",
            "ilr": "isometry of CLR -- must match CLR exactly; a control, not a finding",
            "zscore": "non-compositional; CLR remains the correct default",
            "reporting": "every cell reported; ideology-on-PC1 cells not headlined",
        },
        "cells": results,
    }, indent=2, default=str))

    df = pd.DataFrame([{
        "k": r["k"], "driver": r["driver"], "transform": r["transform"],
        "method": r["method"], "register": r["register_control"],
        "target": r["target"], "n": r["n"], "rank": r["ideology_rank"],
        "rank_label": r["rank_label"], "abs_r": round(r["abs_r"], 3),
        "ci_low": round(r["ci_95"][0], 3), "ci_high": round(r["ci_95"][1], 3),
        "pct_var": round(r["pct_var_of_that_axis"], 2),
        "register_anchored": r["register_anchored"],
        "on_pc1_or_pc2": r["on_pc1_or_pc2"],
        "r_loso_cv": round(r["r_loso_cv"], 3) if r.get("r_loso_cv") is not None else "",
    } for r in results])
    csv = OUT_DIR / "pca_diagnostics_table.csv"
    df.to_csv(csv, index=False)
    print(f"\n[out] {out}\n[out] {csv}")


if __name__ == "__main__":
    main()
