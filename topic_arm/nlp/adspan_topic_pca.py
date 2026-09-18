"""Dimensionality of the TOPIC representation: CLR -> PCA -> Horn's parallel
analysis, and where ideology lives among the resulting components.

Why K=75 is primary
-------------------
K=75 is used as the PRIMARY K because it is the c_v coherence PEAK, which is an
OUTCOME-INDEPENDENT criterion -- coherence is computed from the corpus alone and
never sees the ideology target. The pre-registered smallest-K-within-1SE rule
would instead select K=30 (c_v=0.6720 clears the 0.6601 threshold set by the
K=75 peak minus one SE). Both are defensible readings of the same coherence
curve, so every result here is reported at BOTH K=75 and K=30 and the K=30
answer is treated as a robustness bound on the K=75 claim, not as a footnote.

K=75 is deliberately NOT justified by its ideology R2 (0.4520, the best in the
grid). Selecting K on the outcome and then reporting that outcome at the
selected K is circular; the R2 ranking plays no part in this choice and is not
cited as support for it.

Why CLR before PCA (the crux)
-----------------------------
A show's topic vector is COMPOSITIONAL: 75 proportions constrained to sum to 1.
That constraint makes the covariance structure of the raw proportions an
artifact of the simplex rather than a property of the data -- components are
forced to be negatively correlated (one topic's share can only grow if others
shrink), so ordinary PCA on raw proportions reads that induced negative
correlation as real structure and distorts the principal axes. The centered
log-ratio transform (Aitchison 1982) maps the simplex to unconstrained real
space where ordinary covariance is meaningful. CLR is therefore REQUIRED here,
not a stylistic preference, and is applied before PCA everywhere below.

CLR maps each row to a zero-sum vector, so the CLR matrix has rank at most
K-1. The component count tested is capped at min(n_shows-1, K-1) accordingly;
anything past that rank is structurally zero and not a finding.

Horn's parallel analysis
------------------------
Dimensionality is decided by Horn's parallel analysis, NOT by eyeballing the
scree plot and NOT by Kaiser's eigenvalue>1 rule (both are known to overretain).
Each of the K columns of the show-level proportion matrix is independently
permuted across shows, which destroys between-topic correlation while preserving
every topic's marginal distribution; the permuted matrix is then CLR-transformed
and decomposed exactly as the observed one. A component is retained if its
observed eigenvalue exceeds the 95th percentile of the null eigenvalues at that
same rank, counting contiguously from the top -- the standard stopping rule, and
the same rule and percentile nlp/embed_rank.py used for the embedding arm, so
the topic-space dimension count is comparable to the embedding arm's.

Permute-then-CLR (rather than permuting the CLR matrix) mirrors embed_rank.py's
permute-then-transform ordering and keeps the null inside the same positive
orthant the observed data occupies.

Scope of the claim
------------------
This measures the dimensionality of THIS topic representation -- one downstream
of K, of the moderate-regime vocabulary pruning, and of ad-span excision. It is
not an estimate of "the true dimensionality of political discourse". Changing K
changes the representation and can change the number; that is precisely why the
K=30 replication is run and reported alongside.

Usage
    .venv/bin/python -m nlp.adspan_topic_pca [--b 1000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .adspan_phase_c import OUT_DIR, paths_for
from .adspan_target_robustness import build_targets
from .lda_ideology_clr import clr

K_PRIMARY = 75
K_ROBUST = 30
K_LIST = [K_PRIMARY, K_ROBUST]

PRIMARY_K_RATIONALE = (
    "K=75 is primary because it is the c_v coherence PEAK (outcome-independent). "
    "The smallest-K-within-1SE rule would instead select K=30; results are "
    "reported at both. K=75 is NOT justified by its ideology R2 -- that would be "
    "circular."
)

N_TOP_PCS_INTERPRETED = 7   # covers every PC Horn's retains at K=75
N_TOP_LOADINGS = 8          # topics listed per pole of each interpreted PC
ZERO_GUARD = 1e-12          # same guard adspan_phase_c.probe uses before CLR


# ------------------------------------------------------------ step 1 ----
def show_topic_matrix(k: int) -> tuple[pd.DataFrame, dict]:
    """204 shows x K topics, each row a show's MEAN topic proportion across its
    passages on the ad-span-cleaned corpus."""
    P = paths_for("clean")
    dt = pd.read_csv(P["doctopic_fmt"].format(k=k), dtype={"collection_id": str})
    cols = [f"T{i}" for i in range(k)]
    show = dt.groupby("collection_id")[cols].mean()

    raw = show.to_numpy(dtype=float)
    diagnostics = {
        "n_shows": int(raw.shape[0]),
        "n_topics": int(raw.shape[1]),
        "n_passages": int(len(dt)),
        "row_sums_min": float(raw.sum(axis=1).min()),
        "row_sums_max": float(raw.sum(axis=1).max()),
        "n_all_zero_rows": int((raw <= 0).all(axis=1).sum()),
        "n_rows_with_any_zero": int((raw <= 0).any(axis=1).sum()),
        "min_proportion": float(raw.min()),
        "n_nonfinite_cells": int((~np.isfinite(raw)).sum()),
    }

    X = clr(np.where(raw <= 0, ZERO_GUARD, raw))
    diagnostics.update({
        "clr_n_nonfinite_cells": int((~np.isfinite(X)).sum()),
        "clr_max_abs_row_sum": float(np.abs(X.sum(axis=1)).max()),
        "clr_n_constant_rows": int((X.std(axis=1) == 0).sum()),
        "clr_n_constant_cols": int((X.std(axis=0) == 0).sum()),
    })
    return show, {"clr_matrix": X, **diagnostics}


# ------------------------------------------------------------ step 2 ----
def eigenvalues(X: np.ndarray, n_components: int) -> np.ndarray:
    """PCA eigenvalues of a column-centered matrix, via SVD."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    eig = (s ** 2) / (Xc.shape[0] - 1)
    return eig[:n_components]


def pca_scores(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, :n_components] * s[:n_components]
    loadings = Vt[:n_components].T                      # (K topics, n_components)
    eig = (s ** 2) / (Xc.shape[0] - 1)
    return scores, loadings, eig[:n_components]


def horns_parallel_analysis(raw: np.ndarray, n_components: int, b: int,
                            seed: int) -> dict:
    """Permute each topic column across shows, CLR, decompose. Retain
    contiguously while observed eigenvalue > 95th percentile of the null."""
    rng = np.random.default_rng(seed)
    observed = eigenvalues(clr(np.where(raw <= 0, ZERO_GUARD, raw)), n_components)

    null = np.empty((b, n_components), dtype=float)
    perm = np.empty_like(raw)
    for i in range(b):
        for j in range(raw.shape[1]):
            perm[:, j] = rng.permutation(raw[:, j])
        null[i] = eigenvalues(clr(np.where(perm <= 0, ZERO_GUARD, perm)), n_components)

    null_95 = np.percentile(null, 95, axis=0)
    retained = 0
    for i in range(n_components):
        if observed[i] > null_95[i]:
            retained += 1
        else:
            break

    total_var = float(observed.sum())
    return {
        "b_permutations": b,
        "n_components_tested": n_components,
        "retained_dimensions": int(retained),
        "observed_eigenvalues": observed.tolist(),
        "null_95th_percentile": null_95.tolist(),
        "pct_variance_per_pc": (observed / total_var * 100).tolist(),
        "cum_pct_variance": (np.cumsum(observed) / total_var * 100).tolist(),
        "pct_variance_retained": float(observed[:retained].sum() / total_var * 100),
    }


# ------------------------------------------------------------ step 3 ----
def fisher_ci(r: float, n: int, alpha: float = 0.05) -> list[float]:
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return [float("nan"), float("nan")]
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = 1.959963984540054
    return [float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))]


def ideology_by_pc(show_index: pd.Index, scores: np.ndarray, retained: int,
                   targets: dict[str, pd.Series]) -> dict:
    """Correlate every retained PC score with each ideology target.

    The PC space is built ONCE on all 204 shows; a target simply restricts which
    shows enter its correlation. Refitting PCA per target would give each target
    a different basis and make 'which PC carries ideology' incomparable across
    them, which is the entire question here.

    PC sign is arbitrary under SVD, so |r| carries the finding and the sign is
    reported only for reproducibility against this run's stored loadings.
    """
    out: dict[str, dict] = {}
    for name, y_series in targets.items():
        common = show_index.intersection(y_series.index)
        pos = show_index.get_indexer(common)
        y = y_series.loc[common].to_numpy(dtype=float)
        rows = []
        for pc in range(retained):
            x = scores[pos, pc]
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
            rows.append({
                "pc": pc + 1,
                "r": r,
                "abs_r": abs(r),
                "ci_95": fisher_ci(r, len(y)),
                "r2": r ** 2,
            })
        ranked = sorted(rows, key=lambda d: (-d["abs_r"] if np.isfinite(d["abs_r"]) else 0))
        out[name] = {
            "n": int(len(y)),
            "y_sd": float(y.std(ddof=1)),
            "per_pc": rows,
            "top_pc": ranked[0]["pc"] if ranked else None,
            "top_abs_r": ranked[0]["abs_r"] if ranked else None,
            "top3_pcs": [d["pc"] for d in ranked[:3]],
            "pc1_abs_r": rows[0]["abs_r"] if rows else None,
        }
    return out


# ------------------------------------------------------------ step 4 ----
def load_topic_meta(k: int) -> list[dict]:
    p = OUT_DIR / f"phase_c_ksweep_k{k}.json"
    return json.loads(p.read_text())["topics"]


def interpret_pcs(loadings: np.ndarray, topic_meta: list[dict], n_pcs: int) -> list[dict]:
    """Top positive/negative loading topics per leading PC, with the contaminant
    flags already assigned by the sweep's classify_topic."""
    out = []
    for pc in range(n_pcs):
        w = loadings[:, pc]
        order = np.argsort(w)
        neg_idx = [int(i) for i in order[:N_TOP_LOADINGS]]
        pos_idx = [int(i) for i in order[::-1][:N_TOP_LOADINGS]]

        def describe(idx_list):
            return [{
                "topic": int(i),
                "loading": float(w[i]),
                "top_words": topic_meta[i]["top_words"],
                "is_contaminant": bool(topic_meta[i]["is_contaminant"]),
                "flagged_as": topic_meta[i]["flagged_as"],
            } for i in idx_list]

        pos, neg = describe(pos_idx), describe(neg_idx)
        anchors = pos[:3] + neg[:3]
        out.append({
            "pc": pc + 1,
            "positive_pole": pos,
            "negative_pole": neg,
            "n_contaminant_in_top_loadings": sum(
                1 for t in pos + neg if t["is_contaminant"]),
            "contaminant_anchored": any(t["is_contaminant"] for t in anchors),
        })
    return out


# ---------------------------------------------- per-target Horn's ----
def horns_for_subsample(raw: np.ndarray, show_index: pd.Index, name: str,
                        y_series: pd.Series, b: int, seed: int,
                        topic_meta: list[dict]) -> dict:
    """Refit CLR -> PCA -> Horn's on ONLY the shows a given target covers.

    Differs from the shared-basis analysis in two ways that matter: PCA centering
    is computed over the subsample, and the permutation null is drawn from the
    subsample, so the retained-dimension count is specific to that target's
    shows rather than inherited from all 204. This answers "how many dimensions
    does the topic space have among the shows this target can speak about",
    which is the right question when comparing a guest-derived target against a
    host-derived one -- they cover different shows, so a shared basis silently
    assumes the dimensionality is the same for both.

    Cost: the bases are no longer comparable across targets. A "PC2" here is not
    the "PC2" of another target's fit, so only the RELATIVE position of ideology
    (is it PC1, or is it not) transfers between them.
    """
    common = show_index.intersection(y_series.index)
    pos = show_index.get_indexer(common)
    sub = raw[pos]
    y = y_series.loc[common].to_numpy(dtype=float)

    n_comp = int(min(sub.shape[0] - 1, sub.shape[1] - 1))
    horn = horns_parallel_analysis(sub, n_comp, b, seed)
    retained = max(horn["retained_dimensions"], 1)

    X = clr(np.where(sub <= 0, ZERO_GUARD, sub))
    scores, loadings, _ = pca_scores(X, n_comp)

    rows = []
    for pc in range(retained):
        x = scores[:, pc]
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
        rows.append({"pc": pc + 1, "r": r, "abs_r": abs(r),
                     "ci_95": fisher_ci(r, len(y)), "r2": r ** 2})
    ranked = sorted(rows, key=lambda d: -d["abs_r"] if np.isfinite(d["abs_r"]) else 0)

    return {
        "target": name,
        "n_shows": int(sub.shape[0]),
        "y_sd": float(y.std(ddof=1)),
        "basis": "refit on this target's shows only",
        "horns": horn,
        "per_pc": rows,
        "top_pc": ranked[0]["pc"],
        "top_abs_r": ranked[0]["abs_r"],
        "top_pc_ci": ranked[0]["ci_95"],
        "top_pc_r": ranked[0]["r"],
        "top3_pcs": [d["pc"] for d in ranked[:3]],
        "pc1_abs_r": rows[0]["abs_r"],
        "ideology_on_pc1": ranked[0]["pc"] == 1,
        "pc_interpretation": interpret_pcs(
            loadings, topic_meta, min(N_TOP_PCS_INTERPRETED, n_comp)),
    }


# --------------------------------------------------------------- run ----
def run_for_k(k: int, b: int, seed: int, targets: dict[str, pd.Series]) -> dict:
    print(f"\n{'=' * 70}\n[pca] K={k}  (ad-span-cleaned corpus)\n{'=' * 70}", flush=True)
    show, diag = show_topic_matrix(k)
    X = diag.pop("clr_matrix")
    raw = show.to_numpy(dtype=float)

    print(f"[step1] matrix {diag['n_shows']} shows x {diag['n_topics']} topics "
          f"from {diag['n_passages']} passages")
    print(f"[step1] row sums (pre-CLR) in [{diag['row_sums_min']:.6f}, "
          f"{diag['row_sums_max']:.6f}]; min proportion {diag['min_proportion']:.3e}")
    print(f"[step1] all-zero rows: {diag['n_all_zero_rows']} | rows with any zero "
          f"cell: {diag['n_rows_with_any_zero']} | non-finite cells: "
          f"{diag['n_nonfinite_cells']}")
    print(f"[step1] post-CLR: non-finite {diag['clr_n_nonfinite_cells']}, "
          f"constant rows {diag['clr_n_constant_rows']}, constant cols "
          f"{diag['clr_n_constant_cols']}, max|row sum| "
          f"{diag['clr_max_abs_row_sum']:.2e} (CLR rows are zero-sum by construction)")

    n_comp = int(min(raw.shape[0] - 1, k - 1))
    print(f"[step2] testing {n_comp} components (rank cap: min(n-1={raw.shape[0]-1}, "
          f"K-1={k-1})); Horn's B={b}")
    horn = horns_parallel_analysis(raw, n_comp, b, seed)
    retained = horn["retained_dimensions"]
    print(f"[step2] RETAINED {retained} dimensions "
          f"({horn['pct_variance_retained']:.1f}% of variance)")
    for i in range(min(8, n_comp)):
        mark = "<-retained" if i < retained else ""
        print(f"         PC{i+1:<3} eig={horn['observed_eigenvalues'][i]:8.4f}  "
              f"null95={horn['null_95th_percentile'][i]:8.4f}  "
              f"{horn['pct_variance_per_pc'][i]:5.1f}%  "
              f"cum {horn['cum_pct_variance'][i]:5.1f}%  {mark}")

    scores, loadings, _ = pca_scores(X, n_comp)
    ideo = ideology_by_pc(show.index, scores, max(retained, 1), targets)
    print(f"[step3] ideology location (PC space fit once on all "
          f"{diag['n_shows']} shows):")
    for name, res in ideo.items():
        top = res["per_pc"][res["top_pc"] - 1]
        print(f"         {name:<13} n={res['n']:<4} strongest PC{res['top_pc']} "
              f"|r|={res['top_abs_r']:.3f} CI{np.round(top['ci_95'], 3).tolist()} "
              f"| PC1 |r|={res['pc1_abs_r']:.3f} | top3 PCs {res['top3_pcs']}")

    interp = interpret_pcs(loadings, load_topic_meta(k), min(N_TOP_PCS_INTERPRETED, n_comp))
    print(f"[step4] leading PC anchors:")
    for pc in interp:
        pos = ", ".join(t["top_words"][0] for t in pc["positive_pole"][:4])
        neg = ", ".join(t["top_words"][0] for t in pc["negative_pole"][:4])
        flag = "  [CONTAMINANT-ANCHORED]" if pc["contaminant_anchored"] else ""
        print(f"         PC{pc['pc']}: +({pos})  vs  -({neg}){flag}")

    topic_meta = load_topic_meta(k)
    print(f"[step6] per-target Horn's (PCA + null REFIT on each target's shows):")
    per_target = {}
    for name, y_series in targets.items():
        res = horns_for_subsample(raw, show.index, name, y_series, b, seed, topic_meta)
        per_target[name] = res
        h = res["horns"]
        print(f"         {name:<13} n={res['n_shows']:<4} retained="
              f"{h['retained_dimensions']:<3} ({h['pct_variance_retained']:.1f}% var)"
              f"  ideology strongest PC{res['top_pc']} |r|={res['top_abs_r']:.3f} "
              f"CI{np.round(res['top_pc_ci'], 3).tolist()}  PC1 |r|="
              f"{res['pc1_abs_r']:.3f}  top3 {res['top3_pcs']}")

    return {"k": k, "step1_matrix": diag, "step2_horns": horn,
            "step3_ideology_by_pc": ideo, "step4_pc_interpretation": interp,
            "step6_per_target_horns": per_target}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b", type=int, default=1000, help="Horn's permutations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    targets = build_targets()
    print("[targets] " + ", ".join(f"{k} n={len(v)}" for k, v in targets.items()))
    print(f"[K] {PRIMARY_K_RATIONALE}")

    results = {k: run_for_k(k, args.b, args.seed, targets) for k in K_LIST}

    # ---- step 5: is the ideology-location conclusion stable across K? ----
    stability = {}
    for name in targets:
        per_k = {}
        for k in K_LIST:
            r = results[k]["step3_ideology_by_pc"][name]
            per_k[k] = {"top_pc": r["top_pc"], "top_abs_r": r["top_abs_r"],
                        "pc1_abs_r": r["pc1_abs_r"], "top3_pcs": r["top3_pcs"],
                        "retained": results[k]["step2_horns"]["retained_dimensions"]}
        same_pc = per_k[K_PRIMARY]["top_pc"] == per_k[K_ROBUST]["top_pc"]
        both_pc1 = all(v["top_pc"] == 1 for v in per_k.values())
        neither_pc1 = all(v["top_pc"] != 1 for v in per_k.values())
        stability[name] = {
            "per_k": per_k,
            "same_absolute_pc": bool(same_pc),
            "same_relative_conclusion": bool(both_pc1 or neither_pc1),
            "conclusion": ("ideology on PC1 at both K" if both_pc1 else
                           "ideology NOT on PC1 at either K" if neither_pc1 else
                           "FLIPS across K -- claim is K-dependent"),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": "clean",
        "corpus_label": "ad-span-cleaned corpus",
        "primary_k": K_PRIMARY,
        "robustness_k": K_ROBUST,
        "k_justification": PRIMARY_K_RATIONALE,
        "k_justification_detail": {
            "primary_k_criterion": "c_v coherence peak (outcome-independent)",
            "one_se_rule_would_select": K_ROBUST,
            "reported_at_both": True,
            "explicitly_not_justified_by": (
                "ideology R2 -- selecting K on the outcome then reporting that "
                "outcome at the selected K is circular"),
        },
        "method": {
            "transform": "CLR (Aitchison 1982) applied BEFORE PCA; required "
                         "because topic proportions are compositional",
            "dimensionality": "Horn's parallel analysis, column permutation, "
                              "95th percentile, contiguous retention from the top",
            "null_ordering": "permute raw proportion columns, THEN CLR "
                             "(mirrors nlp/embed_rank.py's permute-then-transform)",
            "pc_space": "fit once on all 204 shows; targets restrict rows, not basis",
            "scope": "dimensionality of THIS topic representation (downstream of K "
                     "and preprocessing), not of political discourse as such",
        },
        "embedding_arm_reference": {
            "retained_dimensions": 85,
            "note": "cited from record for comparison; same Horn's rule and "
                    "percentile as nlp/embed_rank.py",
        },
        "per_k": results,
        "step5_k_robustness": stability,
    }

    out = OUT_DIR / "topic_pca_dimensionality.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    rows = []
    for k in K_LIST:
        for name, res in results[k]["step3_ideology_by_pc"].items():
            for d in res["per_pc"]:
                rows.append({"k": k, "target": name, "n": res["n"], "pc": d["pc"],
                             "r": d["r"], "abs_r": d["abs_r"],
                             "ci_low": d["ci_95"][0], "ci_high": d["ci_95"][1],
                             "r2": d["r2"]})
    csv_path = OUT_DIR / "topic_pca_ideology_by_pc.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}\n[step5] K ROBUSTNESS OF THE IDEOLOGY-LOCATION CLAIM\n{'=' * 70}")
    for name, s in stability.items():
        print(f"  {name:<13} K=75: PC{s['per_k'][75]['top_pc']} "
              f"(|r|={s['per_k'][75]['top_abs_r']:.3f})   "
              f"K=30: PC{s['per_k'][30]['top_pc']} "
              f"(|r|={s['per_k'][30]['top_abs_r']:.3f})   -> {s['conclusion']}")
    print(f"\n[dims] retained (shared 204-show basis): "
          f"K=75 -> {results[75]['step2_horns']['retained_dimensions']}, "
          f"K=30 -> {results[30]['step2_horns']['retained_dimensions']}, "
          f"embedding arm (record) -> 85")

    print(f"\n{'=' * 70}\n[step6] PER-TARGET HORN'S (basis refit per target)\n{'=' * 70}")
    print(f"  {'target':<13} {'n':>4}  {'K=75 dims':>9} {'K=75 ideo':>12}   "
          f"{'K=30 dims':>9} {'K=30 ideo':>12}")
    for name in targets:
        a = results[K_PRIMARY]["step6_per_target_horns"][name]
        c = results[K_ROBUST]["step6_per_target_horns"][name]
        print(f"  {name:<13} {a['n_shows']:>4}  "
              f"{a['horns']['retained_dimensions']:>9} "
              f"{'PC%d |r|=%.3f' % (a['top_pc'], a['top_abs_r']):>12}   "
              f"{c['horns']['retained_dimensions']:>9} "
              f"{'PC%d |r|=%.3f' % (c['top_pc'], c['top_abs_r']):>12}")
    print("  (bases are target-specific and NOT comparable across rows; only "
          "whether ideology is on PC1 transfers)")
    print(f"\n[out] {out}\n[out] {csv_path}")


if __name__ == "__main__":
    main()
