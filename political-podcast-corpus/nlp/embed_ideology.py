"""Does the embedding space's leading structure align with ideology?

    python -m nlp.embed_ideology [--target-words 500] [--top-k-pcs 30]

NOT a dimension-count comparison to a CA arm (that arm isn't built; out of
scope here). Instead: at what rank does an ideological axis appear in the
chunk-embedding PC space, and how much of DIME CFscore is recoverable from
the leading PCs? Unit: show-level primary (mean PC score per show vs.
CFscore, ~120 matched shows after the Step 0 coverage audit). Chunk-level
with show-clustered inference is a cheap robustness pass only.

Step 0 (coverage audit) was done interactively before this module existed --
data/output/shows_205_host_dime_scores.csv, `coverage` in {full, partial}
gives a genuine avg_host_cfscore (verified: no 0.0-placeholder rows; `none`/
`no_host_listed` are correctly NaN, not defaulted). N=120 of 204 shows have
both a real CFscore and embeddings. Distribution is bimodal, real spread
(-1.39 to 2.39), left-skewed (77 left- vs 43 right-leaning shows) -- Pearson
AND Spearman are both reported specifically because the bimodal/skewed shape
could make Pearson r sensitive to the dense left cluster.

This module consumes existing chunk-level pooled embeddings -- it does not
re-embed or re-pool anything.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

from pipeline import config as pipeline_config
from .embed_rank import transform_variant

CFSCORE_PATH = Path("data/output/shows_205_host_dime_scores.csv")
UNIVARIATE_R_MEANINGFUL = 0.3
REGRESSION_KS = [2, 5, 10, 30]


def load_cfscore() -> pd.DataFrame:
    """Genuine avg_host_cfscore only -- coverage in {full, partial}.

    `none`/`no_host_listed` rows are dropped, not treated as centrist; this
    was verified against the raw file (zero 0.0-placeholder rows) before
    this module was written.
    """
    dime = pd.read_csv(CFSCORE_PATH)
    dime["show_id"] = dime["show_id"].astype(str)
    matched = dime[dime["coverage"].isin(["full", "partial"]) & dime["avg_host_cfscore"].notna()]
    return matched.set_index("show_id")[["show", "avg_host_cfscore", "coverage"]]


def chunk_pc_scores(X: np.ndarray, variant: str, k: int, n_components: int,
                     n_iter: int = 4, seed: int = 0) -> np.ndarray:
    """Project chunks onto the top `n_components` PCs under `variant`.

    No show-residualization here (unlike nlp/embed_rank.py's null test) --
    this analysis is specifically about INTER-show variation, so removing
    each show's own mean first would delete the exact signal being measured.
    """
    from sklearn.utils.extmath import randomized_svd
    Xt = transform_variant(X, variant, k, n_iter, seed)
    U, S, _Vt = randomized_svd(Xt, n_components=n_components, random_state=seed, n_iter=n_iter)
    return U * S  # (n_chunks, n_components) PC scores


def load_show_level_data(target: int, top_k_pcs: int) -> dict:
    OUT = pipeline_config.OUTPUT_DIR
    chunks_bert = pd.read_csv(OUT / f"chunks_{target}_bert.csv",
                               usecols=["chunk_id", "collection_id", "bert_text", "n_content_words"])
    meta = json.loads((OUT / f"chunk_embeddings_{target}_weighted.meta.json").read_text())
    chunk_id_order = meta["chunk_id_order"]
    X_chunk = np.load(OUT / f"chunk_embeddings_{target}_weighted.npy")

    by_id = chunks_bert.set_index("chunk_id")
    show_ids = by_id.loc[chunk_id_order, "collection_id"].astype(str).to_numpy()
    n_raw_words = by_id.loc[chunk_id_order, "bert_text"].str.split().apply(len).to_numpy()

    return {"X_chunk": X_chunk, "show_ids": show_ids, "n_raw_words": n_raw_words}


def build_show_level_pc_matrix(X_chunk: np.ndarray, show_ids: np.ndarray, n_raw_words: np.ndarray,
                                variant: str, k: int, top_k_pcs: int) -> pd.DataFrame:
    scores = chunk_pc_scores(X_chunk, variant, k, top_k_pcs)
    cols = [f"PC{i+1}" for i in range(top_k_pcs)]
    df = pd.DataFrame(scores, columns=cols)
    df["show_id"] = show_ids
    df["n_raw_words"] = n_raw_words

    agg = df.groupby("show_id")[cols].mean()
    agg["mean_raw_words_per_chunk"] = df.groupby("show_id")["n_raw_words"].mean()
    agg["n_chunks"] = df.groupby("show_id").size()
    return agg


def step2_univariate(show_pcs: pd.DataFrame, cfscore: pd.Series, top_k_pcs: int) -> list[dict]:
    rows = []
    for i in range(top_k_pcs):
        pc = f"PC{i+1}"
        r_p, p_p = pearsonr(show_pcs[pc], cfscore)
        r_s, p_s = spearmanr(show_pcs[pc], cfscore)
        rows.append({
            "pc": i + 1,
            "pearson_r": float(r_p), "pearson_p": float(p_p),
            "spearman_r": float(r_s), "spearman_p": float(p_s),
            "meaningful": bool(abs(r_p) > UNIVARIATE_R_MEANINGFUL),
        })
    return rows


def step3_multivariate(show_pcs: pd.DataFrame, cfscore: pd.Series, ks: list[int]) -> list[dict]:
    n = len(cfscore)
    y = cfscore.to_numpy()
    out = []
    for k in ks:
        X = show_pcs[[f"PC{i+1}" for i in range(k)]].to_numpy()
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        loo = LeaveOneOut()
        preds = np.empty(n)
        for train_idx, test_idx in loo.split(X):
            m = LinearRegression().fit(X[train_idx], y[train_idx])
            preds[test_idx] = m.predict(X[test_idx])
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        cv_r2 = 1 - ss_res / ss_tot

        out.append({"k": k, "r2": float(r2), "adj_r2": float(adj_r2), "cv_r2_loo": float(cv_r2)})
    return out


def step4_best_axis(show_pcs: pd.DataFrame, cfscore: pd.Series, univariate: list[dict]) -> dict:
    best = max(univariate, key=lambda r: abs(r["pearson_r"]))
    best_pc = f"PC{best['pc']}"
    axis_scores = show_pcs[best_pc]

    r_raw, p_raw = pearsonr(axis_scores, cfscore)

    confounders = show_pcs[["mean_raw_words_per_chunk", "n_chunks"]].to_numpy()
    resid_model = LinearRegression().fit(confounders, axis_scores)
    axis_residual = axis_scores - resid_model.predict(confounders)
    r_resid, p_resid = pearsonr(axis_residual, cfscore)

    extremes = pd.concat([
        pd.DataFrame({"show_id": axis_scores.sort_values().index[:3], "end": "most_negative_axis"}),
        pd.DataFrame({"show_id": axis_scores.sort_values().index[-3:], "end": "most_positive_axis"}),
    ])

    return {
        "best_pc": best["pc"],
        "r_before_confounder_control": float(r_raw), "p_before": float(p_raw),
        "r_after_confounder_control": float(r_resid), "p_after": float(p_resid),
        "labeled_extremes": extremes.to_dict("records"),
        "axis_scores": axis_scores.to_dict(),
        "axis_residual": dict(zip(axis_scores.index, axis_residual)),
    }


def step5_chunk_level_clustered(target: int, X_chunk: np.ndarray, show_ids: np.ndarray,
                                 cfscore: pd.Series, variant: str, k: int) -> dict:
    import statsmodels.api as sm

    scores = chunk_pc_scores(X_chunk, variant, k, 2)
    cf_map = cfscore.to_dict()
    keep = np.array([s in cf_map for s in show_ids])
    X2 = scores[keep]
    groups = show_ids[keep]
    y = np.array([cf_map[s] for s in groups])

    Xc = sm.add_constant(X2)
    model = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return {
        "n_chunks": int(keep.sum()),
        "coef_pc1": float(model.params[1]), "p_pc1_clustered": float(model.pvalues[1]),
        "coef_pc2": float(model.params[2]), "p_pc2_clustered": float(model.pvalues[2]),
        "r2": float(model.rsquared),
    }


def make_figures(report: dict, out_dir: Path, target: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uni = report["step2_univariate_primary"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([r["pc"] for r in uni], [abs(r["pearson_r"]) for r in uni])
    ax.axhline(UNIVARIATE_R_MEANINGFUL, color="red", linestyle="--", label=f"|r|={UNIVARIATE_R_MEANINGFUL}")
    ax.set_xlabel("PC index"); ax.set_ylabel("|Pearson r| vs CFscore")
    ax.set_title(f"Where does ideology appear in the PC spectrum? (target={target}, N={report['n_matched']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"ideology_r_by_pc_{target}.png", dpi=120)
    plt.close(fig)

    axis = report["step4_best_axis"]
    scores_dict = axis["axis_scores"]
    resid_dict = axis["axis_residual"]
    cf = report["_cfscore_series"]
    for label, d, r_key in [("raw", scores_dict, "r_before_confounder_control"),
                             ("residualized", resid_dict, "r_after_confounder_control")]:
        fig, ax = plt.subplots(figsize=(7, 6))
        xs = [d[s] for s in cf.index]
        ys = cf.tolist()
        ax.scatter(xs, ys, alpha=0.6)
        # Stagger label offsets -- extreme shows cluster close together in x
        # (both PC22 and CFscore), so fixed same-point labels overlapped
        # illegibly. Alternate above/below with an arrow back to the point.
        for j, rec in enumerate(axis["labeled_extremes"]):
            sid = rec["show_id"]
            if sid in d:
                dy = 14 if j % 2 == 0 else -18
                ax.annotate(report["_show_names"].get(sid, sid), (d[sid], cf.get(sid, np.nan)),
                            fontsize=7, alpha=0.85, xytext=(0, dy), textcoords="offset points",
                            ha="center", arrowprops=dict(arrowstyle="-", alpha=0.4, lw=0.6))
        ax.set_xlabel(f"PC{axis['best_pc']} show-level score ({label})")
        ax.set_ylabel("avg_host_cfscore")
        ax.set_title(f"Best-aligned axis vs CFscore ({label}), r={axis[r_key]:.3f}, N={len(cf)}")
        fig.tight_layout()
        fig.savefig(out_dir / f"ideology_best_axis_{label}_{target}.png", dpi=120)
        plt.close(fig)


def run(target: int, top_k_pcs: int) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    cfscore_df = load_cfscore()
    print(f"[cfscore] N genuine (coverage full/partial) = {len(cfscore_df)}")

    data = load_show_level_data(target, top_k_pcs)
    X_chunk, show_ids, n_raw_words = data["X_chunk"], data["show_ids"], data["n_raw_words"]

    report: dict = {
        "target": target, "top_k_pcs": top_k_pcs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_matched": 0,
    }

    for variant_name, variant, k in [("primary_centered", "centered", 0),
                                      ("robustness_all_but_top_1", "all_but_top", 1)]:
        print(f"\n=== variant={variant_name} ===")
        show_pcs = build_show_level_pc_matrix(X_chunk, show_ids, n_raw_words, variant, k, top_k_pcs)
        joined = show_pcs.join(cfscore_df, how="inner")
        n_matched = len(joined)
        print(f"[join] N matched (embeddings + genuine CFscore) = {n_matched}")

        cfscore = joined["avg_host_cfscore"]
        show_pcs_matched = joined[show_pcs.columns]

        univariate = step2_univariate(show_pcs_matched, cfscore, top_k_pcs)
        lowest_meaningful = next((r["pc"] for r in univariate if r["meaningful"]), None)
        print(f"[step2] lowest-rank PC with |pearson r|>{UNIVARIATE_R_MEANINGFUL}: {lowest_meaningful}")

        multivariate = step3_multivariate(show_pcs_matched, cfscore, REGRESSION_KS)
        for row in multivariate:
            print(f"[step3] k={row['k']:>2}  R2={row['r2']:.3f}  adjR2={row['adj_r2']:.3f}  "
                  f"CV(LOO)R2={row['cv_r2_loo']:.3f}")

        if variant_name == "primary_centered":
            report["n_matched"] = n_matched
            report["step2_univariate_primary"] = univariate
            report["step3_multivariate_primary"] = multivariate
            report["lowest_meaningful_pc_primary"] = lowest_meaningful

            axis = step4_best_axis(show_pcs_matched, cfscore, univariate)
            print(f"[step4] best axis PC{axis['best_pc']}: r_before={axis['r_before_confounder_control']:.3f} "
                  f"r_after_confounder_control={axis['r_after_confounder_control']:.3f}")
            report["step4_best_axis"] = {k_: v for k_, v in axis.items()
                                          if k_ not in ("axis_scores", "axis_residual")}
            report["_cfscore_series"] = cfscore
            report["_show_names"] = cfscore_df["show"].to_dict()

            clustered = step5_chunk_level_clustered(target, X_chunk, show_ids, cfscore, variant, k)
            print(f"[step5-chunk] N chunks={clustered['n_chunks']} "
                  f"coef_pc1={clustered['coef_pc1']:+.4f} (p_clustered={clustered['p_pc1_clustered']:.4f}) "
                  f"coef_pc2={clustered['coef_pc2']:+.4f} (p_clustered={clustered['p_pc2_clustered']:.4f})")
            report["step5_chunk_level_clustered_primary"] = clustered

            # make_figures needs the untrimmed axis dict (per-show scores/residuals);
            # `report["step4_best_axis"]` above intentionally excludes those from the
            # saved JSON (120 raw per-show floats isn't report content).
            figure_data = dict(report, step4_best_axis=axis)
            make_figures(figure_data, OUT, target)
        else:
            report["step2_univariate_robustness"] = univariate
            report["step3_multivariate_robustness"] = multivariate
            report["lowest_meaningful_pc_robustness"] = lowest_meaningful

    cv_r2_by_k = {row["k"]: row["cv_r2_loo"] for row in report["step3_multivariate_primary"]}
    regime = ("ideology concentrated in the top few PCA components (convergence-with-CA story)"
              if cv_r2_by_k[2] > 0.3 else
              "ideology buried under stylistic variation within the PCA basis, only recoverable with many PCs"
              if cv_r2_by_k[30] > cv_r2_by_k[2] + 0.1 else
              "ideology not recoverable from the TOP-30 PCA COMPONENTS specifically, at any tested k")
    verdict = (
        f"On N={report['n_matched']} shows with genuine DIME CFscore coverage, the lowest-rank PC "
        f"with |Pearson r|>{UNIVARIATE_R_MEANINGFUL} against ideology is "
        f"{'PC' + str(report['lowest_meaningful_pc_primary']) if report['lowest_meaningful_pc_primary'] else 'none in the top ' + str(top_k_pcs)}. "
        f"Leave-one-show-out CV R2 using the top 2 PCs is {cv_r2_by_k[2]:.3f}, rising to {cv_r2_by_k[30]:.3f} "
        f"at 30 PCs. Verdict (SCOPED TO PCA INPUT, see correction below): {regime}."
    )
    report["verdict"] = verdict
    print(f"\n=== VERDICT ===\n{verdict}")

    # CORRECTION (added after nlp/embed_ideology_nonlinear.py's follow-up probe):
    # the verdict above is true only as a statement about the top-30 PCA basis --
    # it does NOT mean ideology is linearly unrecoverable from the embedding space
    # in general. PCA is unsupervised (keeps highest-VARIANCE directions); it can,
    # and here did, discard directions correlated with an outcome it was never
    # shown. A regularized Ridge regression on the FULL 768-dim show-level
    # embedding (not top-k PCs), under IDENTICAL leave-one-show-out CV on the
    # same N=120 shows, recovers CV R2~=0.21 -- verified clean via a per-fold
    # alpha-selection stability check (118/120 folds picked the same alpha, no
    # grid-boundary hugging) and a permutation negative control (shuffled-y CV
    # R2 ~= 0, ruling out a scaling/CV leak). The correct statement is:
    # "linear regression on the top-30 variance-maximizing PCs found no signal,
    # but this was a PCA-truncation artifact, not evidence that ideology is
    # linearly unrecoverable -- full-embedding Ridge recovers R2~=0.21."
    # See data/output/nonlinear_probe_report_500.json for the full comparison
    # (Ridge R2=0.208 vs Kernel Ridge RBF R2=0.272 -- nonlinearity adds only a
    # modest, within-CI bump over the full-embedding linear result, so the
    # relationship looks mostly linear once PCA truncation is removed).
    report["verdict_correction"] = (
        "CORRECTION: the verdict above describes recoverability from the top-30 PCA "
        "components specifically, not from the embedding space in general. That "
        "distinction matters: PCA is unsupervised and can discard outcome-relevant "
        "directions. A Ridge regression on the FULL 768-dim show-level embedding, "
        "under the identical N=120 leave-one-show-out harness, recovers CV R2~=0.21 "
        "(verified via stable alpha selection across folds and a permutation "
        "negative control showing ~0 R2 on shuffled CFscore). Kernel Ridge (RBF) "
        "reaches R2=0.272 -- higher, but within the same confidence interval, so "
        "nonlinearity adds only a modest bump rather than unlocking a qualitatively "
        "different signal. Corrected statement: linear regression on the top-30 "
        "PCs found no signal because of PCA truncation, not because ideology is "
        "linearly unrecoverable -- full-embedding Ridge recovers R2~=0.21. "
        "See nonlinear_probe_report_500.json."
    )
    print(f"\n=== VERDICT CORRECTION ===\n{report['verdict_correction']}")

    del report["_cfscore_series"]
    del report["_show_names"]
    report_path = OUT / f"embed_ideology_report_{target}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-words", type=int, default=500)
    ap.add_argument("--top-k-pcs", type=int, default=30)
    args = ap.parse_args()
    run(args.target_words, args.top_k_pcs)


if __name__ == "__main__":
    main()
