"""Which of the 75 LDA topics carry the ideology signal?

Consumes existing artifacts only -- no re-embedding, no LDA refit, no new
target: the K=75 doc-topic matrix (data/output/lda_doctopic_k75_500.csv,
aggregated to show-level exactly as nlp/lda_ideology_204.py does),
data/output/ideology_targets_204.csv (ideology_primary, N=204), and the same
CLR + Ridge/LOSO-CV machinery already used for the ideology probe
(nlp/lda_ideology_clr.py's clr(), nlp/embed_ideology_nonlinear.py's
ridge_fit_predict/loso_cv/bootstrap_r2_ci).

Three complementary views, triangulated (see module-level task brief for
the full rationale -- CLR coefficients are geometric-mean-relative, and
Ridge spreads signal across correlated topics, so no single view is
trusted alone):

  View 1 -- univariate: Pearson/Spearman r between each topic's raw
            (pre-CLR) show-level share and ideology_primary, plus a
            permutation baseline (shuffle labels, track the null
            distribution of max |r| across all 75 topics).
  View 2 -- multivariate: standardized Ridge coefficients on the CLR
            feature set (single fit on all 204 shows, same GridSearchCV
            alpha-selection as the probe), with bootstrap-resampling
            (of shows) sign-stability -- a topic only counts as a robust
            marker if its coefficient sign is stable across resamples.
  View 3 -- predictive sufficiency: k-sweep (k=3,5,10,20,75) using only
            the top-k View-1-ranked CLR topic columns as Ridge features,
            same LOSO-CV protocol as the full extended_all probe.

Cross-view synthesis: one ranked table of topics where views agree,
topic-labeled with the K=75 top-words already extracted
(data/output/lda_k75_topic_words.csv), with the 5 known conversational-
filler topics (T61/T12/T57/T9/T46) flagged wherever they appear.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import config as pipeline_config
from .embed_ideology_nonlinear import (
    INNER_CV_FOLDS,
    RIDGE_ALPHAS,
    bootstrap_r2_ci,
    loso_cv,
    r2_from_predictions,
    ridge_fit_predict,
)
from .lda_ideology_204 import load_show_topics
from .lda_ideology_clr import clr

K = 75
FILLER_TOPICS = {"T61", "T12", "T57", "T9", "T46"}
N_TOP_UNIVARIATE = 10
N_BOOTSTRAP = 1000
N_PERMUTATION = 1000
SIGN_STABLE_THRESHOLD = 0.90
K_SWEEP = [3, 5, 10, 20, 75]


def load_topic_words() -> dict[str, str]:
    df = pd.read_csv(pipeline_config.OUTPUT_DIR / "lda_k75_topic_words.csv")
    return dict(zip(df["topic"], df["top_words"]))


def build_dataset():
    OUT = pipeline_config.OUTPUT_DIR
    targets = pd.read_csv(OUT / "ideology_targets_204.csv", dtype={"show_id": str}).set_index("show_id")
    show_topics, cols = load_show_topics()  # raw (pre-CLR) show-level mean topic proportions
    joined = show_topics.join(targets, how="inner").dropna(subset=["ideology_primary"])
    print(f"[data] N={len(joined)} shows (K={K} topics)")
    return joined, cols


def view1_univariate(joined: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    y = joined["ideology_primary"].to_numpy(dtype=float)
    rows = []
    for c in cols:
        x = joined[c].to_numpy(dtype=float)
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        rows.append({"topic": c, "pearson_r": r_p, "pearson_p": p_p, "spearman_r": r_s, "spearman_p": p_s})
    df = pd.DataFrame(rows)

    rng = np.random.default_rng(0)
    X = joined[cols].to_numpy(dtype=float)
    null_max_abs_r = np.empty(N_PERMUTATION)
    for b in range(N_PERMUTATION):
        y_shuf = rng.permutation(y)
        rs = np.array([stats.pearsonr(X[:, j], y_shuf)[0] for j in range(X.shape[1])])
        null_max_abs_r[b] = np.max(np.abs(rs))
    real_max_abs_r = df["pearson_r"].abs().max()
    perm_95 = float(np.percentile(null_max_abs_r, 95))
    perm_99 = float(np.percentile(null_max_abs_r, 99))
    pct_exceeding_real = float(np.mean(null_max_abs_r >= real_max_abs_r))

    perm_report = {
        "n_permutations": N_PERMUTATION,
        "real_max_abs_pearson_r": float(real_max_abs_r),
        "null_max_abs_r_95th_pct": perm_95,
        "null_max_abs_r_99th_pct": perm_99,
        "fraction_of_null_draws_ge_real_max": pct_exceeding_real,
        "verdict": (
            "real top association clears the permutation null" if real_max_abs_r > perm_95
            else "real top association does NOT clearly exceed the permutation null -- caution"
        ),
    }
    return df, perm_report


def view2_multivariate(joined: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    X_raw = joined[cols].to_numpy(dtype=float)
    X_clr = clr(X_raw)
    y = joined["ideology_primary"].to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_clr)

    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    grid = GridSearchCV(pipe, {"ridge__alpha": RIDGE_ALPHAS}, cv=INNER_CV_FOLDS, scoring="r2")
    grid.fit(X_clr, y)
    best_alpha = grid.best_params_["ridge__alpha"]
    main_coefs = grid.best_estimator_.named_steps["ridge"].coef_
    print(f"[view2] full-data fit: best alpha={best_alpha}, R2(in-sample)={grid.best_score_:.3f} (CV, not held-out)")

    rng = np.random.default_rng(0)
    n = len(y)
    boot_signs = np.zeros((N_BOOTSTRAP, len(cols)))
    boot_coefs = np.zeros((N_BOOTSTRAP, len(cols)))
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X_clr[idx], y[idx]
        if len(np.unique(yb)) < 2:
            boot_signs[b] = np.nan
            continue
        model = Ridge(alpha=best_alpha)
        model.fit(StandardScaler().fit_transform(Xb), yb)
        boot_coefs[b] = model.coef_
        boot_signs[b] = np.sign(model.coef_)

    main_sign = np.sign(main_coefs)
    pct_matching_main_sign = np.mean(boot_signs == main_sign[None, :], axis=0)

    rows = []
    for i, c in enumerate(cols):
        rows.append({
            "topic": c, "ridge_coef_standardized": float(main_coefs[i]),
            "sign": "positive(conservative)" if main_coefs[i] > 0 else "negative(liberal)",
            "pct_bootstrap_matching_sign": float(pct_matching_main_sign[i]),
            "sign_stable": bool(pct_matching_main_sign[i] >= SIGN_STABLE_THRESHOLD),
        })
    df = pd.DataFrame(rows)
    meta = {"best_alpha": best_alpha, "n_bootstrap": N_BOOTSTRAP, "sign_stable_threshold": SIGN_STABLE_THRESHOLD}
    return df, meta


def view3_k_sweep(joined: pd.DataFrame, cols: list[str], view1_df: pd.DataFrame) -> list[dict]:
    ranked_topics = view1_df.reindex(view1_df["pearson_r"].abs().sort_values(ascending=False).index)["topic"].tolist()
    X_raw = joined[cols].to_numpy(dtype=float)
    X_clr_full = clr(X_raw)
    col_index = {c: i for i, c in enumerate(cols)}
    y = joined["ideology_primary"].to_numpy(dtype=float)

    results = []
    for k in K_SWEEP:
        topic_subset = ranked_topics[:k]
        idxs = [col_index[t] for t in topic_subset]
        X = X_clr_full[:, idxs]

        preds = loso_cv(X, y, ridge_fit_predict)
        cv_r2 = r2_from_predictions(y, preds)
        lo, hi = bootstrap_r2_ci(y, preds)

        rng = np.random.default_rng(0)
        y_shuf = rng.permutation(y)
        perm_preds = loso_cv(X, y_shuf, ridge_fit_predict)
        perm_r2 = r2_from_predictions(y_shuf, perm_preds)

        print(f"[view3] k={k:<3} N={len(y)}  LOSO-CV R2={cv_r2:.4f}  95%CI=[{lo:.4f},{hi:.4f}]  perm R2={perm_r2:.4f}")
        results.append({
            "k": k, "n": len(y), "topics_used": topic_subset,
            "loso_cv_r2": float(cv_r2), "ci_95": [lo, hi], "permutation_r2": float(perm_r2),
        })
    return results


def run() -> None:
    OUT = pipeline_config.OUTPUT_DIR
    topic_words = load_topic_words()
    joined, cols = build_dataset()

    print("\n=== VIEW 1: univariate association ===")
    v1_df, v1_perm = view1_univariate(joined, cols)
    v1_df["top_words"] = v1_df["topic"].map(topic_words)
    top_conservative = v1_df.sort_values("pearson_r", ascending=False).head(N_TOP_UNIVARIATE)
    top_liberal = v1_df.sort_values("pearson_r", ascending=True).head(N_TOP_UNIVARIATE)
    print(f"[view1 permutation] {v1_perm['verdict']} "
          f"(real max|r|={v1_perm['real_max_abs_pearson_r']:.3f}, null 95th pct={v1_perm['null_max_abs_r_95th_pct']:.3f})")
    print("\nTop 10 most CONSERVATIVE-associated (positive r):")
    print(top_conservative[["topic", "pearson_r", "spearman_r", "top_words"]].to_string(index=False))
    print("\nTop 10 most LIBERAL-associated (negative r):")
    print(top_liberal[["topic", "pearson_r", "spearman_r", "top_words"]].to_string(index=False))

    print("\n=== VIEW 2: multivariate (Ridge + bootstrap sign-stability) ===")
    v2_df, v2_meta = view2_multivariate(joined, cols)
    v2_df["top_words"] = v2_df["topic"].map(topic_words)
    stable = v2_df[v2_df["sign_stable"]].copy()
    stable_ranked = stable.reindex(stable["ridge_coef_standardized"].abs().sort_values(ascending=False).index)
    print(f"[view2] {len(stable)}/{K} topics are sign-stable (>= {SIGN_STABLE_THRESHOLD*100:.0f}% "
          f"of {N_BOOTSTRAP} bootstraps agree on sign)")
    print("\nTop 15 sign-stable topics by |ridge coefficient|:")
    print(stable_ranked[["topic", "ridge_coef_standardized", "pct_bootstrap_matching_sign", "top_words"]]
          .head(15).to_string(index=False))

    print("\n=== VIEW 3: predictive sufficiency (k-sweep) ===")
    v3_results = view3_k_sweep(joined, cols, v1_df)

    # --- cross-view synthesis ---
    print("\n=== CROSS-VIEW SYNTHESIS ===")
    candidate_topics = set(top_conservative["topic"]) | set(top_liberal["topic"]) | set(stable_ranked["topic"].head(15))
    v1_idx = v1_df.set_index("topic")
    v2_idx = v2_df.set_index("topic")
    synth_rows = []
    for t in candidate_topics:
        r = v1_idx.loc[t, "pearson_r"]
        coef = v2_idx.loc[t, "ridge_coef_standardized"]
        stable_flag = v2_idx.loc[t, "sign_stable"]
        pct_match = v2_idx.loc[t, "pct_bootstrap_matching_sign"]
        v1_lean = "R" if r > 0 else "L"
        v2_lean = "R" if coef > 0 else "L"
        agree = stable_flag and (v1_lean == v2_lean)
        in_v1_top = t in set(top_conservative["topic"]) | set(top_liberal["topic"])
        in_v2_top = t in set(stable_ranked["topic"].head(15))
        both_views_flag = in_v1_top and in_v2_top
        synth_rows.append({
            "topic": t, "top_words": topic_words.get(t, ""),
            "univariate_r": float(r), "ridge_coef": float(coef),
            "sign_stable_pct": float(pct_match), "sign_stable": bool(stable_flag),
            "views_agree_on_direction": bool(agree), "flagged_by_both_top_lists": bool(both_views_flag),
            "lean": v1_lean if agree else "MIXED/UNSTABLE",
            "is_filler_topic": t in FILLER_TOPICS,
        })
    synth_df = pd.DataFrame(synth_rows)
    synth_df["_abs_r"] = synth_df["univariate_r"].abs()
    synth_df = synth_df.sort_values(
        by=["flagged_by_both_top_lists", "sign_stable", "_abs_r"], ascending=[False, False, False]
    ).drop(columns="_abs_r")
    print(synth_df[["topic", "top_words", "univariate_r", "ridge_coef", "sign_stable", "views_agree_on_direction",
                     "flagged_by_both_top_lists", "lean", "is_filler_topic"]].to_string(index=False))

    n_filler_flagged = synth_df["is_filler_topic"].sum()
    if n_filler_flagged:
        print(f"\n[FLAG] {n_filler_flagged} conversational-filler topic(s) appear in the ideology-associated "
              f"set -- interpret with care (possible stylistic confound, not content).")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_shows": len(joined), "k_topics": K,
        "view1_permutation": v1_perm,
        "view1_top_conservative": top_conservative.drop(columns=[]).to_dict("records"),
        "view1_top_liberal": top_liberal.to_dict("records"),
        "view2_meta": v2_meta,
        "view2_sign_stable_top15": stable_ranked.head(15).to_dict("records"),
        "view3_k_sweep": v3_results,
        "cross_view_synthesis": synth_df.to_dict("records"),
        "filler_topics": sorted(FILLER_TOPICS),
        "n_filler_topics_flagged_in_synthesis": int(n_filler_flagged),
    }
    report_path = OUT / "topic_ideology_synthesis_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path}")


if __name__ == "__main__":
    run()
