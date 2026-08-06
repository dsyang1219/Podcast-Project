"""Closing diagnostic: what do the top ~10 chunk-level embedding PCs (primary
anisotropy variant, centered-only, same PCA nlp/embed_dimensions.py /
nlp/embed_rank.py / nlp/embed_ideology.py all share) actually capture, now
that the register-quantification test came back negative (0-1/10 PCs
register-explained in nlp/register_pc_regression.py / nlp/register_ica_check.py)?

Three non-exclusive tests, run uniformly on all top ~10 PCs -- no
cherry-picking, no new re-embedding/re-PCA/re-feature-engineering:

  Test A (topic):        PC ~ K=75 CLR topic vector (chunk-level), CV R^2.
  Test B (show identity): between-show eta^2 -- REUSED from the already-
                          computed data/output/embedding_dimensions_report_500
                          .json Step 3 (no recomputation, per out-of-scope note).
  Test C (diffuse):      PC ~ register features + topics combined, CV R^2 --
                          the residual after both is the genuinely-diffuse
                          share.

register-only R^2 per PC is REUSED from
data/output/register_quant_step2_report.json (no recomputation).

var_explained % per PC: exact total variance (trace of X^T X, already computed
in data/output/pc_variance_exact.json for PC1-5/22) extended here to PC1-10
via one randomized_svd(n_components=10) call on the same centered matrix --
this is not a new PCA fit, it's the same decomposition already used
everywhere else in this line of work, just read out to more components.

Same Ridge(alpha=1.0, numerical-stability-only)/10-fold-CV/permutation-null
protocol as the register test, so numbers are directly comparable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

from pipeline import config as pipeline_config
from .register_pc_regression import FEATURES as REGISTER_FEATURES, N_FOLDS, RIDGE_ALPHA, r2_from_predictions
from .lda_ideology_clr import clr

TARGET = 500
N_PCS = 10
K_TOPICS = 75


def r2(y_true, y_pred):
    return r2_from_predictions(y_true, y_pred)


def load_topic_matrix() -> pd.DataFrame:
    OUT = pipeline_config.OUTPUT_DIR
    chunks = pd.read_csv(OUT / "chunks_500_nolemma.csv", usecols=["chunk_id"])
    doc_topic = pd.read_csv(OUT / f"lda_doctopic_k{K_TOPICS}_500.csv")
    cols = [c for c in doc_topic.columns if c.startswith("T")]
    X_raw = doc_topic[cols].to_numpy()
    assert (X_raw > 0).all(), "non-positive topic proportion, CLR needs epsilon"
    X_clr = clr(X_raw)
    out = pd.DataFrame(X_clr, columns=cols)
    out["chunk_id"] = doc_topic["chunk_id"]
    return out, cols


def compute_pc_variance_pct(n_components: int) -> dict[str, float]:
    OUT = pipeline_config.OUTPUT_DIR
    exact = json.loads((OUT / "pc_variance_exact.json").read_text())
    total_sumsq = exact["total_sumsq_exact"]

    meta = json.loads((OUT / f"chunk_embeddings_{TARGET}_weighted.meta.json").read_text())
    X = np.load(OUT / f"chunk_embeddings_{TARGET}_weighted.npy")
    Xc = X - X.mean(axis=0, keepdims=True)
    _, S, _ = randomized_svd(Xc, n_components=n_components, random_state=0, n_iter=4)
    eigvals = S ** 2
    return {f"PC{i + 1}": float(100 * eigvals[i] / total_sumsq) for i in range(n_components)}


def cv_r2_and_perm(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    model = Ridge(alpha=RIDGE_ALPHA)
    preds = cross_val_predict(model, X, y, cv=kf)
    cv_r2 = r2(y, preds)

    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y)
    perm_preds = cross_val_predict(model, X, y_shuf, cv=kf)
    perm_r2 = r2(y_shuf, perm_preds)
    return float(cv_r2), float(perm_r2)


def run() -> None:
    OUT = pipeline_config.OUTPUT_DIR
    pc_cols = [f"PC{i}" for i in range(1, N_PCS + 1)]
    pc_df = pd.read_pickle(OUT / f"embed_dimensions_feature_table_{TARGET}.pkl")[["chunk_id"] + pc_cols]
    print(f"[pcs] loaded cached PC1..PC{N_PCS} for {len(pc_df)} chunks (no PCA refit)")

    reg_df = pd.read_csv(OUT / f"register_features_{TARGET}.csv")
    topic_df, topic_cols = load_topic_matrix()
    print(f"[topics] loaded K={K_TOPICS} CLR-transformed topic vectors for {len(topic_df)} chunks")

    df = pc_df.merge(reg_df[["chunk_id"] + REGISTER_FEATURES], on="chunk_id", how="inner")
    df = df.merge(topic_df, on="chunk_id", how="inner")
    df = df.dropna(subset=pc_cols + REGISTER_FEATURES + topic_cols)
    print(f"[join] N={len(df)}")

    reg_scaler = StandardScaler()
    X_reg = reg_scaler.fit_transform(df[REGISTER_FEATURES].to_numpy(dtype=float))
    X_topics = df[topic_cols].to_numpy(dtype=float)  # already CLR-transformed; scale for Ridge fairness
    topic_scaler = StandardScaler()
    X_topics = topic_scaler.fit_transform(X_topics)
    X_combined = np.hstack([X_reg, X_topics])

    print("[variance] computing PC1-10 exact variance-explained %...")
    var_pct = compute_pc_variance_pct(N_PCS)

    register_report = json.loads((OUT / "register_quant_step2_report.json").read_text())
    register_r2_by_pc = {r["pc"]: r["cv_r2"] for r in register_report["per_pc"]}  # keys are "PC1".."PC10"

    dims_report = json.loads((OUT / "embedding_dimensions_report_500.json").read_text())
    eta2_by_pc = {pc: r["between_show_eta_squared"] for pc, r in dims_report["step3_between_within_variance"].items()}

    results = []
    for pc in pc_cols:
        y = df[pc].to_numpy(dtype=float)

        topic_r2, topic_perm = cv_r2_and_perm(X_topics, y)
        combined_r2, combined_perm = cv_r2_and_perm(X_combined, y)
        register_r2 = register_r2_by_pc[pc]
        eta2 = eta2_by_pc[pc]
        vpct = var_pct[pc]

        results.append({
            "pc": pc, "n": len(y),
            "var_explained_pct": vpct,
            "register_only_r2": register_r2,
            "topics_only_r2": topic_r2, "topics_only_perm_r2": topic_perm,
            "register_plus_topics_r2": combined_r2, "register_plus_topics_perm_r2": combined_perm,
            "between_show_eta2": eta2,
        })
        print(f"[{pc}] var%={vpct:.2f}  register-only R2={register_r2:.3f}  "
              f"topics-only R2={topic_r2:.3f} (perm={topic_perm:.3f})  "
              f"combined R2={combined_r2:.3f} (perm={combined_perm:.3f})  eta2={eta2:.3f}")

    avg = {
        "var_explained_pct": float(np.mean([r["var_explained_pct"] for r in results])),
        "register_only_r2": float(np.mean([r["register_only_r2"] for r in results])),
        "topics_only_r2": float(np.mean([r["topics_only_r2"] for r in results])),
        "register_plus_topics_r2": float(np.mean([r["register_plus_topics_r2"] for r in results])),
        "between_show_eta2": float(np.mean([r["between_show_eta2"] for r in results])),
    }
    print("\n=== COLUMN AVERAGES (top {} PCs) ===".format(N_PCS))
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")

    topic_dominant = avg["topics_only_r2"] > 0.4
    show_dominant = avg["between_show_eta2"] > 0.4
    residual = 1 - avg["register_plus_topics_r2"]
    if topic_dominant:
        verdict = ("TOPIC: the embedding space is substantially topic-organized -- average "
                   f"topics-only CV R2={avg['topics_only_r2']:.3f} across the top {N_PCS} PCs clears "
                   "the 0.4 bar. This explains why register features failed: the dominant PCs are "
                   "mostly 'what they talk about,' not 'how they talk.'")
    elif show_dominant:
        verdict = ("SHOW IDENTITY: the top PCs are dominated by between-show variance "
                   f"(average eta^2={avg['between_show_eta2']:.3f}), i.e. largely detecting WHICH "
                   "show a chunk came from rather than encoding any generalizable stylistic or "
                   "topical axis.")
    else:
        verdict = ("DIFFUSE: neither topics nor register features, alone or combined, explain the "
                   f"top PCs well (average register+topics R2={avg['register_plus_topics_r2']:.3f}, "
                   f"leaving ~{residual*100:.0f}% of each PC's variance unaccounted for), and "
                   f"between-show identity is only a partial driver (average eta^2="
                   f"{avg['between_show_eta2']:.3f}). The dominant embedding dimensions are "
                   "genuinely diffuse -- not cleanly captured by topic, show identity, or "
                   "interpretable register features. This is the honest negative result.")
    print(f"\n=== VERDICT ===\n{verdict}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_words": TARGET, "n_pcs": N_PCS, "k_topics": K_TOPICS, "n_folds": N_FOLDS,
        "n_matched_chunks": len(df),
        "per_pc": results, "column_averages": avg, "verdict": verdict,
    }
    report_path = OUT / "embed_pc_diagnosis_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path}")

    print(f"\n{'PC':<6}{'var%':<8}{'register':<11}{'topics':<10}{'reg+topics':<12}{'between-show eta2':<20}")
    for r in results:
        print(f"{r['pc']:<6}{r['var_explained_pct']:<8.2f}{r['register_only_r2']:<11.3f}"
              f"{r['topics_only_r2']:<10.3f}{r['register_plus_topics_r2']:<12.3f}{r['between_show_eta2']:<20.3f}")
    print(f"{'AVG':<6}{avg['var_explained_pct']:<8.2f}{avg['register_only_r2']:<11.3f}"
          f"{avg['topics_only_r2']:<10.3f}{avg['register_plus_topics_r2']:<12.3f}{avg['between_show_eta2']:<20.3f}")


if __name__ == "__main__":
    run()
