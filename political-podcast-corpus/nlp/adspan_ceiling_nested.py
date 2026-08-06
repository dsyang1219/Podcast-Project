"""How well CAN ideology be predicted? Nested-CV ceiling on the ad-cleaned corpus.

This is the honest replacement for nlp/adspan_tuning_ceiling.py, whose own
docstring flags that every number in it is OUTCOME-SELECTED (variants scored on
the same LOSO R2 used to rank them) and that "the honest version requires nested
CV -- variant selection inside each LOSO fold". That is what this module does.

WHAT IS AND IS NOT CLAIMED
--------------------------
This characterizes RECOVERABILITY -- can ideology be predicted from a show's
discourse if you try hard. It says NOTHING about DOMINANCE -- whether ideology
is a high-variance axis of that discourse. The PCA finding that ideology is a
low-variance/secondary axis is untouched by any R2 reported here, however high.
A model can recover a faint axis very well; recovering it does not promote it.
The two questions are kept apart everywhere in the output.

HONESTY CONTROLS (enforced structurally, not by convention)
-----------------------------------------------------------
1. Every hyperparameter is chosen by GridSearchCV INSIDE the outer training
   split. loso_cv() only ever hands a fit_predict_fn the training rows, so no
   code path exists by which a held-out show reaches a scaler, an idf vector,
   or a hyperparameter search.
2. Only OUTER-loop R2 is reported. Inner-loop scores are never written out.
3. Per-show outer errors are stored for every model so comparisons use a PAIRED
   test on the same folds, not overlapping CIs or a bigger point estimate.
4. Arm 1 is reported as a RANGE with the ridge baseline highlighted; the max is
   never headlined.
5. Multiplicity: Arm 2 runs 6 comparisons against one baseline, so Holm-adjusted
   p-values are reported alongside raw ones.

WHAT LEAKS, STATED PLAINLY
--------------------------
The LDA topics, the TF-IDF vocabulary, and the sentence embeddings are all fit
on the FULL corpus before CV. None of them see the target, so this is
unsupervised-representation leakage, not label leakage -- the same status the
project already accepts for its topic features. It is not zero risk (the
representation is shaped by the held-out show's text) and is recorded in the
report as a known limitation. Refitting LDA inside 204 folds is out of scope
here; what is fit in-fold is every supervised component: idf weights, scaling,
and all hyperparameters.

    .venv/bin/python -m nlp.adspan_ceiling_nested --arm 1
    .venv/bin/python -m nlp.adspan_ceiling_nested --arm 2
    .venv/bin/python -m nlp.adspan_ceiling_nested --arm 3
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import config as pipeline_config
from .adspan_phase_b import OUT_DIR as ADSPAN_DIR
from .adspan_target_robustness import build_targets
from .embed_ideology_nonlinear import (
    INNER_CV_FOLDS,
    RIDGE_ALPHAS,
    bootstrap_r2_ci,
    loso_cv,
    r2_from_predictions,
)
from .lda_ideology_clr import clr

OUT = pipeline_config.OUTPUT_DIR
REPORT_DIR = ADSPAN_DIR
CHUNKS_CLEAN = ADSPAN_DIR / "chunks_500_nolemma_adclean.csv"
REGISTER_CLEAN = OUT / "register_features_500_adclean.csv"
EMBED_CLEAN = ADSPAN_DIR / "show_embeddings_adclean.npy"
EMBED_CLEAN_IDS = ADSPAN_DIR / "show_embeddings_adclean_ids.json"
EMBED_DIRTY_FALLBACK = "chunk_embeddings_500_weighted.npy"

K_PRIMARY = 75    # c_v peak (see nlp/adspan_phase_c.py); K=30 is the 1SE choice
K_ROBUST = 30
SEED = 0
TFIDF_MAX_FEATURES = 50_000
TFIDF_MIN_DF = 5

REGISTER_COLS = [
    "citation_density", "abstraction", "present_tense_news_markers",
    "entity_mix", "sentence_complexity", "first_vs_third_person",
]


# --------------------------------------------------------------- features ----
def show_ids_and_chunks() -> pd.DataFrame:
    return pd.read_csv(CHUNKS_CLEAN, usecols=["chunk_id", "collection_id"],
                       dtype={"collection_id": str})


def build_topics(k: int) -> pd.DataFrame:
    """Show x topic CLR matrix from the cleaned-corpus LDA doc-topic file."""
    path = ADSPAN_DIR / f"doctopic_adclean_k{k}.csv"
    dt = pd.read_csv(path, dtype={"collection_id": str})
    cols = [c for c in dt.columns if c[:1] in ("T", "t") and c[1:].isdigit()]
    if len(cols) != k:
        raise ValueError(f"{path.name}: found {len(cols)} topic columns, expected {k}")
    show = dt.groupby("collection_id")[cols].mean()
    X = clr(np.where(show.to_numpy(dtype=float) <= 0, 1e-12, show.to_numpy(dtype=float)))
    return pd.DataFrame(X, index=show.index)


def build_register() -> pd.DataFrame:
    """Show-level mean of the 6 a-priori register features, cleaned corpus."""
    if not REGISTER_CLEAN.exists():
        raise FileNotFoundError(
            f"{REGISTER_CLEAN} missing -- run: .venv/bin/python -m nlp.register_features "
            "--corpus clean. Do NOT substitute register_features_500.csv: the cleaned "
            "corpus is re-chunked, so those values belong to different text.")
    reg = pd.read_csv(REGISTER_CLEAN)
    ch = show_ids_and_chunks()
    m = ch.merge(reg, on="chunk_id", how="inner")
    return m.groupby("collection_id")[REGISTER_COLS].mean()


def build_counts() -> tuple[pd.Index, "np.ndarray", CountVectorizer]:
    """Show x term COUNT matrix. Vocabulary is built once on the full corpus --
    unsupervised and label-free, same status as the LDA fit. The supervised part
    of TF-IDF (the idf weighting) is fit INSIDE each training fold, never here."""
    ch = pd.read_csv(CHUNKS_CLEAN, usecols=["collection_id", "clean_text"],
                     dtype={"collection_id": str})
    ch["clean_text"] = ch["clean_text"].fillna("")
    docs = ch.groupby("collection_id")["clean_text"].apply(" ".join)
    vec = CountVectorizer(min_df=TFIDF_MIN_DF, max_features=TFIDF_MAX_FEATURES,
                          token_pattern=r"\S+")
    counts = vec.fit_transform(docs.to_list())
    return docs.index, counts, vec


def build_embeddings() -> tuple[pd.Index, np.ndarray, bool]:
    """Show-level embeddings on the CLEANED corpus. Returns (ids, X, is_clean)."""
    if EMBED_CLEAN.exists() and EMBED_CLEAN_IDS.exists():
        ids = json.loads(EMBED_CLEAN_IDS.read_text())
        return pd.Index(ids, name="collection_id"), np.load(EMBED_CLEAN), True
    # Documented placeholder path: dirty-corpus embeddings, must be rerun.
    warnings.warn("cleaned embeddings missing -- falling back to DIRTY-corpus "
                  "embeddings; this arm must be rerun", RuntimeWarning)
    from .embed_ideology_nonlinear import build_show_level_embeddings
    ids, X = build_show_level_embeddings(500)
    return pd.Index(ids, name="collection_id"), X, False


# ------------------------------------------------------------- estimators ----
class BlockNormalizer(BaseEstimator, TransformerMixin):
    """Standardize columns, then divide each feature BLOCK by sqrt(its width).

    Without this, concatenating a 6-column register block onto a 768-column
    embedding block lets the wide block dominate a single shared ridge penalty
    purely by having more columns, so "topics+embeddings" would be testing
    dimensionality rather than information. Scaling each block to comparable
    total norm makes the combination test what it claims to test. Fit on train
    rows only (it is a step inside the in-fold Pipeline)."""

    def __init__(self, widths: tuple[int, ...] = ()):
        self.widths = widths

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        return self

    def transform(self, X):
        Z = self.scaler_.transform(X)
        out, start = [], 0
        for w in self.widths:
            out.append(Z[:, start:start + w] / np.sqrt(w))
            start += w
        return np.hstack(out) if out else Z


# Deliberately not -1. The outer LOSO loop is already 204 sequential fits and
# this box shares cores with other long jobs; n_jobs=-1 here oversubscribed
# 20 cores to a load average of 35 and tripled a concurrent STM run's
# per-iteration time. Override with --n-jobs when the machine is idle.
N_JOBS = 4


def _grid(pipe, params, cv=INNER_CV_FOLDS, n_jobs=None):
    return GridSearchCV(pipe, params, cv=cv, scoring="r2",
                        n_jobs=N_JOBS if n_jobs is None else n_jobs)


def make_ridge(widths=()):
    def f(Xtr, ytr, Xte):
        pipe = Pipeline([("blk", BlockNormalizer(widths or (Xtr.shape[1],))),
                         ("m", Ridge())])
        g = _grid(pipe, {"m__alpha": RIDGE_ALPHAS})
        g.fit(Xtr, ytr)
        return g.predict(Xte)
    return f


def make_enet(widths=()):
    def f(Xtr, ytr, Xte):
        pipe = Pipeline([("blk", BlockNormalizer(widths or (Xtr.shape[1],))),
                         ("m", ElasticNet(max_iter=20000))])
        g = _grid(pipe, {"m__alpha": [0.001, 0.01, 0.1, 1.0],
                         "m__l1_ratio": [0.1, 0.5, 0.9, 1.0]})
        g.fit(Xtr, ytr)
        return g.predict(Xte)
    return f


def make_rf():
    def f(Xtr, ytr, Xte):
        g = _grid(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
                  {"max_depth": [None, 6], "max_features": ["sqrt", 0.3]})
        g.fit(Xtr, ytr)
        return g.predict(Xte)
    return f


def make_gbm():
    def f(Xtr, ytr, Xte):
        g = _grid(HistGradientBoostingRegressor(random_state=SEED),
                  {"learning_rate": [0.03, 0.1], "max_iter": [200, 400],
                   "max_leaf_nodes": [7, 31]})
        g.fit(Xtr, ytr)
        return g.predict(Xte)
    return f


def make_mlp(widths=()):
    def f(Xtr, ytr, Xte):
        pipe = Pipeline([("blk", BlockNormalizer(widths or (Xtr.shape[1],))),
                         ("m", MLPRegressor(random_state=SEED, max_iter=2000,
                                            early_stopping=True, n_iter_no_change=20))])
        g = _grid(pipe, {"m__hidden_layer_sizes": [(32,), (64, 32)],
                         "m__alpha": [1e-3, 1e-1, 1.0]})
        g.fit(Xtr, ytr)
        return g.predict(Xte)
    return f


def make_tfidf_ridge():
    """Counts in, ridge out. idf is fit on TRAIN ROWS ONLY inside each fold."""
    def f(Xtr, ytr, Xte):
        pipe = Pipeline([("tfidf", TfidfTransformer()),
                         ("scale", StandardScaler(with_mean=False)),
                         ("m", Ridge())])
        g = _grid(pipe, {"m__alpha": RIDGE_ALPHAS})
        g.fit(Xtr, ytr)
        return g.predict(Xte)
    return f


# ------------------------------------------------------------ evaluation -----
def evaluate(X, y, fit_predict, label: str) -> dict:
    """Outer LOSO. Returns outer-loop R2, CI, and PER-SHOW errors for pairing."""
    preds = loso_cv(X, y, fit_predict)
    r2 = r2_from_predictions(y, preds)
    lo, hi = bootstrap_r2_ci(y, preds)
    err = (y - preds) ** 2
    return {"label": label, "n": int(len(y)), "outer_r2": float(r2),
            "ci_95": [float(lo), float(hi)], "sq_err": err.tolist(),
            "preds": preds.tolist()}


def paired_test(base: dict, other: dict) -> dict:
    """Paired comparison on per-show squared errors, same folds.

    'Beats' requires this to be significant. A higher point estimate or a
    non-overlapping CI is NOT sufficient and is not reported as such."""
    b = np.asarray(base["sq_err"])
    o = np.asarray(other["sq_err"])
    d = b - o                      # positive => `other` has smaller error
    try:
        w_stat, w_p = stats.wilcoxon(d)
    except ValueError:             # all-zero differences
        w_stat, w_p = float("nan"), 1.0
    t_stat, t_p = stats.ttest_rel(b, o)
    return {"delta_r2": float(other["outer_r2"] - base["outer_r2"]),
            "mean_sq_err_reduction": float(d.mean()),
            "wilcoxon_p": float(w_p), "paired_t_p": float(t_p),
            "n_shows_improved": int((d > 0).sum()), "n_shows": int(len(d))}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def align(blocks: list[pd.DataFrame], y_series: pd.Series):
    """Inner-join every feature block with the target; identical show order."""
    idx = y_series.index
    for b in blocks:
        idx = idx.intersection(b.index)
    idx = pd.Index(sorted(idx))
    Xs = [b.loc[idx].to_numpy(dtype=float) for b in blocks]
    return idx, Xs, y_series.loc[idx].to_numpy(dtype=float)


def save(name: str, payload: dict) -> Path:
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    p = REPORT_DIR / name
    p.write_text(json.dumps(payload, indent=2))
    print(f"[report] -> {p}")
    return p


# ----------------------------------------------------------------- arm 1 -----
def arm1(targets: dict[str, pd.Series], k: int = K_PRIMARY) -> dict:
    """Estimator comparison, features held fixed at CLR topics.

    Answers: is ridge leaving nonlinear signal on the table, or is the
    representation the bottleneck? Reported as a RANGE. The max is not the
    headline -- picking the best estimator by the same outer R2 used to report
    it would reintroduce exactly the outcome-selection this module exists to
    remove, so the ridge baseline is what carries forward into Arm 2."""
    topics = build_topics(k)
    makers = {"ridge": make_ridge(), "elasticnet": make_enet(),
              "random_forest": make_rf(), "grad_boost": make_gbm(),
              "mlp": make_mlp()}
    out = {"k": k, "arm": 1, "question": "recoverability, not dominance",
           "results": {}}
    for tname, y_series in targets.items():
        idx, (X,), y = align([topics], y_series)
        print(f"\n[arm1] target={tname} n={len(y)} dim={X.shape[1]}")
        res = {}
        for mname, mk in makers.items():
            r = evaluate(X, y, mk, f"{tname}/{mname}")
            res[mname] = r
            print(f"  {mname:<15} outer R2 = {r['outer_r2']:+.3f} "
                  f"[{r['ci_95'][0]:+.3f}, {r['ci_95'][1]:+.3f}]")
        r2s = [v["outer_r2"] for v in res.values()]
        out["results"][tname] = {
            "show_ids": list(idx), "models": res,
            "range": [float(min(r2s)), float(max(r2s))],
            "median": float(np.median(r2s)),
            "ridge_baseline": res["ridge"]["outer_r2"],
            "vs_ridge_paired": {m: paired_test(res["ridge"], res[m])
                                for m in res if m != "ridge"},
        }
    return out


# ----------------------------------------------------------------- arm 2 -----
def arm2(targets: dict[str, pd.Series], k: int = K_PRIMARY) -> dict:
    """Representation comparison at a FIXED estimator (ridge), the
    claim-relevant arm: does adding style/embeddings/register beat topics
    alone? 'Beats' = significant PAIRED test on per-show errors."""
    topics = build_topics(k)
    register = build_register()
    emb_idx, emb_X, emb_clean = build_embeddings()
    embeddings = pd.DataFrame(emb_X, index=emb_idx)
    cnt_idx, counts, _ = build_counts()

    out = {"k": k, "arm": 2, "embeddings_are_cleaned_corpus": bool(emb_clean),
           "question": "does style add BEYOND topic (recoverability framing)",
           "results": {}}
    if not emb_clean:
        out["PLACEHOLDER_WARNING"] = (
            "Embedding rows use DIRTY-corpus embeddings and MUST be rerun once "
            "the cleaned re-embed exists.")

    for tname, y_series in targets.items():
        idx, (Xt, Xe, Xr), y = align([topics, embeddings, register], y_series)
        cnt_pos = [cnt_idx.get_loc(s) for s in idx]
        Xc = counts[cnt_pos]
        dt, de, dr = Xt.shape[1], Xe.shape[1], Xr.shape[1]
        print(f"\n[arm2] target={tname} n={len(y)} "
              f"dims topics={dt} emb={de} reg={dr} tfidf={Xc.shape[1]}")

        specs = {
            "topics_only":              (Xt, make_ridge((dt,))),
            "embeddings_only":          (Xe, make_ridge((de,))),
            "tfidf_only":               (Xc, make_tfidf_ridge()),
            "register_only":            (Xr, make_ridge((dr,))),
            "topics+embeddings":        (np.hstack([Xt, Xe]), make_ridge((dt, de))),
            "topics+register":          (np.hstack([Xt, Xr]), make_ridge((dt, dr))),
            "topics+embeddings+register": (np.hstack([Xt, Xe, Xr]),
                                           make_ridge((dt, de, dr))),
        }
        res = {}
        for name, (X, mk) in specs.items():
            r = evaluate(X, y, mk, f"{tname}/{name}")
            res[name] = r
            print(f"  {name:<28} outer R2 = {r['outer_r2']:+.3f} "
                  f"[{r['ci_95'][0]:+.3f}, {r['ci_95'][1]:+.3f}]")

        base = res["topics_only"]
        comparisons = {n: paired_test(base, res[n]) for n in res if n != "topics_only"}
        names = list(comparisons)
        adj = holm([comparisons[n]["wilcoxon_p"] for n in names])
        for n, a in zip(names, adj):
            comparisons[n]["wilcoxon_p_holm"] = float(a)
            comparisons[n]["beats_topics_only"] = bool(a < 0.05 and
                                                       comparisons[n]["delta_r2"] > 0)
        out["results"][tname] = {"show_ids": list(idx), "models": res,
                                 "paired_vs_topics_only": comparisons}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", type=int, required=True, choices=[1, 2])
    ap.add_argument("--k", type=int, default=K_PRIMARY)
    args = ap.parse_args()

    t = build_targets()
    targets = {"extended_all": t["extended_all"], "host_only": t["host_only"]}
    print(f"[targets] extended_all n={len(targets['extended_all'])}, "
          f"host_only n={len(targets['host_only'])}")

    if args.arm == 1:
        save(f"ceiling_nested_arm1_k{args.k}.json", arm1(targets, args.k))
    else:
        save(f"ceiling_nested_arm2_k{args.k}.json", arm2(targets, args.k))


if __name__ == "__main__":
    main()
