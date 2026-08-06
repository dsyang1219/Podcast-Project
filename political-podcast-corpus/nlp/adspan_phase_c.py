"""Phase C -- re-derive K on the cleaned corpus, refit LDA, and run the
corrected ideology probe.

K=75 was chosen against the CONTAMINATED corpus BY IDEOLOGY R2, so it is not
inherited here. This sweeps K in {30, 50, 75, 100, 150} and selects K on an
OUTCOME-INDEPENDENT criterion.

SELECTION IS BY COHERENCE, NEVER BY IDEOLOGY R2
------------------------------------------------
An earlier version of this script selected K by the ideology probe itself
(smallest K within 0.01 of the best R2). That picks the topic representation
using the same outcome later reported as the finding -- mild circularity that
biases the headline upward. It is replaced by:

  Rule (fixed before any result is inspected): the chosen K is the SMALLEST K
  whose c_v coherence is within ONE STANDARD ERROR of the peak c_v across the
  sweep. The SE is computed across per-topic c_v scores at the peak K
  (std / sqrt(K)), so "indistinguishable from the best" is measured, not
  asserted. If per-topic scores are unavailable, the documented fallback is
  smallest K within 1% of peak c_v; which rule fired is recorded in the output.

Smallest-in-plateau avoids crediting a larger K that only looks best by noise.

Ideology R2 is MEASURED at the selected K and reported across every swept K.
It never participates in choosing K. That separation is the point of the
change and is enforced structurally: choose_k_by_coherence() is not given
access to the ideology results at all.

c_v IS THE SELECTION CRITERION, NOT A RESULT. c_v has known instabilities
(sliding-window + indirect cosine confirmation can reward degenerate topics),
so c_npmi and u_mass are computed as cross-checks. If c_v and c_npmi disagree
sharply about the best-K region, the disagreement is flagged and the more
conservative (smaller) K is preferred.

    .venv/bin/python -m nlp.adspan_phase_c --export-tokens
    .venv/bin/python -m nlp.adspan_phase_c --sweep   [--corpus preview|clean]
    .venv/bin/python -m nlp.adspan_phase_c --report  [--corpus preview|clean]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tomotopy as tp
from tomotopy.coherence import Coherence

from pipeline import config as pipeline_config
from . import preprocess_regimes as pp
from .adspan_phase_b import OUT_DIR
from .adspan_rechunk import OUT_PATH as CHUNKS_PATH
from .embed_ideology_nonlinear import (
    bootstrap_r2_ci,
    loso_cv,
    r2_from_predictions,
    ridge_fit_predict,
)
from .lda_ideology_clr import clr
from .lda_regime_refit import (
    N_ITER,
    PROFANITY,
    SEED,
    TOMOTOPY_WORKERS,
    load_spoken_stopwords,
)

K_GRID = [30, 50, 75, 90, 100, 125, 150]
BASELINE_K = 75          # the OLD, ideology-selected K on the contaminated corpus
REGIME = "moderate"      # base regime per the task brief

# c_v is primary (field standard); c_npmi and u_mass are cross-checks against a
# c_v artifact. u_mass runs on a different scale (negative, larger = better) and
# is reported for completeness rather than used for selection.
COHERENCE_METRICS = ["c_v", "c_npmi", "u_mass"]
FALLBACK_REL_TOL = 0.01  # documented fallback if per-topic SE is unavailable

# Two corpora this can run on. `clean` is the deliverable; `preview` runs the
# identical machinery on the pre-excision passages so the pipeline can be
# exercised before Phase B lands -- its ideology R2 is still contamination-
# inflated and every output says so.
CORPORA = {
    "clean": {
        "tokens": OUT_DIR / f"tokens_{REGIME}_adclean.csv",
        "label": "ad-span-cleaned corpus",
        "is_preview": False,
    },
    "preview": {
        "tokens": pipeline_config.OUTPUT_DIR / "regimes" / f"tokens_{REGIME}.csv",
        "label": "PREVIEW -- pre-ad-clean corpus (contamination-inflated; "
                 "rerun on ad-cleaned corpus required)",
        "is_preview": True,
    },
}


def paths_for(corpus: str) -> dict:
    c = dict(CORPORA[corpus])
    suffix = "" if corpus == "clean" else f"_{corpus}"
    c["sweep"] = OUT_DIR / f"phase_c_ksweep{suffix}.json"
    c["report"] = OUT_DIR / f"phase_c_report{suffix}.json"
    c["topics_csv"] = OUT_DIR / f"ideology_topics_adclean{suffix}.csv"
    c["doctopic_fmt"] = str(OUT_DIR / ("doctopic_adclean" + suffix + "_k{k}.csv"))
    return c


TOKENS_PATH = CORPORA["clean"]["tokens"]

# Lexical markers for auto-flagging any residual ad/meta topic in the refit.
AD_WORDS = {
    "com", "code", "promo", "sponsor", "sponsored", "discount", "offer", "offers",
    "shipping", "slash", "dot", "checkout", "deal", "deals", "save", "off",
    "visit", "trial", "free", "order", "purchase", "customers", "brand",
}
META_WORDS = {
    "podcast", "podcasts", "subscribe", "listen", "listening", "episode",
    "episodes", "welcome", "thanks", "review", "reviews", "producer",
    "produced", "host", "show", "channel", "youtube", "apple", "spotify",
    "rate", "download", "newsletter",
}


# ------------------------------------------------------------ tokenize ----
def export_tokens() -> None:
    """Apply the SAME moderate-regime toggles used for the 0.451 baseline to
    the ad-cleaned passages. Toggles come from nlp/preprocess_regimes.py --
    not re-specified here -- so the regime is identical by construction."""
    df = pd.read_csv(CHUNKS_PATH, usecols=["chunk_id", "collection_id", "text"],
                      dtype={"chunk_id": str, "collection_id": str})
    df["text"] = df["text"].fillna("")
    print(f"[tokens] {len(df)} ad-cleaned passages from {CHUNKS_PATH.name}")

    toggles = pp.regime_toggles(REGIME)
    token_lists = pp.build_token_lists(df["text"].tolist(), toggles,
                                        n_process=1, verbose=True)
    pruned, acc = pp.apply_docfreq_prune(token_lists, toggles, verbose=True)

    out = pd.DataFrame({
        "chunk_id": df["chunk_id"],
        "collection_id": df["collection_id"],
        "clean_text": [" ".join(t) for t in pruned],
    })
    out.to_csv(TOKENS_PATH, index=False)
    n_empty = int((out["clean_text"] == "").sum())
    print(f"[tokens] -> {TOKENS_PATH} ({len(out)} rows, {n_empty} empty)")
    acc.update({"regime": REGIME, "n_empty_docs": n_empty,
                "toggles": vars(toggles), "source": str(CHUNKS_PATH)})
    (OUT_DIR / f"vocab_accounting_{REGIME}_adclean.json").write_text(
        json.dumps(acc, indent=2, default=str))
    print(f"[tokens] vocab {acc['vocab_before']} -> {acc['vocab_after']}, "
          f"tokens {acc['tokens_before']} -> {acc['tokens_after']}")


# --------------------------------------------------------------- probe ----
def load_targets() -> pd.DataFrame:
    return pd.read_csv(pipeline_config.OUTPUT_DIR / "ideology_targets_204.csv",
                        dtype={"show_id": str}).set_index("show_id")


def probe(doc_topic: np.ndarray, collection_ids: np.ndarray,
           targets: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    """Identical harness to nlp/lda_regime_refit.probe_ideology: show-level
    mean of chunk doc-topic proportions, CLR, Ridge with alpha chosen inside
    each LOSO fold, plus a permutation null."""
    cols = [f"T{i}" for i in range(doc_topic.shape[1])]
    df = pd.DataFrame(doc_topic, columns=cols)
    df["collection_id"] = collection_ids
    show_topics = df.groupby("collection_id")[cols].mean()

    joined = show_topics.join(targets, how="inner").dropna(subset=["ideology_primary"])
    X = clr(np.where(joined[cols].to_numpy(dtype=float) <= 0, 1e-12,
                      joined[cols].to_numpy(dtype=float)))
    y = joined["ideology_primary"].to_numpy(dtype=float)

    preds = loso_cv(X, y, ridge_fit_predict)
    cv_r2 = r2_from_predictions(y, preds)
    lo, hi = bootstrap_r2_ci(y, preds)

    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    perm_r2 = r2_from_predictions(y_shuf, loso_cv(X, y_shuf, ridge_fit_predict))

    return ({"n_shows": int(len(y)), "loso_cv_r2": float(cv_r2),
             "ci_95": [float(lo), float(hi)], "permutation_r2": float(perm_r2)},
            X, y, cols)


def fit_lda_with_coherences(token_lists: list[list[str]], k: int):
    """Fit LDA exactly as nlp/lda_regime_refit.fit_lda does (same TermWeight,
    seed, iteration count and worker count -- so the only thing that varies
    across this sweep is K), then compute every coherence metric plus the
    PER-TOPIC c_v scores the selection rule's standard error needs."""
    mdl = tp.LDAModel(k=k, seed=SEED, tw=tp.TermWeight.ONE)
    for toks in token_lists:
        mdl.add_doc(toks if toks else ["_empty_doc_placeholder_"])
    mdl.train(0, workers=TOMOTOPY_WORKERS)
    for _ in range(0, N_ITER, 20):
        mdl.train(20, workers=TOMOTOPY_WORKERS)

    coh: dict[str, float] = {}
    per_topic_cv: list[float] = []
    for metric in COHERENCE_METRICS:
        try:
            c = Coherence(mdl, coherence=metric, top_n=10)
            coh[metric] = float(c.get_score())
            if metric == "c_v":
                per_topic_cv = [float(c.get_score(topic_id=t)) for t in range(k)]
        except Exception as e:  # a metric failing must not lose the whole fit
            print(f"  [warn] coherence {metric} failed at K={k}: {e}")
            coh[metric] = float("nan")

    doc_topic = np.array([d.get_topic_dist() for d in mdl.docs])
    return mdl, coh, per_topic_cv, doc_topic


def choose_k_by_coherence(results: list[dict]) -> dict:
    """Select K from coherence ONLY.

    Deliberately takes just the coherence fields -- it is never handed the
    ideology results, so the outcome cannot leak into the choice even by
    accident. Rule is fixed in the module docstring, not derived from what the
    numbers turn out to look like.
    """
    cv = {r["k"]: r["coherence"]["c_v"] for r in results}
    peak_k = max(cv, key=lambda k: cv[k])
    peak = cv[peak_k]

    peak_per_topic = next((r["per_topic_cv"] for r in results if r["k"] == peak_k), [])
    if peak_per_topic and len(peak_per_topic) > 1:
        se = float(np.std(peak_per_topic, ddof=1) / np.sqrt(len(peak_per_topic)))
        threshold = peak - se
        rule = (f"smallest K with c_v >= peak - 1SE  (peak={peak:.4f} at K={peak_k}, "
                f"SE={se:.4f} over {len(peak_per_topic)} per-topic scores, "
                f"threshold={threshold:.4f})")
    else:
        se = float("nan")
        threshold = peak * (1 - FALLBACK_REL_TOL)
        rule = (f"FALLBACK: smallest K within {FALLBACK_REL_TOL:.0%} of peak c_v "
                f"(peak={peak:.4f} at K={peak_k}, threshold={threshold:.4f}) "
                f"-- per-topic c_v unavailable")

    chosen = min(k for k in sorted(cv) if cv[k] >= threshold)

    # Cross-check against the secondary metric. Disagreement is reported, and
    # resolved toward the smaller K, per the module docstring.
    npmi = {r["k"]: r["coherence"].get("c_npmi", float("nan")) for r in results}
    warn = ""
    if not all(np.isnan(v) for v in npmi.values()):
        npmi_peak_k = max(npmi, key=lambda k: (npmi[k] if not np.isnan(npmi[k]) else -np.inf))
        if npmi_peak_k != peak_k:
            warn = (f"c_v peaks at K={peak_k} but c_npmi peaks at K={npmi_peak_k}; "
                    f"metrics disagree on the best-K region -- preferring the "
                    f"smaller K ({min(peak_k, npmi_peak_k)}) per the pre-fixed rule")
            if npmi_peak_k < chosen:
                chosen = min(chosen, npmi_peak_k)
    if chosen == max(cv):
        warn = (warn + " | " if warn else "") + (
            "WARNING: rule selected the LARGEST K swept -- no coherence plateau "
            "reached; extend the sweep before trusting this K")

    return {"chosen_k": chosen, "rule": rule, "peak_k": peak_k, "peak_cv": peak,
            "se": se, "threshold": threshold, "warning": warn,
            "cv_by_k": cv, "npmi_by_k": npmi}


def classify_topic(top_words: list[str], filler_words: set[str]) -> dict:
    n_filler = sum(1 for w in top_words if w in filler_words or w in PROFANITY)
    n_ad = sum(1 for w in top_words if w in AD_WORDS)
    n_meta = sum(1 for w in top_words if w in META_WORDS)
    kinds = []
    if n_filler >= 5:
        kinds.append("filler")
    if n_ad >= 3:
        kinds.append("ad")
    if n_meta >= 4:
        kinds.append("meta")
    return {"n_filler": n_filler, "n_ad": n_ad, "n_meta": n_meta,
            "flagged_as": kinds, "is_contaminant": bool(kinds)}


# --------------------------------------------------------------- sweep ----
def _k_path(P: dict, k: int) -> Path:
    return P["sweep"].with_name(P["sweep"].stem + f"_k{k}.json")


def sweep(corpus: str, only_k: int | None = None) -> None:
    """Fit the K grid, or a single K when `only_k` is given.

    Each K writes its own result file, so the five fits can run as independent
    concurrent processes and be merged afterwards. Sequentially the sweep used
    only TOMOTOPY_WORKERS (8) of this machine's 20 cores and left 60% idle;
    running the K values concurrently saturates the box WITHOUT changing
    `workers`, which matters because tomotopy's result depends on the worker
    count (see its RuntimeWarning) and workers=8 is what the on-record 0.451
    baseline used. Speed comes from filling idle cores, not from altering the fit.
    """
    P = paths_for(corpus)
    ks = [only_k] if only_k is not None else list(K_GRID)
    todo = [k for k in ks if not _k_path(P, k).exists()]
    if not todo:
        print(f"[sweep] K={ks} already fit; nothing to do")
        merge(corpus)
        return

    df = pd.read_csv(P["tokens"], dtype={"collection_id": str})
    df["clean_text"] = df["clean_text"].fillna("")
    token_lists = [t.split() for t in df["clean_text"]]
    targets = load_targets()
    filler_words = load_spoken_stopwords()
    print(f"[sweep] CORPUS: {P['label']}")
    print(f"[sweep] {len(token_lists)} passages from {P['tokens'].name}, "
          f"mean {np.mean([len(t) for t in token_lists]):.1f} tokens/passage")
    print(f"[sweep] fitting K={todo}")

    for k in todo:
        print(f"\n{'=' * 70}\n[sweep] fitting K={k}  ({P['label']})\n{'=' * 70}", flush=True)
        t0 = datetime.now(timezone.utc)
        mdl, coh, per_topic_cv, doc_topic = fit_lda_with_coherences(token_lists, k)
        # Ideology is MEASURED here and stored, but plays no part in selecting K.
        ideo, _, _, _ = probe(doc_topic, df["collection_id"].to_numpy(), targets)

        dt = pd.DataFrame(doc_topic, columns=[f"T{i}" for i in range(k)])
        dt.insert(0, "collection_id", df["collection_id"].to_numpy())
        dt.insert(0, "chunk_id", df["chunk_id"].to_numpy())
        dt.to_csv(P["doctopic_fmt"].format(k=k), index=False)

        topics = []
        for t in range(k):
            tw = [w for w, _ in mdl.get_topic_words(t, top_n=10)]
            topics.append({"topic": t, "top_words": tw,
                           **classify_topic(tw, filler_words)})
        n_contam = sum(1 for t in topics if t["is_contaminant"])

        res = {"k": k, "coherence": coh, "per_topic_cv": per_topic_cv,
                "coherence_cv": coh.get("c_v"),  # back-compat field
                "vocab_size": len(mdl.used_vocabs),
                "ideology": ideo, "n_contaminant_topics": n_contam,
                "topics": topics,
                "fit_seconds": (datetime.now(timezone.utc) - t0).total_seconds(),
                "tomotopy_workers": TOMOTOPY_WORKERS, "n_iter": N_ITER, "seed": SEED}
        _k_path(P, k).write_text(json.dumps(res, indent=2, default=str))
        print(f"[sweep] K={k}: c_v={coh.get('c_v', float('nan')):.4f} "
              f"c_npmi={coh.get('c_npmi', float('nan')):.4f} "
              f"u_mass={coh.get('u_mass', float('nan')):.4f} | "
              f"[measured, not used for selection] R2={ideo['loso_cv_r2']:.4f} "
              f"| contaminant={n_contam}/{k} | {res['fit_seconds'] / 60:.1f} min", flush=True)

    merge(corpus)


def merge(corpus: str) -> None:
    """Combine the per-K result files into the sweep file `report` reads."""
    P = paths_for(corpus)
    results = []
    for k in K_GRID:
        p = _k_path(P, k)
        if p.exists():
            results.append(json.loads(p.read_text()))
    if not results:
        raise SystemExit(f"no per-K results found for corpus={corpus}")
    P["sweep"].write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(),
         "corpus": corpus, "corpus_label": P["label"],
         "is_preview": P["is_preview"],
         "regime": REGIME, "k_grid": K_GRID,
         "selection_criterion": "coherence (c_v); ideology R2 never selects K",
         "results": results}, indent=2, default=str))
    print(f"[merge] {len(results)}/{len(K_GRID)} K values -> {P['sweep']}")
    if len(results) < len(K_GRID):
        missing = [k for k in K_GRID if not _k_path(P, k).exists()]
        print(f"[merge] WARNING: missing K={missing} -- selection rule will run "
              f"over an incomplete grid")


# -------------------------------------------------------------- report ----
def report(corpus: str) -> None:
    P = paths_for(corpus)
    sw = json.loads(P["sweep"].read_text())
    results = sw["results"]
    by_k = {r["k"]: r for r in results}

    banner = "!" * 78
    if P["is_preview"]:
        print(f"\n{banner}\n!! {P['label']}\n"
              f"!! Ideology R2 below is contamination-inflated. NOT the final number.\n{banner}")
    else:
        print(f"\n[report] CORPUS: {P['label']}")

    # ---- STEP 1+2: coherence table and the coherence-only selection --------
    sel = choose_k_by_coherence(results)
    k_star = sel["chosen_k"]

    print(f"\n[report] STEP 1 -- COHERENCE TABLE (the selection criterion)")
    print(f"{'K':<7}{'c_v':<12}{'c_npmi':<12}{'u_mass':<12}{'vocab':<9}")
    for r in results:
        c = r["coherence"]
        star = "  <- selected" if r["k"] == k_star else ""
        print(f"{r['k']:<7}{c.get('c_v', float('nan')):<12.4f}"
              f"{c.get('c_npmi', float('nan')):<12.4f}"
              f"{c.get('u_mass', float('nan')):<12.4f}{r['vocab_size']:<9}{star}")

    print(f"\n[report] STEP 2 -- SELECTED K = {k_star}")
    print(f"  rule: {sel['rule']}")
    if sel["warning"]:
        print(f"  NOTE: {sel['warning']}")

    # ---- STEP 4: ideology across ALL K, measured after selection -----------
    print(f"\n[report] STEP 3+4 -- IDEOLOGY R2 (measured at every K; never used to select)")
    print(f"{'K':<7}{'R2':<10}{'95% CI':<22}{'perm':<10}{'contam topics':<16}")
    for r in results:
        i = r["ideology"]
        ci = f"[{i['ci_95'][0]:.3f},{i['ci_95'][1]:.3f}]"
        tag = ""
        if r["k"] == k_star:
            tag = "  <- coherence-selected"
        elif r["k"] == BASELINE_K:
            tag = "  <- OLD ideology-selected K"
        print(f"{r['k']:<7}{i['loso_cv_r2']:<10.4f}{ci:<22}{i['permutation_r2']:<10.4f}"
              f"{r['n_contaminant_topics']}/{r['k']:<12}{tag}")

    r2s = [r["ideology"]["loso_cv_r2"] for r in results]
    ideology_best_k = max(results, key=lambda r: r["ideology"]["loso_cv_r2"])["k"]
    agreement = (
        f"coherence-selected K={k_star} MATCHES the ideology-optimal K"
        if ideology_best_k == k_star else
        f"coherence selects K={k_star} but ideology R2 peaks at K={ideology_best_k} "
        f"-- informative about how ideology relates to topic granularity, not a defect"
    )
    print(f"\n[report] STEP 5 -- coherence vs ideology: {agreement}")
    print(f"[report] robustness: ideology R2 ranges {min(r2s):.4f}-{max(r2s):.4f} "
          f"across K in {K_GRID}")

    chosen = by_k[k_star]
    baseline = by_k.get(BASELINE_K)
    if baseline:
        print(f"[report] reference: K={BASELINE_K} (the PREVIOUSLY USED, "
              f"OUTCOME-SELECTED K) R2 = {baseline['ideology']['loso_cv_r2']:.4f}")

    # ------- ideology-carrying topics at the chosen K, on a full-data fit --
    dt = pd.read_csv(P["doctopic_fmt"].format(k=k_star),
                      dtype={"collection_id": str, "chunk_id": str})
    targets = load_targets()
    cols = [f"T{i}" for i in range(k_star)]
    show_topics = dt.groupby("collection_id")[cols].mean()
    joined = show_topics.join(targets, how="inner").dropna(subset=["ideology_primary"])
    X = clr(np.where(joined[cols].to_numpy(dtype=float) <= 0, 1e-12,
                      joined[cols].to_numpy(dtype=float)))
    y = joined["ideology_primary"].to_numpy(dtype=float)

    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=[1.0, 10.0, 100.0, 1e3, 1e4]))
    pipe.fit(X, y)
    coefs = pipe[-1].coef_

    tmap = {t["topic"]: t for t in chosen["topics"]}
    rows = []
    for i, c in enumerate(coefs):
        t = tmap[i]
        rows.append({"topic": i, "ridge_coef": float(c), "abs_coef": abs(float(c)),
                     "direction": "conservative(+)" if c > 0 else "liberal(-)",
                     "flagged_as": ",".join(t["flagged_as"]),
                     "top_words": " ".join(t["top_words"])})
    tbl = pd.DataFrame(rows).sort_values("abs_coef", ascending=False)
    tbl.insert(0, "corpus", corpus)
    tbl.to_csv(P["topics_csv"], index=False)

    print(f"\n[report] top 20 ideology-carrying topics at K={k_star} "
          f"(full-data RidgeCV coefficients on CLR show-level topic shares):")
    for _, r in tbl.head(20).iterrows():
        flag = f" [{r['flagged_as']}]" if r["flagged_as"] else ""
        print(f"  T{int(r['topic']):<4} {r['ridge_coef']:+.4f} "
              f"{r['direction']:<18}{flag} {r['top_words']}")

    contam = [t for t in chosen["topics"] if t["is_contaminant"]]
    print(f"\n[report] residual ad/meta/filler topics at K={k_star}: "
          f"{len(contam)}/{k_star}")
    for t in contam:
        print(f"  T{t['topic']}: {','.join(t['flagged_as'])} -> {t['top_words']}")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus, "corpus_label": P["label"], "is_preview": P["is_preview"],
        "regime": REGIME, "k_grid": K_GRID,
        "selection": {
            "criterion": "topic coherence (c_v), outcome-independent",
            "rule": sel["rule"],
            "chosen_k": k_star,
            "peak_cv_k": sel["peak_k"], "peak_cv": sel["peak_cv"],
            "se": sel["se"], "threshold": sel["threshold"],
            "warning": sel["warning"],
            "note": "ideology R2 played no role in selecting K; it is measured "
                    "at every K and reported after selection",
        },
        "coherence_table": [{"k": r["k"], **r["coherence"],
                              "vocab_size": r["vocab_size"]} for r in results],
        "ideology_across_k": [{"k": r["k"], "r2": r["ideology"]["loso_cv_r2"],
                                "ci_95": r["ideology"]["ci_95"],
                                "permutation_r2": r["ideology"]["permutation_r2"],
                                "n_contaminant_topics": r["n_contaminant_topics"]}
                               for r in results],
        "ideology_r2_range_across_k": [min(r2s), max(r2s)],
        "headline_at_coherence_selected_k": chosen["ideology"],
        "reference_old_ideology_selected_k75": (
            {**baseline["ideology"],
             "note": "the PREVIOUSLY USED K, selected on ideology R2 "
                     "(outcome-selected -- reported for comparison only)"}
            if baseline else None),
        "coherence_vs_ideology_k": {
            "coherence_selected_k": k_star,
            "ideology_optimal_k": ideology_best_k,
            "agree": ideology_best_k == k_star,
            "note": agreement,
        },
        "n_contaminant_topics_chosen_k": len(contam),
        "contaminant_topics_chosen_k": contam,
        "before_after_chain": {
            "moderate_raw_contaminated_k75_ideology_selected": 0.4510,
            "drop_ads_meta_topics_k75": 0.3953,
            "drop_all_flagged_topics_k75": 0.3756,
            "this_run_at_coherence_selected_k": chosen["ideology"]["loso_cv_r2"],
            "this_run_at_k75": (baseline["ideology"]["loso_cv_r2"] if baseline else None),
            "corpus": corpus,
            "valid_as_final": not P["is_preview"],
        },
    }
    P["report"].write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[report] -> {P['report']}")
    print(f"[report] ideology topic table -> {P['topics_csv']}")

    ch = out["before_after_chain"]
    print(f"\n[report] BEFORE/AFTER CHAIN  ({P['label']})")
    print(f"  moderate raw, contaminated   K=75 (ideology-selected)  R2 = "
          f"{ch['moderate_raw_contaminated_k75_ideology_selected']:.4f}")
    print(f"  drop ad/meta topics post hoc K=75                      R2 = "
          f"{ch['drop_ads_meta_topics_k75']:.4f}")
    print(f"  drop all flagged topics      K=75                      R2 = "
          f"{ch['drop_all_flagged_topics_k75']:.4f}")
    if ch["this_run_at_k75"] is not None:
        print(f"  this run                     K=75                      R2 = "
              f"{ch['this_run_at_k75']:.4f}")
    print(f"  this run  K={k_star} (coherence-selected)             R2 = "
          f"{ch['this_run_at_coherence_selected_k']:.4f}"
          f"{'   <- headline' if not P['is_preview'] else '   <- PREVIEW ONLY'}")
    print(f"\n[report] HEADLINE: ideology R2 is {min(r2s):.3f}-{max(r2s):.3f} across "
          f"K in {K_GRID}, and {chosen['ideology']['loso_cv_r2']:.3f} "
          f"[{chosen['ideology']['ci_95'][0]:.3f},{chosen['ideology']['ci_95'][1]:.3f}] "
          f"at the coherence-selected K={k_star}.")
    if P["is_preview"]:
        print(f"\n{banner}\n!! PREVIEW ONLY -- rerun on the ad-cleaned corpus "
              f"(--corpus clean) before reporting.\n{banner}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-tokens", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--merge", action="store_true",
                     help="combine per-K result files into the sweep file")
    ap.add_argument("--only-k", type=int, default=None,
                     help="fit a single K (used to run the grid as concurrent processes)")
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="clean",
                     help="'clean' = ad-span-excised corpus (the deliverable); "
                          "'preview' = pre-excision passages, labelled preview")
    args = ap.parse_args()
    if args.export_tokens:
        export_tokens()
    if args.sweep:
        sweep(args.corpus, only_k=args.only_k)
    if args.merge:
        merge(args.corpus)
    if args.report:
        report(args.corpus)
    if not any((args.export_tokens, args.sweep, args.report, args.merge)):
        ap.error("pass --export-tokens, --sweep, --merge, and/or --report")


if __name__ == "__main__":
    main()
