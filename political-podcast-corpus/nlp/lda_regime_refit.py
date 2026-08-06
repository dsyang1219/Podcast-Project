"""Part B substantive checks (advisor task: cleaner topics via better
preprocessing + Denny & Spirling sensitivity check).

For each curated preprocessing regime (minimal / moderate / aggressive /
aggressive_pos, produced by r/preprocess.R and exported as a per-regime
token CSV under data/output/regimes/), this script:

  1. Refits K=75 LDA with tomotopy -- SAME settings as the existing baseline
     fit (nlp/lda_ideology.py's fit_lda: TermWeight.ONE, seed=0, 500 Gibbs
     iterations, c_v coherence @top_n=10) so the only thing that changes
     across regimes is the input token stream. No re-sweep of K here (task
     scope allows "or re-sweep K briefly" but the headline comparison holds
     K fixed at 75 to isolate the preprocessing effect specifically).
  2. Flags "filler-like" topics: a topic is filler-like if >= FILLER_TOPIC_
     THRESHOLD of its top-10 words are in data/spoken_stopwords.txt (the
     same auditable list used to build the regimes) OR are common curse
     words (kept as a small inline list here since profanity is a register
     marker distinct from filler and the original T46 finding was
     profanity-driven, not spoken-stopword-driven).
  3. Aggregates chunk-level doc-topic proportions to show level (mean),
     CLR-transforms (reuses nlp/lda_ideology_clr.py's clr()), joins
     ideology_targets_204.csv on ideology_primary (matches lda_ideology_204.
     py's "extended_all" condition -- the full-coverage headline target),
     and runs the SAME Ridge / LOSO-CV / permutation-null harness as the
     rest of this project's ideology probes (embed_ideology_nonlinear.py).
     R2 is reported as-is, never selected/tuned -- see module docstring
     honesty note in the task brief this implements.

Usage:
    .venv/bin/python -m nlp.lda_regime_refit --regime moderate
    .venv/bin/python -m nlp.lda_regime_refit --all   # all 4 + baseline reuse
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tomotopy as tp
from tomotopy.coherence import Coherence

from pipeline import config as pipeline_config
from .embed_ideology_nonlinear import bootstrap_r2_ci, loso_cv, r2_from_predictions, ridge_fit_predict
from .lda_ideology_clr import clr

K = 75
SEED = 0
N_ITER = 500
TOMOTOPY_WORKERS = 8  # capped below all-cores(20) to reduce per-thread memory overhead
FILLER_TOPIC_THRESHOLD = 0.5   # >=5 of top-10 words on the spoken-filler/profanity list

PROFANITY = {"fucking", "fuck", "shit", "shitty", "damn", "hell", "ass", "bitch", "goddamn"}

REGIMES_DIR = Path("data/output/regimes")
STOPWORD_FILE = Path("data/spoken_stopwords.txt")


def load_spoken_stopwords() -> set[str]:
    words = set()
    for line in STOPWORD_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.add(line.replace("-", ""))
        words.add(line)
    return words


def load_regime_tokens(regime: str) -> pd.DataFrame:
    path = REGIMES_DIR / f"tokens_{regime}.csv"
    df = pd.read_csv(path, dtype={"collection_id": str})
    df["clean_text"] = df["clean_text"].fillna("")
    return df


def fit_lda(token_lists: list[list[str]], k: int, seed: int = SEED):
    mdl = tp.LDAModel(k=k, seed=seed, tw=tp.TermWeight.ONE)
    for toks in token_lists:
        if toks:
            mdl.add_doc(toks)
        else:
            mdl.add_doc(["_empty_doc_placeholder_"])
    mdl.train(0, workers=TOMOTOPY_WORKERS)
    for i in range(0, N_ITER, 20):
        mdl.train(20, workers=TOMOTOPY_WORKERS)
    coh = Coherence(mdl, coherence="c_v", top_n=10).get_score()
    doc_topic = np.array([d.get_topic_dist() for d in mdl.docs])
    return mdl, coh, doc_topic


def detect_filler_topics(mdl, filler_words: set[str], top_n: int = 10) -> list[dict]:
    results = []
    for t in range(mdl.k):
        top_words = [w for w, _ in mdl.get_topic_words(t, top_n=top_n)]
        n_filler = sum(1 for w in top_words if w in filler_words or w in PROFANITY)
        n_profanity = sum(1 for w in top_words if w in PROFANITY)
        frac = n_filler / top_n
        results.append({
            "topic": t, "top_words": top_words,
            "n_filler_of_top10": n_filler, "n_profanity_of_top10": n_profanity,
            "frac_filler": frac,
            "is_filler_like": frac >= FILLER_TOPIC_THRESHOLD,
        })
    return results


def probe_ideology(doc_topic: np.ndarray, collection_ids: np.ndarray, targets: pd.DataFrame) -> dict:
    cols = [f"T{i}" for i in range(doc_topic.shape[1])]
    df = pd.DataFrame(doc_topic, columns=cols)
    df["collection_id"] = collection_ids
    show_topics = df.groupby("collection_id")[cols].mean()

    joined = show_topics.join(targets, how="inner").dropna(subset=["ideology_primary"])
    X_raw = joined[cols].to_numpy(dtype=float)
    eps = 1e-12
    X_raw = np.where(X_raw <= 0, eps, X_raw)
    X = clr(X_raw)
    y = joined["ideology_primary"].to_numpy(dtype=float)
    n = len(y)

    preds = loso_cv(X, y, ridge_fit_predict)
    cv_r2 = r2_from_predictions(y, preds)
    lo, hi = bootstrap_r2_ci(y, preds)

    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    perm_preds = loso_cv(X, y_shuf, ridge_fit_predict)
    perm_r2 = r2_from_predictions(y_shuf, perm_preds)

    return {
        "n_shows": int(n), "loso_cv_r2": float(cv_r2), "ci_95": [float(lo), float(hi)],
        "permutation_r2": float(perm_r2),
    }


def run_regime(regime: str, targets: pd.DataFrame, filler_words: set[str]) -> dict:
    print(f"\n{'=' * 70}\n[regime: {regime}] loading tokens\n{'=' * 70}")
    df = load_regime_tokens(regime)
    token_lists = [t.split() for t in df["clean_text"]]
    n_empty = sum(1 for t in token_lists if not t)
    print(f"[{regime}] N docs={len(token_lists)}  empty docs={n_empty}  "
          f"mean tokens/doc={np.mean([len(t) for t in token_lists]):.1f}")

    mdl, coh, doc_topic = fit_lda(token_lists, k=K)
    print(f"[{regime}] fit done. c_v coherence={coh:.4f}  vocab size={len(mdl.used_vocabs)}")

    # Persist the full chunk-level doc-topic matrix -- NOT saved by the
    # original comparison run (only aggregate stats + top words were kept),
    # which meant downstream tasks (e.g. contamination dropout probes) had
    # no way to "reuse existing profiles" without a silent re-fit. Matches
    # the existing project convention (data/output/lda_doctopic_k75_500.csv)
    # so it can be loaded the same way.
    doctopic_cols = [f"T{i}" for i in range(mdl.k)]
    doctopic_df = pd.DataFrame(doc_topic, columns=doctopic_cols)
    doctopic_df.insert(0, "collection_id", df["collection_id"].to_numpy())
    doctopic_df.insert(0, "chunk_id", df["chunk_id"].to_numpy())
    doctopic_path = REGIMES_DIR / f"lda_doctopic_{regime}_k{K}.csv"
    doctopic_df.to_csv(doctopic_path, index=False)
    print(f"[{regime}] doc-topic matrix -> {doctopic_path} ({len(doctopic_df)} rows)")

    filler_results = detect_filler_topics(mdl, filler_words)
    n_filler_topics = sum(1 for r in filler_results if r["is_filler_like"])
    print(f"[{regime}] filler-like topics: {n_filler_topics} / {mdl.k}")
    for r in filler_results:
        if r["is_filler_like"]:
            print(f"    T{r['topic']}: {r['top_words']}  (filler={r['n_filler_of_top10']}/10)")

    ideo = probe_ideology(doc_topic, df["collection_id"].to_numpy(), targets)
    print(f"[{regime}] ideology LOSO-CV R2={ideo['loso_cv_r2']:.4f}  "
          f"95%CI=[{ideo['ci_95'][0]:.4f},{ideo['ci_95'][1]:.4f}]  perm={ideo['permutation_r2']:.4f}")

    return {
        "regime": regime,
        "n_docs": len(token_lists),
        "n_empty_docs": n_empty,
        "vocab_size": len(mdl.used_vocabs),
        "coherence_cv": float(coh),
        "n_filler_topics": n_filler_topics,
        "k": mdl.k,
        "filler_topic_detail": [r for r in filler_results if r["is_filler_like"]],
        "all_topics_top_words": {r["topic"]: r["top_words"] for r in filler_results},
        "ideology_probe": ideo,
    }


def run(regimes: list[str]) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    targets = pd.read_csv(OUT / "ideology_targets_204.csv", dtype={"show_id": str}).set_index("show_id")
    filler_words = load_spoken_stopwords()

    results = []
    for regime in regimes:
        results.append(run_regime(regime, targets, filler_words))

    # Merge with any existing report -- lets each regime run as its own
    # subprocess (see r/run_regimes_capped.sh, which caps each regime's
    # memory independently) without later regimes clobbering earlier ones'
    # results in the combined file.
    out_path = REGIMES_DIR / "regime_comparison_report.json"
    existing_by_regime = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            for r in prev.get("regimes", []):
                existing_by_regime[r["regime"]] = r
        except (json.JSONDecodeError, KeyError):
            pass
    for r in results:
        existing_by_regime[r["regime"]] = r

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "k": K,
        "n_iter": N_ITER,
        "filler_topic_threshold": FILLER_TOPIC_THRESHOLD,
        "regimes": list(existing_by_regime.values()),
    }
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {out_path}")

    print(f"\n{'regime':<16}{'n_filler_topics':<18}{'ideology_R2':<14}{'95% CI':<20}{'coherence_cv':<14}{'vocab':<8}")
    for r in results:
        ci = f"[{r['ideology_probe']['ci_95'][0]:.3f},{r['ideology_probe']['ci_95'][1]:.3f}]"
        print(f"{r['regime']:<16}{r['n_filler_topics']:<18}{r['ideology_probe']['loso_cv_r2']:<14.3f}"
              f"{ci:<20}{r['coherence_cv']:<14.3f}{r['vocab_size']:<8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        run(["minimal", "moderate", "aggressive", "aggressive_pos"])
    elif args.regime:
        run([args.regime])
    else:
        ap.error("pass --regime NAME or --all")


if __name__ == "__main__":
    main()
