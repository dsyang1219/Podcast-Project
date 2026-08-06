"""Does the topic->ideology finding survive dropping ALL hand-flagged
contamination, not just the automatically-detected filler topics?

Earlier check (nlp/ideology_filler_robustness.py) dropped only the
classifier-flagged filler topics from the ORIGINAL baseline K=75 fit and
the finding held. Since then, a hand audit of the moderate-regime topic
list (nlp/lda_regime_refit.py's output, see the topic explorer artifact)
found MORE contamination the classifier missed: ad/sponsor topics, podcast
meta/outro boilerplate, a Spanish-language ad topic, and a profanity topic
that was under the classifier's 50% threshold (3/10 words, not 5/10). This
script closes the question: drop the FULL 18-topic hand-flagged set (see
data/handflagged_contamination.csv) and confirm (or refute) that the
finding survives.

Inputs (all pre-existing, no LDA refit here):
  - data/output/regimes/lda_doctopic_moderate_k75.csv -- chunk-level
    doc-topic matrix for the moderate regime (regenerated once by
    nlp/lda_regime_refit.py specifically because the original comparison
    run never persisted it -- see that module's run_regime() for the
    now-added save step. This is the ONE re-fit in this task's lineage;
    everything downstream of that file is probe-only, matching the "no
    re-fitting LDA" instruction for the ideology-dropout task itself).
  - data/output/ideology_targets_204.csv -- ideology_primary target
    (extended_all condition, matches nlp/lda_ideology_204.py).
  - data/handflagged_contamination.csv -- the auditable 18-topic drop set,
    categorized (ad_sponsor / meta_boilerplate / spanish_ad /
    profanity_register / filler).

Six conditions, same Ridge/LOSO-CV/permutation harness throughout
(imported from nlp.embed_ideology_nonlinear, not reimplemented) + CLR
transform (nlp.lda_ideology_clr.clr, not reimplemented):
  full_75            all 75 topics (baseline for THIS run)
  drop_filler        drop the 5 filler-category topics
  drop_ads_meta      drop ad_sponsor + meta_boilerplate + spanish_ad (12 topics)
  drop_profanity     drop the 1 profanity_register topic
  drop_all_flagged   drop all 18 hand-flagged topics
  substantive_only   keep only the 57 non-flagged topics (== drop_all_flagged
                      by construction; reported as a separate row per the
                      task spec, not a different feature set)

Plus a standalone T52 with/without check: T52 is THIS regime's actual
gold/bitcoin/crypto topic (not T71 -- T71 in the moderate regime is a
Kalshi/mobile-promo ad-read; the task brief's "T71" reference doesn't match
the moderate-regime topic list and is treated as a mislabeling of T52,
flagged explicitly rather than silently followed).

Honesty controls: report all conditions, not just the favorable one; never
re-add a topic to rescue R2 once it's in the hand-flagged file.

Usage:
    .venv/bin/python -m nlp.contamination_dropout_probe
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config as pipeline_config
from .embed_ideology_nonlinear import bootstrap_r2_ci, loso_cv, r2_from_predictions, ridge_fit_predict
from .lda_ideology_clr import clr

REGIME = "moderate"
K = 75
DOCTOPIC_PATH = Path(f"data/output/regimes/lda_doctopic_{REGIME}_k{K}.csv")
CONTAMINATION_PATH = Path("data/handflagged_contamination.csv")
OUT_DIR = Path("data/output/regimes")
GOLD_CRYPTO_TOPIC = 52  # see module docstring: NOT T71 as stated in the task brief


def load_show_topics() -> tuple[pd.DataFrame, list[str]]:
    doc_topic = pd.read_csv(DOCTOPIC_PATH, dtype={"chunk_id": str, "collection_id": str})
    cols = [c for c in doc_topic.columns if c.startswith("T")]
    show_topics = doc_topic.groupby("collection_id")[cols].mean()
    return show_topics, cols


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    OUT = pipeline_config.OUTPUT_DIR
    targets = pd.read_csv(OUT / "ideology_targets_204.csv", dtype={"show_id": str}).set_index("show_id")
    show_topics, cols = load_show_topics()
    joined = show_topics.join(targets, how="inner").dropna(subset=["ideology_primary"])
    print(f"[data] N={len(joined)} shows, {len(cols)} topics (regime={REGIME})")
    return joined, cols


def probe(joined: pd.DataFrame, feature_cols: list[str], label: str) -> dict:
    X_raw = joined[feature_cols].to_numpy(dtype=float)
    eps = 1e-12
    X_raw = np.where(X_raw <= 0, eps, X_raw)
    X = clr(X_raw) if len(feature_cols) > 1 else X_raw
    y = joined["ideology_primary"].to_numpy(dtype=float)
    n = len(y)

    preds = loso_cv(X, y, ridge_fit_predict)
    cv_r2 = r2_from_predictions(y, preds)
    lo, hi = bootstrap_r2_ci(y, preds)

    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    perm_preds = loso_cv(X, y_shuf, ridge_fit_predict)
    perm_r2 = r2_from_predictions(y_shuf, perm_preds)

    print(f"[{label}] n_topics={len(feature_cols)}  N={n}  LOSO-CV R2={cv_r2:.4f}  "
          f"95%CI=[{lo:.4f},{hi:.4f}]  perm R2={perm_r2:.4f}")
    return {
        "condition": label, "n_topics": len(feature_cols), "n_shows": n,
        "loso_cv_r2": float(cv_r2), "ci_95": [float(lo), float(hi)],
        "permutation_r2": float(perm_r2),
    }


def run() -> None:
    joined, all_cols = load_dataset()
    contamination = pd.read_csv(CONTAMINATION_PATH)
    contamination["col"] = contamination["topic_id"].apply(lambda t: f"T{t}")

    all_flagged = set(contamination["col"])
    by_cat = {cat: set(g["col"]) for cat, g in contamination.groupby("category")}
    filler_cols = by_cat.get("filler", set())
    ads_meta_cols = (by_cat.get("ad_sponsor", set()) | by_cat.get("meta_boilerplate", set())
                     | by_cat.get("spanish_ad", set()))
    profanity_cols = by_cat.get("profanity_register", set())

    assert all_flagged == filler_cols | ads_meta_cols | profanity_cols, \
        "category partition doesn't reconstruct the full flagged set -- check handflagged_contamination.csv categories"
    assert len(all_flagged) == 18, f"expected 18 hand-flagged topics, got {len(all_flagged)}"

    conditions = {
        "full_75": all_cols,
        "drop_filler": [c for c in all_cols if c not in filler_cols],
        "drop_ads_meta": [c for c in all_cols if c not in ads_meta_cols],
        "drop_profanity": [c for c in all_cols if c not in profanity_cols],
        "drop_all_flagged": [c for c in all_cols if c not in all_flagged],
        "substantive_only": [c for c in all_cols if c not in all_flagged],
    }

    print("\n=== SIX-CONDITION CONTAMINATION DROPOUT ===")
    results = []
    for label, cols in conditions.items():
        results.append(probe(joined, cols, label))

    by_label = {r["condition"]: r for r in results}
    baseline_r2 = by_label["full_75"]["loso_cv_r2"]
    baseline_ci = by_label["full_75"]["ci_95"]

    print("\n=== T52 (gold/bitcoin/crypto) WITH vs WITHOUT ===")
    gold_col = f"T{GOLD_CRYPTO_TOPIC}"
    assert gold_col in all_cols, f"{gold_col} not found in topic columns"
    with_gold = probe(joined, all_cols, "with_T52")
    without_gold = probe(joined, [c for c in all_cols if c != gold_col], "without_T52")
    gold_delta = with_gold["loso_cv_r2"] - without_gold["loso_cv_r2"]
    gold_verdict = (
        f"T52 behaves like ordinary content: dropping it alone moves R2 by only {gold_delta:+.4f} "
        f"(signal is spread across many topics, not concentrated in this one)."
        if abs(gold_delta) < 0.02 else
        f"T52 behaves like a load-bearing single predictor: dropping it alone moves R2 by {gold_delta:+.4f} "
        f"-- the gold/crypto sponsor topic IS carrying meaningful ideology signal on its own."
    )
    print(f"[T52 check] delta={gold_delta:+.4f} -- {gold_verdict}")

    print("\n=== VERDICT ===")
    drop_all_r2 = by_label["drop_all_flagged"]["loso_cv_r2"]
    drop_all_ci = by_label["drop_all_flagged"]["ci_95"]
    within_ci = drop_all_ci[0] <= baseline_r2 <= drop_all_ci[1] or baseline_ci[0] <= drop_all_r2 <= baseline_ci[1]
    delta = drop_all_r2 - baseline_r2

    category_deltas = {
        "drop_filler": by_label["drop_filler"]["loso_cv_r2"] - baseline_r2,
        "drop_ads_meta": by_label["drop_ads_meta"]["loso_cv_r2"] - baseline_r2,
        "drop_profanity": by_label["drop_profanity"]["loso_cv_r2"] - baseline_r2,
    }
    worst_cat = min(category_deltas, key=lambda k: category_deltas[k])

    if within_ci and abs(delta) < 0.05:
        verdict = (
            f"ROBUST TO ALL CONTAMINATION: dropping all 18 hand-flagged topics moves CV R2 from "
            f"{baseline_r2:.3f} to {drop_all_r2:.3f} (delta={delta:+.3f}), within the baseline's own "
            f"95% CI. The ads/profanity/boilerplate are not load-bearing -- cleaning them further is "
            f"COSMETIC (presentation quality), not scientifically necessary for the ideology finding. "
            f"No category individually moves R2 by more than {max(abs(v) for v in category_deltas.values()):.3f}."
        )
    else:
        verdict = (
            f"NOT FULLY ROBUST: dropping all 18 hand-flagged topics moves CV R2 from {baseline_r2:.3f} "
            f"to {drop_all_r2:.3f} (delta={delta:+.3f}). The category driving the largest drop is "
            f"'{worst_cat}' (delta={category_deltas[worst_cat]:+.3f}). That category carries real "
            f"ideology signal and must be cleaned properly before the finding can be restated as "
            f"content-driven; the current headline R2 is partly a contamination artifact."
        )
    print(verdict)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regime": REGIME,
        "k": K,
        "contamination_file": str(CONTAMINATION_PATH),
        "n_hand_flagged": len(all_flagged),
        "category_counts": {k: len(v) for k, v in by_cat.items()},
        "conditions": results,
        "category_deltas_vs_baseline": category_deltas,
        "gold_crypto_check": {
            "topic_used": gold_col,
            "note_on_task_brief_T71": "task brief cited T71; moderate regime's actual gold/bitcoin/crypto "
                                       "topic is T52 (T71 is a Kalshi/mobile-promo ad-read) -- used T52.",
            "with_topic": with_gold, "without_topic": without_gold,
            "delta": gold_delta, "verdict": gold_verdict,
        },
        "verdict": verdict,
    }
    report_path = OUT_DIR / "contamination_dropout_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path}")

    print(f"\n{'condition':<20}{'n_topics':<10}{'N':<6}{'CV R2':<10}{'95% CI':<20}{'perm R2':<10}")
    for r in results:
        ci = f"[{r['ci_95'][0]:.3f},{r['ci_95'][1]:.3f}]"
        print(f"{r['condition']:<20}{r['n_topics']:<10}{r['n_shows']:<6}{r['loso_cv_r2']:<10.3f}{ci:<20}{r['permutation_r2']:<10.3f}")


if __name__ == "__main__":
    run()
