"""Robustness check: does the topic->ideology finding survive removing the
register-confound topics (T57 generic filler, T46 profanity) flagged in
nlp/topic_ideology_synthesis.py as carrying strong ideology associations
that plausibly reflect CASUAL SPEECH STYLE, not political content?

This is a robustness check on the existing ideology finding, not an attempt
to revive "register organizes the embedding space" (that was tested
separately and directly in nlp/register_pc_regression.py /
nlp/register_ica_check.py and came back negative -- this task does not
touch that conclusion).

Consumes existing artifacts only: K=75 CLR show-level topic profiles
(nlp/lda_ideology_204.py's load_show_topics()), ideology_targets_204.csv
(N=204), and the SAME Ridge/LOSO-CV/permutation harness already used for the
ideology probe (nlp/embed_ideology_nonlinear.py). No new model type -- only
the feature-column subset changes across the four conditions.

Four conditions:
  full_75          all 75 topics (baseline, matches extended_all: R2=0.430)
  drop_T57_T46     73 topics, remove just the two flagged register-confound
                   topics
  drop_all_filler  70 topics, remove the 5 topics explicitly identified as
                   the conversational/register cluster: T61, T12, T57, T39,
                   T46 (top words: "think/way/lot/kind"; "people/know/think/
                   going"; "going/know/right/got"; "know/think/mean/kind";
                   "fucking/right/shit/na" -- all generic discourse markers
                   or profanity, no identifiable political content)
  substantive_only 63 topics -- drop_all_filler's 5 PLUS 7 more topics that
                   are not filler/register in the profanity-vs-generic sense
                   but are equally non-substantive for a "what themes carry
                   ideology" claim: T28 (another generic-filler-pattern
                   topic: "got/know/right/going/guy"), T60 (podcast outro/
                   thank-you boilerplate: "thank/podcast/week/great/thanks"),
                   and 5 ad-read/sponsor-content topics identified by
                   inspecting top words (promo codes, a Spanish-language ad,
                   a chicken-feed/supplement ad, a SiriusXM/insurance ad):
                   T9, T16, T32, T48, T7. This 7-topic extension beyond the
                   named filler-5 is THIS SCRIPT'S OWN judgment call, not a
                   previously-"known" list -- flagged explicitly as such so
                   it's inspectable/contestable, not asserted as objective.

Optional companion: ideology regressed directly on the 6 show-level register
features (mean of nlp/register_features.py's chunk-level features per show),
same CV harness -- a clean, non-topic-removal measurement of register's
direct contribution to the ideology target.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline import config as pipeline_config
from .embed_ideology_nonlinear import bootstrap_r2_ci, loso_cv, r2_from_predictions, ridge_fit_predict
from .lda_ideology_204 import load_show_topics
from .lda_ideology_clr import clr
from .register_pc_regression import FEATURES as REGISTER_FEATURES

TARGET = 500

FILLER_5 = {"T61", "T12", "T57", "T39", "T46"}
REGISTER_CONFOUND_PAIR = {"T57", "T46"}
SUBSTANTIVE_ONLY_EXTRA_DROPS = {
    "T28": "generic-filler-pattern (got/know/right/going/guy) -- same character as the named filler-5, not in original list",
    "T60": "podcast outro/thank-you boilerplate (thank/podcast/week/great/thanks)",
    "T9": "ad-read/promo-code content (slash/code/free/promo/plus)",
    "T16": "ad-read/podcast-meta mix (news/slash/delta/podcast/app)",
    "T32": "ad-read, sponsor product (chicken/sugar/lincoln/groons)",
    "T48": "Spanish-language ad copy (de/la/en/el/seafoam/duracell/clorox)",
    "T7": "ad-read, insurance/SiriusXM sponsor content",
}


def build_dataset():
    OUT = pipeline_config.OUTPUT_DIR
    targets = pd.read_csv(OUT / "ideology_targets_204.csv", dtype={"show_id": str}).set_index("show_id")
    show_topics, cols = load_show_topics()
    joined = show_topics.join(targets, how="inner").dropna(subset=["ideology_primary"])
    print(f"[data] N={len(joined)} shows, {len(cols)} topics")
    return joined, cols


def probe(joined: pd.DataFrame, feature_cols: list[str], label: str) -> dict:
    X_raw = joined[feature_cols].to_numpy(dtype=float)
    X = clr(X_raw) if len(feature_cols) > 1 else X_raw  # CLR needs >=2 parts; guard not expected to trigger here
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
        "loso_cv_r2": float(cv_r2), "ci_95": [lo, hi], "permutation_r2": float(perm_r2),
    }


def register_only_companion(joined_index: pd.Index) -> dict:
    """Direct register->ideology measurement (not a topic removal): mean of
    the 6 chunk-level register features per show, same CV harness."""
    OUT = pipeline_config.OUTPUT_DIR
    targets = pd.read_csv(OUT / "ideology_targets_204.csv", dtype={"show_id": str}).set_index("show_id")
    reg = pd.read_csv(OUT / f"register_features_{TARGET}.csv")
    chunks = pd.read_csv(OUT / f"chunks_{TARGET}_bert.csv", usecols=["chunk_id", "collection_id"])
    chunks["collection_id"] = chunks["collection_id"].astype(str)
    reg = reg.merge(chunks, on="chunk_id", how="inner")
    show_reg = reg.groupby("collection_id")[REGISTER_FEATURES].mean()

    joined = show_reg.join(targets, how="inner").dropna(subset=["ideology_primary"])
    joined = joined.loc[joined.index.intersection(joined_index)]
    print(f"\n[register companion] N={len(joined)} shows with both register features and ideology_primary")

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(joined[REGISTER_FEATURES].to_numpy(dtype=float))
    y = joined["ideology_primary"].to_numpy(dtype=float)

    preds = loso_cv(X, y, ridge_fit_predict)
    cv_r2 = r2_from_predictions(y, preds)
    lo, hi = bootstrap_r2_ci(y, preds)

    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    perm_preds = loso_cv(X, y_shuf, ridge_fit_predict)
    perm_r2 = r2_from_predictions(y_shuf, perm_preds)

    print(f"[register-only -> ideology] N={len(y)}  LOSO-CV R2={cv_r2:.4f}  "
          f"95%CI=[{lo:.4f},{hi:.4f}]  perm R2={perm_r2:.4f}")
    return {
        "n_shows": len(y), "loso_cv_r2": float(cv_r2), "ci_95": [lo, hi], "permutation_r2": float(perm_r2),
    }


def run() -> None:
    OUT = pipeline_config.OUTPUT_DIR
    joined, all_cols = build_dataset()

    substantive_drops = FILLER_5 | set(SUBSTANTIVE_ONLY_EXTRA_DROPS)
    conditions = {
        "full_75": all_cols,
        "drop_T57_T46": [c for c in all_cols if c not in REGISTER_CONFOUND_PAIR],
        "drop_all_filler": [c for c in all_cols if c not in FILLER_5],
        "substantive_only": [c for c in all_cols if c not in substantive_drops],
    }

    print("\n=== FOUR-CONDITION COMPARISON ===")
    results = []
    for label, cols in conditions.items():
        results.append(probe(joined, cols, label))

    baseline_r2 = results[0]["loso_cv_r2"]
    filler_r2 = next(r["loso_cv_r2"] for r in results if r["condition"] == "drop_all_filler")
    delta = filler_r2 - baseline_r2
    if abs(delta) < 0.05:
        verdict = (f"HOLDS: dropping the filler/register cluster barely moves CV R2 "
                   f"({baseline_r2:.3f} -> {filler_r2:.3f}, delta={delta:+.3f}). The ideology signal "
                   f"lives in substantive content; the register confound was NOT driving it. This "
                   f"STRENGTHENS the 'ideology is thematic' claim.")
    elif delta < 0:
        verdict = (f"PARTIAL REGISTER CONTRIBUTION: CV R2 drops from {baseline_r2:.3f} to {filler_r2:.3f} "
                   f"(delta={delta:+.3f}) after removing filler/register topics. Some of the apparent "
                   f"ideology-from-topics signal was register (casual/profane speech tracking one side). "
                   f"Revised claim: ideology is carried by content AND a register component "
                   f"(NOT that register organizes the discourse space -- that was tested and rejected "
                   f"separately).")
    else:
        verdict = (f"R2 IMPROVES after dropping filler/register topics ({baseline_r2:.3f} -> {filler_r2:.3f}, "
                   f"delta={delta:+.3f}) -- the filler topics were adding noise, not signal, to the "
                   f"ideology probe. Ideology is thematic; filler topics were actively diluting the fit.")
    print(f"\n[verdict] {verdict}")

    print("\n=== OPTIONAL COMPANION: register-only -> ideology ===")
    reg_result = register_only_companion(joined.index)
    print(f"[comparison] register-only R2={reg_result['loso_cv_r2']:.3f} vs topics-only (full_75) "
          f"R2={baseline_r2:.3f} -- register alone explains "
          f"{'more' if reg_result['loso_cv_r2'] > baseline_r2 else 'less'} of ideology than topics do.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_shows": len(joined),
        "filler_5_topics": sorted(FILLER_5),
        "substantive_only_extra_drops": SUBSTANTIVE_ONLY_EXTRA_DROPS,
        "conditions": results,
        "verdict": verdict,
        "register_only_companion": reg_result,
        "note": ("This task does NOT revive 'register organizes the embedding space' -- that question "
                 "was tested directly and separately (nlp/register_pc_regression.py, negative result). "
                 "This task only tests whether the IDEOLOGY finding is confound-robust."),
    }
    report_path = OUT / "ideology_filler_robustness_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[report] -> {report_path}")

    print(f"\n{'condition':<20}{'n_topics':<10}{'N':<6}{'CV R2':<10}{'95% CI':<20}{'perm R2':<10}")
    for r in results:
        ci = f"[{r['ci_95'][0]:.3f},{r['ci_95'][1]:.3f}]"
        print(f"{r['condition']:<20}{r['n_topics']:<10}{r['n_shows']:<6}{r['loso_cv_r2']:<10.3f}{ci:<20}{r['permutation_r2']:<10.3f}")


if __name__ == "__main__":
    run()
