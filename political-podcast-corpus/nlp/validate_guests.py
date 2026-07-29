"""Validation harness for the guest-extraction cascade -- this is what turns
extraction into science (see nlp/extract_guests.py's docstring: quality here
must be established empirically since Haiku != GPT-4, Pew/DeMets's model).

Two modes:

    .venv/bin/python -m nlp.validate_guests emit  [--n 250] [--seed 0]
    .venv/bin/python -m nlp.validate_guests score --gold data/output/gold_template.csv

`emit`: draws a sample STRATIFIED by show prominence (episode count --
tertiles: small/medium/large, sampled ~evenly so precision/recall is
estimable per stratum, not dominated by a few high-volume shows), runs the
IDENTICAL extraction cascade used by nlp.extract_guests (imports
extract_for_episode directly -- no reimplementation) to produce predictions,
then writes TWO files:
  - data/output/gold_template.csv -- BLIND: episode_id, show, title,
    description, and two EMPTY columns (gold_has_guests, gold_names) for a
    human to fill in. No model predictions appear in this file.
  - data/output/gold_predictions.jsonl -- the predictions, kept separate so
    coding stays blind. Not meant for human review before coding is done.

`score`: takes the FILLED gold_template.csv (gold_has_guests filled with
TRUE/FALSE, gold_names filled with a semicolon-separated name list where
applicable), joins to gold_predictions.jsonl by episode_id, and reports:
  - episode-level P/R/F1 (predicted has_guests vs gold_has_guests)
  - name-level P/R/F1 (fuzzy set match, token_set_ratio>=85), OVERALL and
    PER TIER (so tier-specific accuracy is visible, not just a blended
    number)
  - explicit comparison to Pew's raw F1~=0.77 benchmark
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz

from pipeline import config as pipeline_config
from .extract_guests import Usage, extract_for_episode, load_approved_patterns
from .guest_common import load_episodes, load_hosts, strip_html_urls

load_dotenv()

PEW_BENCHMARK_F1 = 0.77
NAME_MATCH_MIN_RATIO = 85  # slightly looser than canonicalization's 92 --
# tolerates informal gold-coding entry (e.g. coder writes "Ted Cruz" while
# model wrote "Sen. Ted Cruz") without conflating genuinely different people.


# ---------------------------------------------------------------- emit ----

def stratify_sample(eps: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    counts = eps.groupby("collection_id").size()
    q1, q2 = counts.quantile([1 / 3, 2 / 3])
    strata = pd.cut(counts, bins=[-1, q1, q2, np.inf], labels=["small", "medium", "large"])
    eps = eps.copy()
    eps["stratum"] = eps["collection_id"].map(strata)

    per_stratum = n // 3
    parts = []
    for s in ["small", "medium", "large"]:
        pool = eps[eps["stratum"] == s]
        take = min(per_stratum, len(pool))
        parts.append(pool.sample(n=take, random_state=seed))
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    return sample


def emit(n: int, seed: int) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    eps = load_episodes()
    sample = stratify_sample(eps, n, seed)
    print(f"[sample] n={len(sample)}  by stratum: {sample['stratum'].value_counts().to_dict()}")

    client = anthropic.Anthropic(max_retries=6)
    usage = Usage()
    hosts_by_show = load_hosts()
    approved_patterns = load_approved_patterns()

    pred_path = OUT / "gold_predictions.jsonl"
    predictions = []
    with pred_path.open("w") as pf:
        for i, row in sample.iterrows():
            try:
                result = extract_for_episode(client, usage, row, approved_patterns, hosts_by_show)
                pred = {
                    "episode_id": result["episode_id"], "collection_id": result["collection_id"],
                    "tier": result["tier"], "guests": [g["name"] for g in result["guests"]],
                }
            except anthropic.APIError as e:
                # Persistent failure on this one episode (SDK already retried up to
                # max_retries internally) -- don't lose the rest of the paid-for batch.
                print(f"[warn] episode {row['episode_id']} failed after SDK retries ({e}); marking tier=error")
                pred = {"episode_id": row["episode_id"], "collection_id": row["collection_id"],
                        "tier": "error", "guests": []}
            predictions.append(pred)
            pf.write(json.dumps(pred, ensure_ascii=False) + "\n")
            pf.flush()
            if (i + 1) % 25 == 0 or (i + 1) == len(sample):
                print(f"[progress] {i+1}/{len(sample)}  cost so far=${usage.cost():.4f}")

    print(f"[predictions, NOT for review before coding] -> {pred_path.name}  "
          f"(cost: ${usage.cost():.4f})")

    template_path = OUT / "gold_template.csv"
    with template_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_id", "collection_id", "show_name", "stratum",
                          "episode_title", "episode_description",
                          "gold_has_guests", "gold_names"])
        for _, row in sample.iterrows():
            raw_desc = row["episode_description"] if isinstance(row["episode_description"], str) else ""
            writer.writerow([
                row["episode_id"], row["collection_id"], row["show_name"], row["stratum"],
                row["episode_title"], strip_html_urls(raw_desc),
                "", "",
            ])
    print(f"[BLIND hand-coding template] -> {template_path.name}")
    print(f"\nFill in gold_has_guests (TRUE/FALSE) and gold_names (semicolon-separated, blank if none) "
          f"per row, then run:\n  .venv/bin/python -m nlp.validate_guests score --gold {template_path}")


# --------------------------------------------------------------- score ----

def parse_bool(s) -> bool | None:
    if not isinstance(s, str) or not s.strip():
        return None
    return s.strip().lower() in {"true", "t", "1", "yes", "y"}


def parse_gold_names(s) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [n.strip() for n in s.split(";") if n.strip()]


def match_name_sets(pred_names: list[str], gold_names: list[str]) -> tuple[int, int, int, list[str], list[str]]:
    """Fuzzy set match (greedy, one-to-one). Returns (tp, fp, fn, extra_names, missed_names)
    where extra_names are unmatched pred names (false positives) and missed_names are
    unmatched gold names (false negatives) -- the raw material for the error dump."""
    pred_left = list(pred_names)
    gold_left = list(gold_names)
    tp = 0
    matched_pred_idx = set()
    matched_gold_idx = set()
    for gi, g in enumerate(gold_left):
        best_j, best_score = None, -1
        for pi, p in enumerate(pred_left):
            if pi in matched_pred_idx:
                continue
            score = fuzz.token_set_ratio(g, p)
            if score > best_score:
                best_j, best_score = pi, score
        if best_j is not None and best_score >= NAME_MATCH_MIN_RATIO:
            tp += 1
            matched_pred_idx.add(best_j)
            matched_gold_idx.add(gi)
    extra_names = [p for pi, p in enumerate(pred_left) if pi not in matched_pred_idx]
    missed_names = [g for gi, g in enumerate(gold_left) if gi not in matched_gold_idx]
    fp = len(extra_names)
    fn = len(missed_names)
    return tp, fp, fn, extra_names, missed_names


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    r = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def score(gold_path: Path) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    pred_path = OUT / "gold_predictions.jsonl"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found -- run `emit` first")

    preds = {json.loads(l)["episode_id"]: json.loads(l) for l in pred_path.open()}
    gold_df = pd.read_csv(gold_path, dtype=str)

    n_unfilled = 0
    episode_tp = episode_fp = episode_fn = episode_tn = 0
    name_tp = name_fp = name_fn = 0
    per_tier = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_episodes": 0})
    episode_errors = []  # has_guests FP/FN, for inspecting failure modes
    name_errors = []     # per-episode extra/missed names, for inspecting failure modes

    for _, row in gold_df.iterrows():
        eid = row["episode_id"]
        gold_has = parse_bool(row.get("gold_has_guests"))
        if gold_has is None:
            n_unfilled += 1
            continue
        pred = preds.get(eid)
        if pred is None:
            continue
        pred_names = pred["guests"]
        pred_has = len(pred_names) > 0
        gold_names = parse_gold_names(row.get("gold_names"))
        title = row.get("episode_title", "")

        if pred_has and gold_has:
            episode_tp += 1
        elif pred_has and not gold_has:
            episode_fp += 1
            episode_errors.append({"type": "false_positive", "episode_id": eid, "title": title,
                                    "tier": pred["tier"], "model_said": pred_names, "gold_said": gold_names})
        elif not pred_has and gold_has:
            episode_fn += 1
            episode_errors.append({"type": "false_negative", "episode_id": eid, "title": title,
                                    "tier": pred["tier"], "model_said": pred_names, "gold_said": gold_names})
        else:
            episode_tn += 1

        if gold_has or pred_has:
            tp, fp, fn, extra_names, missed_names = match_name_sets(pred_names, gold_names)
            name_tp += tp
            name_fp += fp
            name_fn += fn
            t = per_tier[pred["tier"]]
            t["tp"] += tp
            t["fp"] += fp
            t["fn"] += fn
            t["n_episodes"] += 1
            if extra_names or missed_names:
                name_errors.append({"episode_id": eid, "title": title, "tier": pred["tier"],
                                     "model_said": pred_names, "gold_said": gold_names,
                                     "extra_names": extra_names, "missed_names": missed_names})

    n_scored = len(gold_df) - n_unfilled
    if n_unfilled:
        print(f"[warn] {n_unfilled}/{len(gold_df)} rows have no gold_has_guests filled in -- skipped")

    ep_p, ep_r, ep_f1 = prf1(episode_tp, episode_fp, episode_fn)
    ep_acc = (episode_tp + episode_tn) / n_scored if n_scored else 0.0

    name_p, name_r, name_f1 = prf1(name_tp, name_fp, name_fn)

    tier_report = {}
    for tier, c in per_tier.items():
        p, r, f1 = prf1(c["tp"], c["fp"], c["fn"])
        tier_report[tier] = {"n_episodes": c["n_episodes"], "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                              "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_gold_rows": len(gold_df), "n_scored": n_scored, "n_unfilled_skipped": n_unfilled,
        "episode_level": {
            "tp": episode_tp, "fp": episode_fp, "fn": episode_fn, "tn": episode_tn,
            "accuracy": round(ep_acc, 3), "precision": round(ep_p, 3), "recall": round(ep_r, 3), "f1": round(ep_f1, 3),
        },
        "name_level_overall": {
            "tp": name_tp, "fp": name_fp, "fn": name_fn,
            "precision": round(name_p, 3), "recall": round(name_r, 3), "f1": round(name_f1, 3),
        },
        "name_level_per_tier": tier_report,
        "pew_benchmark_f1": PEW_BENCHMARK_F1,
        "vs_pew": ("meets or beats Pew's raw F1~=0.77" if name_f1 >= PEW_BENCHMARK_F1
                   else f"below Pew's raw F1~=0.77 by {PEW_BENCHMARK_F1 - name_f1:.3f}"),
        "error_dump": {
            "n_episode_level_errors": len(episode_errors),
            "n_name_level_errors": len(name_errors),
            "episode_level": episode_errors,
            "name_level": name_errors,
        },
    }

    report_path = OUT / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    error_dump_path = OUT / "error_dump.json"
    error_dump_path.write_text(json.dumps(
        {"episode_level": episode_errors, "name_level": name_errors}, indent=2, ensure_ascii=False))

    print(f"\n=== EPISODE-LEVEL (has_guests) === n_scored={n_scored}")
    print(f"  accuracy={ep_acc:.3f}  precision={ep_p:.3f}  recall={ep_r:.3f}  f1={ep_f1:.3f}")
    print(f"\n=== NAME-LEVEL (fuzzy set match, overall) ===")
    print(f"  tp={name_tp} fp={name_fp} fn={name_fn}  precision={name_p:.3f}  recall={name_r:.3f}  f1={name_f1:.3f}")
    print(f"\n=== NAME-LEVEL PER TIER ===")
    for tier, r in tier_report.items():
        print(f"  {tier:<20} n_ep={r['n_episodes']:>4}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")

    print(f"\n=== ERROR DUMP ({len(episode_errors)} episode-level, {len(name_errors)} name-level) ===")
    for e in episode_errors:
        print(f"  [{e['type']}] {e['episode_id']}  \"{e['title'][:70]}\"\n"
              f"      model={e['model_said']}  gold={e['gold_said']}")
    for e in name_errors:
        if e["extra_names"] or e["missed_names"]:
            print(f"  [name-error/{e['tier']}] {e['episode_id']}  \"{e['title'][:70]}\"\n"
                  f"      extra(fp)={e['extra_names']}  missed(fn)={e['missed_names']}")
    print(f"\n[error dump] -> {error_dump_path.name}")
    print(f"\n=== vs Pew (raw F1~={PEW_BENCHMARK_F1}) ===\n  {report['vs_pew']}")
    print(f"\n[report] -> {report_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_emit = sub.add_parser("emit")
    ap_emit.add_argument("--n", type=int, default=250)
    ap_emit.add_argument("--seed", type=int, default=0)

    ap_score = sub.add_parser("score")
    ap_score.add_argument("--gold", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "emit":
        emit(args.n, args.seed)
    else:
        score(args.gold)


if __name__ == "__main__":
    main()
