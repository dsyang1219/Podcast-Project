"""Phase A -- validate sentence-level span detection against the hand-annotated
gold set BEFORE any transcript is modified.

Reports, at the SENTENCE level:
  * precision / recall / F1 for ad_sponsor, meta_boilerplate, and removed
    (ad OR meta -- the union is what excision actually acts on)
  * THE KEY NUMBER: content sentences flagged for removal, printed verbatim,
    split by stratum so substantive political/economic false positives are
    visible separately from ad-adjacent banter
  * boundary accuracy: signed offset between each matched span's edges and the
    gold edges

Gate (enforced here, not eyeballed): ad_sponsor precision >= 0.90 AND the
content false-positive rate <= 2% of gold content sentences. Failing either
means STOP -- report and fall back to the stoplist, do not run Phase B.

    .venv/bin/python -m nlp.adspan_phase_a --model gpt-4o-mini-2024-07-18
    .venv/bin/python -m nlp.adspan_phase_a --model gpt-4.1-nano-2025-04-14
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from pipeline import config as pipeline_config
from .adspan_common import Span, merge_spans
from .adspan_detect import PRICING, detect_window

load_dotenv(override=True)

OUT_DIR = pipeline_config.OUTPUT_DIR / "adspan"
GOLD_PATH = OUT_DIR / "gold_spans.json"

GATE_AD_PRECISION = 0.90
GATE_CONTENT_FP_RATE = 0.02

# Strata whose content sentences are substantive political/economic discourse.
# A false positive here is the failure mode that would deflate the ideology R2
# -- weighted separately from, say, ad-adjacent studio banter.
SUBSTANTIVE_STRATA = {"ambiguous_product", "pure_content"}


def label_sentences(n: int, spans: list[Span]) -> list[str]:
    out = ["content"] * n
    for s in spans:
        for i in range(s.start, min(s.end, n - 1) + 1):
            # ad wins ties so a sentence is never double-counted
            if out[i] == "content" or s.label == "ad_sponsor":
                out[i] = s.label
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f = 2 * p * r / (p + r) if (p + r) and p == p and r == r else float("nan")
    return p, r, f


def boundary_offsets(pred: list[Span], gold: list[Span]) -> list[dict]:
    """For each gold span, the nearest predicted span by overlap, and the
    signed edge offsets (predicted - gold). Negative start = cut too early."""
    out = []
    for g in gold:
        gi = g.indices()
        best, best_ov = None, 0
        for p in pred:
            ov = len(gi & p.indices())
            if ov > best_ov:
                best, best_ov = p, ov
        if best is None:
            out.append({"gold": [g.start, g.end], "label": g.label, "matched": False})
        else:
            out.append({
                "gold": [g.start, g.end], "pred": [best.start, best.end],
                "label": g.label, "matched": True,
                "start_offset": best.start - g.start,
                "end_offset": best.end - g.end,
                "overlap": best_ov, "gold_len": len(gi),
            })
    return out


def run(model: str) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    gold = json.loads(GOLD_PATH.read_text())
    print(f"[phase A] {len(gold)} gold windows, "
          f"{sum(g['n_sentences'] for g in gold)} sentences, model={model}")

    tok_in = tok_out = 0
    per_window = []
    t0 = time.time()
    for i, g in enumerate(gold):
        numbered = "\n".join(f"[{j}] {s}" for j, s in enumerate(g["sentences"]))
        spans, err, pin, pout = detect_window(client, numbered, g["n_sentences"], model=model)
        tok_in += pin
        tok_out += pout
        if err:
            print(f"  [warn] window {i} ({g['window_id']}): {err}")
        per_window.append({"gold": g, "pred": merge_spans(spans), "error": err})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(gold)} windows...", flush=True)
    elapsed = time.time() - t0

    # ---------------------------------------------------------- confusion --
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    content_fps: list[dict] = []
    fn_examples: list[dict] = []
    boundaries: list[dict] = []
    per_window_rows = []

    for w in per_window:
        g = w["gold"]
        n = g["n_sentences"]
        gold_spans = [Span.from_dict(s) for s in g["spans"]]
        y_true = label_sentences(n, gold_spans)
        y_pred = label_sentences(n, w["pred"])
        for t, p in zip(y_true, y_pred):
            counts[(t, p)] += 1
        for j, (t, p) in enumerate(zip(y_true, y_pred)):
            if t == "content" and p != "content":
                content_fps.append({
                    "window_id": g["window_id"], "show_name": g["show_name"],
                    "stratum": g["stratum"], "sentence_index": j,
                    "predicted": p, "text": g["sentences"][j],
                    "substantive": g["stratum"] in SUBSTANTIVE_STRATA,
                })
            elif t != "content" and p == "content":
                fn_examples.append({
                    "window_id": g["window_id"], "stratum": g["stratum"],
                    "sentence_index": j, "gold": t, "text": g["sentences"][j],
                })
        b = boundary_offsets(w["pred"], gold_spans)
        for e in b:
            e["window_id"] = g["window_id"]
        boundaries.extend(b)
        per_window_rows.append({
            "window_id": g["window_id"], "stratum": g["stratum"],
            "show_name": g["show_name"], "n_sentences": n,
            "gold_spans": [s for s in g["spans"]],
            "pred_spans": [s.to_dict() for s in w["pred"]],
            "n_content_fp": sum(1 for t, p in zip(y_true, y_pred)
                                 if t == "content" and p != "content"),
        })

    labels = ["ad_sponsor", "meta_boilerplate", "content"]
    print(f"\n[phase A] done in {elapsed:.1f}s")
    print("\n[phase A] sentence-level confusion (rows=gold, cols=predicted):")
    header = f"{'':<20}" + "".join(f"{c:<20}" for c in labels)
    print(header)
    for t in labels:
        print(f"{t:<20}" + "".join(f"{counts[(t, p)]:<20}" for p in labels))

    metrics = {}
    for lab in ("ad_sponsor", "meta_boilerplate"):
        tp = counts[(lab, lab)]
        fp = sum(counts[(t, lab)] for t in labels if t != lab)
        fn = sum(counts[(lab, p)] for p in labels if p != lab)
        p, r, f = prf(tp, fp, fn)
        metrics[lab] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}
        print(f"\n[phase A] {lab}: precision={p:.3f} recall={r:.3f} F1={f:.3f} "
              f"(tp={tp} fp={fp} fn={fn})")

    # "removed" = the union the excision actually applies.
    rem_tp = sum(counts[(t, p)] for t in ("ad_sponsor", "meta_boilerplate")
                  for p in ("ad_sponsor", "meta_boilerplate"))
    rem_fp = sum(counts[("content", p)] for p in ("ad_sponsor", "meta_boilerplate"))
    rem_fn = sum(counts[(t, "content")] for t in ("ad_sponsor", "meta_boilerplate"))
    rp, rr, rf = prf(rem_tp, rem_fp, rem_fn)
    metrics["removed_union"] = {"precision": rp, "recall": rr, "f1": rf,
                                 "tp": rem_tp, "fp": rem_fp, "fn": rem_fn}
    print(f"\n[phase A] REMOVED (ad OR meta): precision={rp:.3f} recall={rr:.3f} "
          f"F1={rf:.3f} (tp={rem_tp} fp={rem_fp} fn={rem_fn})")

    # ------------------------------------------ the number that gates this --
    n_content = sum(counts[("content", p)] for p in labels)
    n_sub_fp = sum(1 for c in content_fps if c["substantive"])
    fp_rate = len(content_fps) / n_content if n_content else 0.0
    print(f"\n[phase A] *** DEFLATION RISK *** content sentences marked for removal: "
          f"{len(content_fps)}/{n_content} ({fp_rate:.2%})")
    print(f"[phase A] of those, in substantive political/economic strata: {n_sub_fp}")
    if content_fps:
        print("\n[phase A] every content false positive, verbatim:")
        for c in content_fps:
            flag = "SUBSTANTIVE" if c["substantive"] else "other"
            print(f"  [{flag}] {c['window_id']} s{c['sentence_index']} "
                  f"-> {c['predicted']}  ({c['show_name']}, {c['stratum']})")
            print(f"      {c['text'][:400]!r}")

    matched = [b for b in boundaries if b.get("matched")]
    n_exact = sum(1 for b in matched if b["start_offset"] == 0 and b["end_offset"] == 0)
    if matched:
        mean_abs_start = sum(abs(b["start_offset"]) for b in matched) / len(matched)
        mean_abs_end = sum(abs(b["end_offset"]) for b in matched) / len(matched)
        print(f"\n[phase A] boundary accuracy over {len(matched)} matched gold spans "
              f"({len(boundaries) - len(matched)} gold spans missed entirely):")
        print(f"  exact on both edges: {n_exact}/{len(matched)}")
        print(f"  mean |start offset| = {mean_abs_start:.2f} sentences, "
              f"mean |end offset| = {mean_abs_end:.2f}")
        over = [b for b in matched if b["start_offset"] < 0 or b["end_offset"] > 0]
        print(f"  spans extending BEYOND the gold span (the risky direction): {len(over)}")
        for b in over:
            print(f"    {b['window_id']} gold {b['gold']} -> pred {b['pred']} ({b['label']})")
    else:
        mean_abs_start = mean_abs_end = float("nan")

    # ------------------------------------------------------------- costs --
    price = PRICING.get(model, PRICING["gpt-4o-mini"])
    cost = tok_in / 1e6 * price["in"] + tok_out / 1e6 * price["out"]
    n_sent_gold = sum(g["n_sentences"] for g in gold)
    in_per_sent = tok_in / n_sent_gold
    out_per_sent = tok_out / n_sent_gold
    print(f"\n[phase A] tokens: {tok_in} in / {tok_out} out over {len(gold)} windows "
          f"({n_sent_gold} sentences). cost=${cost:.4f}")
    print(f"[phase A] per-sentence: {in_per_sent:.1f} in / {out_per_sent:.2f} out")

    summary = {
        "model": model, "n_windows": len(gold), "n_sentences": n_sent_gold,
        "elapsed_sec": elapsed,
        "confusion": {f"{t}->{p}": counts[(t, p)] for t in labels for p in labels},
        "metrics": metrics,
        "n_content_sentences": n_content,
        "content_false_positives": len(content_fps),
        "content_false_positive_rate": fp_rate,
        "substantive_content_false_positives": n_sub_fp,
        "content_false_positive_detail": content_fps,
        "n_false_negatives": len(fn_examples),
        "boundary": {
            "n_gold_spans": len(boundaries), "n_matched": len(matched),
            "n_exact_both_edges": n_exact,
            "mean_abs_start_offset": mean_abs_start,
            "mean_abs_end_offset": mean_abs_end,
            "detail": boundaries,
        },
        "tokens_in": tok_in, "tokens_out": tok_out, "cost_usd": cost,
        "tokens_in_per_sentence": in_per_sent,
        "tokens_out_per_sentence": out_per_sent,
        "per_window": per_window_rows,
    }

    gate_p = metrics["ad_sponsor"]["precision"] >= GATE_AD_PRECISION
    gate_fp = fp_rate <= GATE_CONTENT_FP_RATE
    summary["gate_ad_precision_ok"] = bool(gate_p)
    summary["gate_content_fp_ok"] = bool(gate_fp)
    summary["gate_pass"] = bool(gate_p and gate_fp)

    print(f"\n[phase A] GATE ad_sponsor precision >= {GATE_AD_PRECISION}: {gate_p} "
          f"({metrics['ad_sponsor']['precision']:.3f})")
    print(f"[phase A] GATE content FP rate <= {GATE_CONTENT_FP_RATE:.0%}: {gate_fp} "
          f"({fp_rate:.2%}, n={len(content_fps)}, substantive={n_sub_fp})")
    print(f"[phase A] GATE RESULT: "
          f"{'PASS -- Phase B authorized' if summary['gate_pass'] else 'FAIL -- STOP'}")

    tag = model.replace(".", "").replace("-", "_")
    (OUT_DIR / f"phase_a_{tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[phase A] -> {OUT_DIR / f'phase_a_{tag}.json'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    args = ap.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
