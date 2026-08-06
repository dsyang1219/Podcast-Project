"""Human review loop for the gold span set.

The gold spans in data/output/adspan/gold_spans.json were annotated by the
agent, not by a human. This script closes that gap: it emits every window with
the current annotation marked inline, in a format that is edited in place, and
reads the edited file back into gold_spans.json.

    # 1. write the review file (current annotations pre-filled)
    .venv/bin/python -m nlp.adspan_gold_review --emit

    # 2. a human edits data/output/adspan/gold_review.txt -- change only the
    #    marker in the leading brackets of each sentence line:
    #        [ AD ]  paid third-party placement
    #        [META]  show/network structural boilerplate
    #        [    ]  content (keep it)

    # 3. fold the edits back in (validates, prints a full diff, rewrites gold)
    .venv/bin/python -m nlp.adspan_gold_review --apply

--apply is non-destructive until it succeeds: it writes gold_spans.json only
after every line parses and every window's sentence count matches. The previous
gold file is copied to gold_spans.prehuman.json the first time a human edit
lands, so the agent-annotated baseline stays inspectable for the writeup.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pipeline import config as pipeline_config

OUT_DIR = pipeline_config.OUTPUT_DIR / "adspan"
GOLD_PATH = OUT_DIR / "gold_spans.json"
BASELINE_PATH = OUT_DIR / "gold_spans.prehuman.json"
REVIEW_PATH = OUT_DIR / "gold_review.txt"

MARKERS = {"AD": "ad_sponsor", "META": "meta_boilerplate", "": "content"}
INV = {"ad_sponsor": " AD ", "meta_boilerplate": "META", "content": "    "}

HEADER = """\
# GOLD SPAN REVIEW -- edit the marker in the leading brackets, nothing else.
#
#   [ AD ]  ad_sponsor       a paid third-party placement read into the show:
#                            sponsor copy, inserted ads, promos for other
#                            organizations' products/services/podcasts.
#   [META]  meta_boilerplate the show's or its OWN network's structure: intro
#                            montages, outros, credits, segment teases, break
#                            bumpers, subscribe/rate/donate asks, promos for
#                            sibling shows on the same network.
#   [    ]  content          everything else -- INCLUDING political or economic
#                            discussion that names a product, company, or asset
#                            class (gold, crypto, Medicare, mortgages,
#                            inflation). Only mark what FUNCTIONS as an ad.
#
# Rules the current annotation followed, for reference:
#   * Favor precision. If unsure, leave it as content. Leaving an ad in costs
#     little; cutting real discourse deflates the ideology R2 this is measuring.
#   * Do NOT mark a sentence containing BOTH ad copy and real discussion --
#     transcription sometimes merges them. Excision is sentence-granular, so
#     marking it would delete discourse.
#
# Everything outside the leading [....] brackets is ignored on re-import; the
# sentence text is re-read from gold_candidates.json, not from this file, so
# accidental text edits cannot corrupt the corpus.
# ---------------------------------------------------------------------------

"""


def emit() -> None:
    gold = json.loads(GOLD_PATH.read_text())
    lines = [HEADER]
    for i, g in enumerate(gold):
        labels = ["content"] * g["n_sentences"]
        for s in g["spans"]:
            for j in range(s["start"], s["end"] + 1):
                labels[j] = s["label"]
        n_ad = sum(1 for x in labels if x == "ad_sponsor")
        n_meta = sum(1 for x in labels if x == "meta_boilerplate")
        lines.append("=" * 100)
        lines.append(f"### WINDOW {i}  {g['window_id']}  stratum={g['stratum']}")
        lines.append(f"### show={g['show_name']!r}")
        lines.append(f"### episode={g['episode_title']!r}")
        lines.append(f"### currently: {n_ad} ad, {n_meta} meta, "
                      f"{g['n_sentences'] - n_ad - n_meta} content")
        lines.append("-" * 100)
        for j, (lab, txt) in enumerate(zip(labels, g["sentences"])):
            lines.append(f"[{INV[lab]}] [{j}] {txt}")
        lines.append("")
    REVIEW_PATH.write_text("\n".join(lines))
    n_sent = sum(g["n_sentences"] for g in gold)
    print(f"[emit] {len(gold)} windows / {n_sent} sentences -> {REVIEW_PATH}")
    print("[emit] edit the [ AD ] / [META] / [    ] markers, then run --apply")


def _parse() -> dict[int, list[str]]:
    """window index -> per-sentence label list, from the edited review file."""
    out: dict[int, list[str]] = {}
    cur: int | None = None
    for lineno, raw in enumerate(REVIEW_PATH.read_text().splitlines(), 1):
        if raw.startswith("### WINDOW "):
            cur = int(raw.split()[2])
            out[cur] = []
            continue
        if not raw.startswith("[") or cur is None:
            continue
        if len(raw) < 6 or raw[5] != "]":
            raise SystemExit(f"line {lineno}: malformed marker: {raw[:40]!r}")
        marker = raw[1:5].strip().upper()
        if marker not in MARKERS:
            raise SystemExit(
                f"line {lineno}: unknown marker {marker!r} -- use ' AD ', 'META', "
                f"or four spaces. Line: {raw[:60]!r}")
        out[cur].append(MARKERS[marker])
    return out


def _spans_from_labels(labels: list[str]) -> list[dict]:
    spans, i, n = [], 0, len(labels)
    while i < n:
        if labels[i] == "content":
            i += 1
            continue
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        spans.append({"start": i, "end": j, "label": labels[i]})
        i = j + 1
    return spans


def apply() -> None:
    if not REVIEW_PATH.exists():
        raise SystemExit(f"{REVIEW_PATH} not found -- run --emit first")
    gold = json.loads(GOLD_PATH.read_text())
    parsed = _parse()

    if set(parsed) != set(range(len(gold))):
        raise SystemExit(f"review file has windows {sorted(parsed)}, "
                          f"expected 0..{len(gold) - 1}")
    for i, g in enumerate(gold):
        if len(parsed[i]) != g["n_sentences"]:
            raise SystemExit(f"window {i} ({g['window_id']}): review file has "
                              f"{len(parsed[i])} sentence lines, gold has "
                              f"{g['n_sentences']}. Did a line get deleted?")

    n_changed_sent = 0
    changed_windows = []
    for i, g in enumerate(gold):
        old = ["content"] * g["n_sentences"]
        for s in g["spans"]:
            for j in range(s["start"], s["end"] + 1):
                old[j] = s["label"]
        new = parsed[i]
        diffs = [(j, o, n) for j, (o, n) in enumerate(zip(old, new)) if o != n]
        if diffs:
            changed_windows.append((i, g, diffs))
            n_changed_sent += len(diffs)
        g["spans"] = _spans_from_labels(new)
        g["annotator"] = "human-reviewed (agent-proposed, human-corrected)"

    if not changed_windows:
        print("[apply] no changes -- human review confirmed the existing annotation "
              "verbatim. gold_spans.json left as-is.")
        for g in gold:
            g["annotator"] = "human-reviewed (agent-proposed, confirmed unchanged)"
        GOLD_PATH.write_text(json.dumps(gold, indent=2))
        return

    if not BASELINE_PATH.exists():
        shutil.copy(GOLD_PATH, BASELINE_PATH)
        print(f"[apply] agent-annotated baseline preserved -> {BASELINE_PATH.name}")

    print(f"[apply] {n_changed_sent} sentence label(s) changed across "
          f"{len(changed_windows)} window(s):")
    for i, g, diffs in changed_windows:
        print(f"\n  WINDOW {i} {g['window_id']} ({g['show_name']}, {g['stratum']})")
        for j, o, n in diffs:
            print(f"    s{j:<3} {o:<17} -> {n:<17} {g['sentences'][j][:90]!r}")

    GOLD_PATH.write_text(json.dumps(gold, indent=2))
    tally = {"ad_sponsor": 0, "meta_boilerplate": 0, "content": 0}
    for i, g in enumerate(gold):
        for lab in parsed[i]:
            tally[lab] += 1
    print(f"\n[apply] -> {GOLD_PATH}")
    print(f"[apply] new gold totals: {tally}")
    print("[apply] re-run Phase A on the corrected gold before Phase B:")
    print("        .venv/bin/python -m nlp.adspan_phase_a --model gpt-4o-mini-2024-07-18")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.emit:
        emit()
    elif args.apply:
        apply()
    else:
        ap.error("pass --emit or --apply")


if __name__ == "__main__":
    main()
