#!/usr/bin/env python3
"""Pilot: what the accelerated pipeline costs in accuracy, and what it buys in speed.

Three sub-commands.

  sample   Pick the pilot episodes: --per-show episodes from every show with
           archived audio, stratified over the show's episode list, and write
           them as show_id,episode_id lines for diarize.py --episodes-file.
           Also picks --hand-label episodes across speaker-count strata for
           manual labelling and writes a labelling sheet
           (data/output/diarization/handlabel_<episode>.csv: one row per
           Whisper segment with start, end, text and an empty `label` column;
           fill in H for host, G for guest/other, A for ad/clip, X for unsure).

  compare  Diarization error rate of every variant against the stock variant
           (pyannote.metrics, 0.25 s collar, overlap included) plus speed, written
           to results/diarization_pilot.md. Variants are the diarize.py --suffix
           values found under data/diarization/.

  score    Agreement of the host labels (identify_hosts.py) with the hand
           labels: segment-level precision/recall of "host" by duration.

    ../whisperx-env/bin/python step4_diarize/pilot_eval.py sample --per-show 10 --hand-label 20
    ../whisperx-env/bin/python step4_diarize/pilot_eval.py compare
    ../whisperx-env/bin/python step4_diarize/pilot_eval.py score
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "data/audio_archive"
TRANSCRIPTS_DIR = PROJECT_ROOT / "data/transcripts"
DIAR_DIR = PROJECT_ROOT / "data/diarization"
OUT_DIR = PROJECT_ROOT / "data/output/diarization"
RESULTS = PROJECT_ROOT / "results/diarization_pilot.md"
CORPUS_HOURS = 75_359  # measured: sum of transcript durations over 84,879 episodes (7 Nov 2026)
AUDIO_SUFFIXES = (".opus", ".mp3", ".m4a", ".wav")


def archived() -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for show_dir in ARCHIVE_DIR.iterdir() if ARCHIVE_DIR.exists() else []:
        if show_dir.is_dir() and not show_dir.name.startswith("_"):
            for p in show_dir.iterdir():
                if p.suffix in AUDIO_SUFFIXES and list((TRANSCRIPTS_DIR / show_dir.name).glob(f"{p.stem}_*.json")):
                    out[show_dir.name].append(p.stem)
    return out


# --------------------------------------------------------------------------- sample

def cmd_sample(args) -> None:
    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = archived()
    rows = []
    for show_id, eps in sorted(pool.items()):
        eps = sorted(eps)
        if len(eps) <= args.per_show:
            pick = eps
        else:  # evenly spaced through the (id-sorted) list, then shuffled within
            idx = np.linspace(0, len(eps) - 1, args.per_show).round().astype(int)
            pick = [eps[i] for i in idx]
        rows += [(show_id, e) for e in pick]
    out = OUT_DIR / "pilot_episodes.txt"
    out.write_text("# show_id,episode_id\n" + "\n".join(f"{s},{e}" for s, e in rows) + "\n")
    print(f"pilot: {len(rows)} episodes from {len(pool)} shows -> {out}")

    hand = rng.sample(rows, min(args.hand_label, len(rows)))
    (OUT_DIR / "handlabel_episodes.txt").write_text("# show_id,episode_id\n" + "\n".join(f"{s},{e}" for s, e in hand) + "\n")
    for show_id, eid in hand:
        t = json.loads(sorted((TRANSCRIPTS_DIR / show_id).glob(f"{eid}_*.json"))[0].read_text())
        with open(OUT_DIR / f"handlabel_{eid}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["show_id", "episode_id", "start", "end", "text", "label"])
            for s in t["segments"]:
                w.writerow([show_id, eid, s["start"], s["end"], s["text"], ""])
    print(f"hand-label sheets: {len(hand)} episodes -> {OUT_DIR}/handlabel_<episode>.csv (label H/G/A/X)")


# --------------------------------------------------------------------------- compare

def load_variants() -> dict[str, dict[tuple[str, str], dict]]:
    out: dict[str, dict] = defaultdict(dict)
    for show_dir in DIAR_DIR.iterdir() if DIAR_DIR.exists() else []:
        if not show_dir.is_dir():
            continue
        for p in show_dir.glob("*.json"):
            parts = p.name.split(".")
            variant = parts[1] if len(parts) == 3 else "default"
            rec = json.loads(p.read_text())
            out[variant][(show_dir.name, rec["episode_id"])] = rec
    return out


def to_annotation(turns: list[dict]):
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for t in turns:
        ann[Segment(t["start"], t["end"])] = t["speaker"]
    return ann


def cmd_compare(args) -> None:
    from pyannote.metrics.diarization import DiarizationErrorRate
    variants = load_variants()
    if args.reference not in variants:
        raise SystemExit(f"reference variant {args.reference!r} not found; have {sorted(variants)}")
    ref = variants[args.reference]
    if args.episodes_file:
        keep = {tuple(l.strip().split(",")[:2]) for l in Path(args.episodes_file).read_text().splitlines()
                if l.strip() and not l.startswith("#")}
        ref = {k: v for k, v in ref.items() if k in keep}
    lines = [f"# Diarization pilot — reference `{args.reference}`", "",
             f"Reference variant: `{args.reference}`. DER is computed treating the reference "
             "variant's output as ground truth, so it measures how far each acceleration moves the result, not "
             "absolute accuracy. 0.25 s collar, overlapping speech included.", "",
             f"| variant | episodes | hours | DER vs reference | speakers (mean) | realtime factor | GPU-hours for {CORPUS_HOURS:,} h |",
             "|---|---|---|---|---|---|---|"]
    for name, recs in sorted(variants.items()):
        common = sorted(set(recs) & set(ref))
        if not common:
            continue
        metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
        hours = 0.0
        for key in common:
            metric(to_annotation(ref[key]["turns"]), to_annotation(recs[key]["turns"]))
            hours += recs[key]["duration"] / 3600
        der = abs(metric)
        rtf = np.array([r["provenance"]["realtime_factor"] or np.nan for k, r in recs.items() if k in set(common)])
        nspk = np.mean([recs[k]["n_speakers"] for k in common])
        rtf_m = np.nanmean(rtf)
        lines.append(f"| {name} | {len(common)} | {hours:.1f} | {der:.3f} | {nspk:.2f} | {rtf_m:.0f}x | {CORPUS_HOURS / rtf_m:.0f} |")
    cfg = next(iter(variants[args.reference].values()))["provenance"]
    lines += ["", f"Model: `{cfg['model']}`, clustering `{cfg['clustering']}`, device {cfg['device']}.", ""]
    for name, recs in sorted(variants.items()):
        p = next(iter(recs.values()))["provenance"]
        lines.append(f"- `{name}`: step {p['segmentation_step_s']:.1f} s, batch {p['batch_size']}, "
                     f"embed {p['embed_mode']}, mcs {p['mcs_fraction'] or 'stock'}")
    out = Path(args.out) if args.out else RESULTS
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with open(out, mode) as f:
        f.write(("\n" if args.append else "") + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", out)


# --------------------------------------------------------------------------- score

def cmd_score(args) -> None:
    ep = pd.read_csv(OUT_DIR / "episode_speakers.csv", dtype={"show_id": str, "episode_id": str})
    host_of = {(r.show_id, r.episode_id, r.speaker) for r in ep.itertuples() if r.is_host == 1}
    variants = load_variants()
    recs = variants.get(args.variant, {})
    tp = fp = fn = 0.0
    n_ep = 0
    per_ep = []
    for sheet in sorted(OUT_DIR.glob("handlabel_*.csv")):
        lab = pd.read_csv(sheet, dtype={"show_id": str, "episode_id": str})
        if lab.label.isna().all():
            continue
        key = (lab.show_id.iloc[0], lab.episode_id.iloc[0])
        if key not in recs:
            continue
        pv = recs[key].get("provenance", {})
        td = pv.get("transcript_duration")
        if not pv.get("aligned", td is not None and abs(float(recs[key]["duration"]) - float(td)) <= 0.5):
            print(f"  skipping {key[1]}: transcript timestamps do not match the diarized audio (re-transcribe it)")
            continue
        n_ep += 1
        pred = {(round(s["start"], 2), round(s["end"], 2)): s.get("speaker") for s in recs[key]["segments"]}
        e_tp = e_fp = e_fn = 0.0
        for r in lab.itertuples():
            if not isinstance(r.label, str) or r.label.strip().upper() in ("", "X", "A"):
                continue
            truth = r.label.strip().upper() == "H"
            spk = pred.get((round(r.start, 2), round(r.end, 2)))
            guess = (key[0], key[1], spk) in host_of
            d = r.end - r.start
            e_tp += d * (truth and guess)
            e_fp += d * (guess and not truth)
            e_fn += d * (truth and not guess)
        tp, fp, fn = tp + e_tp, fp + e_fp, fn + e_fn
        per_ep.append((key[1], e_tp / max(1e-6, e_tp + e_fp), e_tp / max(1e-6, e_tp + e_fn)))
    if not n_ep:
        print("no filled hand-label sheets found under", OUT_DIR)
        return
    prec, rec = tp / max(1e-6, tp + fp), tp / max(1e-6, tp + fn)
    print(f"host identification vs hand labels, {n_ep} episodes, by speech duration:")
    print(f"  precision {prec:.3f}  recall {rec:.3f}  F1 {2 * prec * rec / max(1e-6, prec + rec):.3f}")
    for eid, p, r in per_ep:
        print(f"  {eid}: precision {p:.2f} recall {r:.2f}")
    with open(RESULTS, "a") as f:
        f.write(f"\n## Host identification vs hand labels ({args.variant})\n\n{n_ep} episodes, duration-weighted: "
                f"precision {prec:.3f}, recall {rec:.3f}.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--per-show", type=int, default=10)
    s.add_argument("--hand-label", type=int, default=20); s.add_argument("--seed", type=int, default=20261026)
    c = sub.add_parser("compare"); c.add_argument("--reference", default="stock")
    c.add_argument("--out", default=None); c.add_argument("--append", action="store_true")
    c.add_argument("--episodes-file", default=None, help="restrict the comparison to these show_id,episode_id lines")
    sc = sub.add_parser("score"); sc.add_argument("--variant", default="default")
    args = ap.parse_args()
    {"sample": cmd_sample, "compare": cmd_compare, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
