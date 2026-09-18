#!/usr/bin/env python3
"""Layer 1: deterministic lexical measures of outrage language.

    ./measure_outrage_lexical.py --run      # -> data/output/outrage_lexical.csv
    ./measure_outrage_lexical.py --report   # within- vs between-show variance
    ./measure_outrage_lexical.py --kwic vulgarity --n 15

Why this layer exists before any model
--------------------------------------
Every measure here is a regex over the transcript. Deterministic, auditable,
reproducible in five years, and free. If outrage markers show no within-show
variation across quarters, no LLM will find any either, and the fixed-effects
design has nothing to explain. This is the cheap version of the pilot gate.

What is measured, and what is NOT
---------------------------------
Sobieraj & Berry (2011) code 13 outrage types from talk radio, blogs and cable
news. They split three ways under text analysis:

  lexically detectable   obscene language, emotional language (intensifiers),
                         ideologically extremizing language (absolutes),
                         part of insulting language
  needs a model          mockery, misrepresentative exaggeration, character
                         assassination, conflagration, slippery slope, belittling
  NOT MEASURABLE HERE    emotional display -- shouting, vocal escalation. They
                         coded AUDIO. This corpus is text. One of the thirteen
                         is gone in principle and should be reported as such,
                         not quietly dropped.

`speech_rate_wps` is a partial recovery of that last one: segments carry
start/end timestamps, so words-per-second is computable without audio. Faster
speech is a weak but real correlate of agitation. It is not loudness.

Out-group terms follow Rathje, Van Bavel & van der Linden (2021), who count
out-group references per post with the poster's side fixed EXTERNALLY. Here the
side comes from Brookings partisan lean, not from DIME. (An earlier note here said
DIME had failed validation against Brookings; that comparison used fuzzy Brookings
matches that mapped shows to the wrong label. On exact matches DIME separates the
Brookings sides by 1.4 SD and agrees with the LLM side score at r = 0.68 on 112
shows. Only exact-title Brookings matches are used now, 17 Sept 2026, so the
out-group rate is defined for those shows only.)

The lexicon is deliberately high-precision and low-recall
---------------------------------------------------------
Bare "left" and "right" are excluded. In spoken conversation "right" is a
discourse marker roughly constantly ("right, so...", "that's right"), and
including it would swamp the measure with backchannel. This project has already
been bitten seven times by raw-string matching -- Volts, Destiny, Time, Reason,
Progressive Insurance, chart ranks that were HTML entities, and a Trends panel
where a fifth of the "usable" shows were measuring English phrases. Missing some
true positives is the cheaper error. Use --kwic to read actual matches before
believing any number here.
"""
import argparse, csv, glob, hashlib, json, os, re, statistics, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repository root (this script lives one folder down)
TRANSCRIPTS = ROOT / "data/transcripts"
EPISODES = ROOT / "data/output/episodes.csv.gz"
LEAN = ROOT / "data/output/lean_validation.csv"
OUT = ROOT / "data/output/outrage_lexical.csv"

# (?: ) throughout: capturing groups make re.findall return tuples, which
# silently breaks any per-token inspection downstream.
PAT = {
 "vulgarity": r"\b(?:fuck\w*|shit\w*|goddamn|damn|asshole\w*|bastard\w*|bitch\w*|"
              r"piss(?:ed|ing)?|bullshit|dumbass|jackass)\b",
 "intensifier": r"\b(?:absolutely|completely|utterly|totally|literally|insane|insanity|"
                r"outrageous|outrage|ridiculous|absurd|disgusting|disgraceful|shameful|"
                r"pathetic|appalling|unbelievable|horrific)\b",
 "absolute": r"\b(?:never|always|every single|nobody|no one ever|everyone knows|"
             r"not one|zero chance|complete(?:ly)? failure)\b",
 "contempt": r"\b(?:so-?called|supposedly|allegedly|apparently|if you can call|"
             r"quote unquote|whatever that means)\b",
 "insult": r"\b(?:idiot\w*|moron\w*|clown\w*|liar\w*|lying|corrupt|crook\w*|thug\w*|"
           r"coward\w*|grifter\w*|hack\w*|stupid|fraud\w*|traitor\w*|scum\w*)\b",
 "catastrophe": r"\b(?:destroy\w*|destruction|collapse|end of (?:democracy|america|"
                r"the republic)|existential threat|civil war|dictatorship|tyranny|"
                r"authoritarian)\b",
}
# High-precision party references only. No bare "left"/"right".
SIDE = {
 "left": r"\b(?:democrat|democrats|democratic party|liberal|liberals|progressives|"
         r"biden|kamala|pelosi|schumer|aoc|squad)\b",
 "right": r"\b(?:republican|republicans|gop|conservative|conservatives|trump|maga|"
          r"desantis|mcconnell|maga republicans)\b",
}
RX = {k: re.compile(v, re.I) for k, v in {**PAT, **SIDE}.items()}


def episode_meta():
    eps = pd.read_csv(EPISODES, compression="gzip", low_memory=False)
    eps["pub_date"] = pd.to_datetime(eps.pub_date, utc=True, errors="coerce", format="mixed")
    eps = eps[eps.pub_date.notna() & (eps.pub_date > "2000-01-01")]
    eps = eps[eps.audio_url.notna() & (eps.audio_url != "")].copy()
    eps["episode_id"] = eps.audio_url.map(lambda u: hashlib.md5(u.encode()).hexdigest()[:16])
    eps = eps.drop_duplicates(subset=["collection_id", "episode_id"])
    eps["q"] = eps.pub_date.dt.tz_localize(None).dt.to_period("Q").astype(str)
    return eps.set_index("episode_id")[["collection_id", "show_name", "q"]].to_dict("index")


def show_side():
    """Brookings partisan lean -> {show_id: 'left'|'right'}. DIME is not used."""
    out = {}
    if not LEAN.exists():
        return out
    for r in csv.DictReader(open(LEAN)):
        # Only exact-title matches carry a trustworthy label; the fuzzy "review" and
        # "ambiguous" rows map corpus shows to the wrong Brookings show (17 Sept 2026).
        if (r.get("match_status") or "").strip() != "matched":
            continue
        v = (r.get("brookings_partisan_leaning") or "").strip()
        if v == "More Conservative":
            out[str(r["collection_id"])] = "right"
        elif v == "More Liberal":
            out[str(r["collection_id"])] = "left"
    return out


def cmd_run(args):
    meta = episode_meta()
    side = show_side()
    files = glob.glob(str(TRANSCRIPTS / "*/*.json"))
    print(f"{len(files):,} transcripts; Brookings side known for {len(side)} shows\n")
    rows = []
    for i, p in enumerate(files, 1):
        eid = os.path.basename(p).split("_", 1)[0]
        m = meta.get(eid)
        if not m:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        segs = d.get("segments") or []
        text = " ".join(s.get("text", "") for s in segs)
        nw = len(text.split())
        if nw < 200:
            continue
        dur = 0.0
        for s in segs:
            try: dur += float(s.get("end", 0)) - float(s.get("start", 0))
            except Exception: pass
        rec = {"episode_id": eid, "show_id": str(m["collection_id"]),
               "show_name": m["show_name"], "q": m["q"], "n_words": nw,
               "speech_rate_wps": round(nw / dur, 3) if dur > 30 else ""}
        for k, rx in RX.items():
            rec[f"{k}_per10k"] = round(len(rx.findall(text)) / nw * 10000, 3)
        s = side.get(str(m["collection_id"]))
        if s:
            og, ig = ("right", "left") if s == "left" else ("left", "right")
            rec["show_side"] = s
            rec["outgroup_per10k"] = rec[f"{og}_per10k"]
            rec["ingroup_per10k"] = rec[f"{ig}_per10k"]
        else:
            rec["show_side"] = ""; rec["outgroup_per10k"] = ""; rec["ingroup_per10k"] = ""
        rows.append(rec)
        if i % 4000 == 0:
            print(f"  {i:,}/{len(files):,}  {len(rows):,} usable", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n{OUT}\n  {len(rows):,} episodes, {len({r['show_id'] for r in rows})} shows")
    cmd_report(args)


def cmd_report(args):
    d = pd.read_csv(OUT)
    d = d[d.q >= "2018Q1"]
    cols = [c for c in d.columns if c.endswith("_per10k")] + ["speech_rate_wps"]
    cell = d.groupby(["show_id", "q"]).agg(
        {**{c: "mean" for c in cols}, "n_words": "size"}).rename(columns={"n_words": "n_eps"})
    cell = cell[cell.n_eps >= 3]
    print(f"\nSHOW-QUARTER PANEL (2018+, >=3 episodes): {len(cell):,} cells, "
          f"{cell.index.get_level_values(0).nunique()} shows\n")
    print(f"{'measure':<22}{'mean':>9}{'within sd':>11}{'between sd':>12}{'ratio':>8}")
    for c in cols:
        v = cell[c].dropna()
        if v.empty: continue
        g = cell[c].groupby(level=0)
        w = g.std().median(); b = g.mean().std()
        if not b or pd.isna(b) or pd.isna(w): continue
        print(f"  {c:<20}{v.mean():>9.2f}{w:>11.2f}{b:>12.2f}{w/b:>8.2f}")
    print("\n  ratio = median within-show sd / between-show sd.")
    print("  Show fixed effects remove the between-show part. A ratio near zero")
    print("  means the measure is a fixed show trait and the design cannot use it;")
    print("  the static DIME ideology score has a ratio of exactly 0.")


def cmd_kwic(args):
    """Read actual matches. Every wrong number in this project came from not doing this."""
    rx = RX[args.kwic]
    files = glob.glob(str(TRANSCRIPTS / "*/*.json"))[:400]
    shown = 0
    for p in files:
        try: d = json.load(open(p))
        except Exception: continue
        t = re.sub(r"\s+", " ", " ".join(s.get("text", "") for s in d.get("segments", [])))
        for m in rx.finditer(t):
            print(f"  ...{t[max(0,m.start()-70):m.end()+70]}...")
            shown += 1
            if shown >= args.n: return


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--kwic", default=None, choices=list(PAT) + list(SIDE))
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    if a.kwic: cmd_kwic(a)
    elif a.run: cmd_run(a)
    elif a.report: cmd_report(a)
    else: ap.print_help()


if __name__ == "__main__":
    main()
