"""Freeze the hand annotations into data/output/adspan/gold_spans.json.

PROVENANCE -- read this before trusting any Phase A number
----------------------------------------------------------
These spans were annotated by reading all 26 windows (886 sentences) in full,
by the agent running this session, NOT by a human. The task brief asks for
human annotation; that constraint is NOT met and is reported rather than
papered over. What this does and does not buy:

  * The annotator (Claude) is a different model from the detector under test
    (gpt-4o-mini / gpt-4.1-nano), so Phase A is not a model grading itself on
    its own output -- but it is still model-vs-model, and a bias the two share
    would be invisible here.
  * Every window is stored with its full sentence text alongside the spans, so
    a human can audit or correct any judgment cheaply. Re-running Phase A after
    an edit costs a few cents.

The annotation rule actually applied
------------------------------------
  ad_sponsor       third-party paid placement: products, services, and OTHER
                   organizations' podcasts (Alliant University, Chevron's
                   "Energy Tailgate", Mint Mobile).
  meta_boilerplate the show's or its own network's structural content: intros,
                   outros, credits, subscribe/review/donate asks, segment
                   teases and break bumpers, and cross-promos for SIBLING shows
                   on the same network (Crooked's "Pod Save the UK", NPR's
                   "Embedded", MS NOW's lineup).
  content          everything else, including political/economic discussion
                   that happens to name a product category. A guest's book
                   discussed editorially is content; a paid read for that book
                   would be ad_sponsor.

Two judgment calls that shape the metrics, stated up front:

  1. PRECISION OVER RECALL AT MIXED SENTENCES. Whisper occasionally merges an
     ad tail and the return to content into ONE sentence (W4 s29, W13 s28).
     Since excision is sentence-granular, marking those would delete real
     political discourse. They are left UNMARKED -- deliberately conceding
     recall so the gate measures the error that actually matters.
  2. AD-ADJACENT BANTER IS NOT AUTOMATICALLY AD. W5 s0-3 (comedic setup that
     may or may not be scripted into the Quince read) is left unmarked for the
     same reason.

    .venv/bin/python -m nlp.adspan_gold_build
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pipeline import config as pipeline_config

OUT_DIR = pipeline_config.OUTPUT_DIR / "adspan"

A = "ad_sponsor"
M = "meta_boilerplate"

# window index (position in gold_candidates.json) -> list of (start, end, label)
ANNOTATIONS: dict[int, list[tuple[int, int, str]]] = {
    # American Prestige -- Quince read, then hard cut to Israel/Palestine at s17.
    0: [(0, 16, A)],
    # Joy Reid -- shopredbag/Vontale frames w/ code "joy", then on-camera
    # modeling and packaging talk, which is still the promo segment, not
    # discourse. s35 is a segment transition.
    1: [(0, 34, A), (35, 35, M)],
    # Lincoln Project -- s0-4 is real Biden/Trump-age content; Groons read runs
    # s5 ("summer routines live or die...") to the end of the window.
    2: [(5, 35, A)],
    # Verdict w/ Ted Cruz -- Rough Greens dog supplement, s3-32; s34 returns to
    # the Kimmel/FCC discussion.
    3: [(3, 32, A)],
    # The Wright Report -- break bumper, then Jase.com. s29 is an ASR run-on
    # holding BOTH the ad tail and the return to China/Venezuela coverage:
    # deliberately UNMARKED (see docstring judgment call 1).
    4: [(14, 15, M), (16, 28, A)],
    # The Skepticrat -- Quince again, as a two-host comedy bit. s0-3 setup left
    # unmarked (judgment call 2); read + post-read banter s4-35.
    5: [(4, 35, A)],
    # Hacks On Tap -- Helix mattress read already in progress at window start;
    # s29 cuts to the Putin summit.
    6: [(0, 28, A)],
    # Focus Group -- s0-27 is a guest discussing his late daughter and his
    # wife's book on grief. Genuine discourse that superficially resembles a
    # book promo; a detector that cuts this is doing real damage.
    7: [(28, 33, M), (34, 35, A)],
    # Al Franken -- outro + credits, then the Alliant University insert.
    8: [(13, 16, M), (17, 26, A)],
    # Politics War Room -- same Alliant insert dropped mid-episode, s1-9, with
    # content on both sides.
    9: [(1, 9, A)],
    # Pod Save the World -- Alliant insert s9-16, then Crooked's own sibling-show
    # promo (Pod Save the UK) s17 to window end.
    10: [(9, 16, A), (17, 35, M)],
    # POLITICO Energy -- outro/credits s4-10, then "Energy Tailgate, a podcast
    # from Chevron" (third-party) s11-15.
    11: [(4, 10, M), (11, 15, A)],
    # Federalist Radio Hour -- like/subscribe/email intro s0-4. The guest intro
    # at s5 mentions her book editorially: content, not marked.
    12: [(0, 4, M)],
    # Lars Larson -- produced cold-open montage s0-16 (the political soundbites
    # inside it are branding, not discourse); content s17-21; station/contact
    # boilerplate s22-27. s28 merges the poll ask with the LA story: unmarked.
    13: [(0, 16, M), (22, 27, M)],
    # On the Media -- credits roll s3-10. s0-2 (guest sign-off + bio) and s11-14
    # (a teaser clip) left unmarked.
    14: [(3, 10, M)],
    # Peter McCormack -- 29 sentences of substantive Reform UK / Brexit talk,
    # then a genuine Casa bitcoin-security sponsor read s29-35. The hard case in
    # both directions.
    15: [(29, 35, A)],
    # RealClearPolitics -- mortgage/bank-fraud prosecutions. Trips the
    # "mortgage" marker; 100% content.
    16: [],
    # This Week -- Obamacare premium fight (health insurance, mortgage rates:
    # all content), plus a two-sentence segment tease at s25-26.
    17: [(25, 26, M)],
    # American Thought Leaders -- Medicare/Medicaid fraud. Content throughout.
    18: [],
    # Russian Roulette -- Russian central bank, key rate, inflation. Content.
    19: [],
    # Kyle Kulinski -- the canonical gold-as-content case: "golden balloon",
    # "gold shop", "golden statue", "selling gold merchandise", all political
    # commentary on a Trump Gaza AI video. Zero ad.
    20: [],
    # What the Hell Is Going On -- Mamdani / Democratic Party. Content.
    21: [],
    # Trump's Terms (NPR) -- heavily front-loaded: Embedded house promo s0-6,
    # show intro s7-14, Mint Mobile s15-19 (third-party), NPR donation appeal
    # s20-23, NPR Politics Podcast promo s24-31. Content starts at s32.
    22: [(0, 14, M), (15, 19, A), (20, 31, M)],
    # Deadline: White House -- tease + break bumper + MS NOW network promo,
    # s10-20. The tease carries real political substance but is structurally
    # meta ("we'll get to that later in the hour").
    23: [(10, 20, M)],
    # The Rest Is Politics: US -- midterms. Content.
    24: [],
    # Pod Force One -- Sen. Kennedy on debt/inflation/appropriations. Content.
    25: [],
}


def main() -> None:
    candidates = json.loads((OUT_DIR / "gold_candidates.json").read_text())
    if len(candidates) != len(ANNOTATIONS):
        raise SystemExit(f"candidates={len(candidates)} but annotations={len(ANNOTATIONS)}")

    gold = []
    label_counts: Counter = Counter()
    n_sent_total = 0
    for i, c in enumerate(candidates):
        n = len(c["sentences"])
        spans = []
        covered: set[int] = set()
        for start, end, label in ANNOTATIONS[i]:
            if not (0 <= start <= end < n):
                raise SystemExit(f"window {i} ({c['window_id']}): span {start}-{end} "
                                  f"out of range for {n} sentences")
            idx = set(range(start, end + 1))
            if idx & covered:
                raise SystemExit(f"window {i}: span {start}-{end} overlaps an earlier span")
            covered |= idx
            spans.append({"start": start, "end": end, "label": label})
            label_counts[label] += len(idx)
        n_sent_total += n
        label_counts["content"] += n - len(covered)
        gold.append({
            "window_id": c["window_id"],
            "episode_id": c["episode_id"],
            "show_id": c["show_id"],
            "show_name": c["show_name"],
            "episode_title": c["episode_title"],
            "stratum": c["stratum"],
            "offset": c["offset"],
            "n_sentences": n,
            "sentences": c["sentences"],
            "spans": spans,
            "annotator": "agent (Claude, this session) -- NOT human; see module docstring",
        })

    out = OUT_DIR / "gold_spans.json"
    out.write_text(json.dumps(gold, indent=2))
    print(f"[gold] {len(gold)} windows, {n_sent_total} sentences -> {out}")
    print(f"[gold] sentence labels: {dict(label_counts)}")
    non_content = label_counts[A] + label_counts[M]
    print(f"[gold] ad+meta = {non_content}/{n_sent_total} ({non_content / n_sent_total:.1%})")

    by_stratum: Counter = Counter()
    for g in gold:
        by_stratum[g["stratum"]] += 1
    print(f"[gold] windows per stratum: {dict(by_stratum)}")
    n_clean = sum(1 for g in gold if not g["spans"])
    print(f"[gold] windows with zero ad/meta (pure content): {n_clean}")


if __name__ == "__main__":
    main()
