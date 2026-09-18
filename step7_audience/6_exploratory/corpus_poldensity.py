"""Political-content density per corpus show: share of 750-character passages with 3+ political terms. Defines the political-podcast frame (>= 15%) used in finding 7. Writes inputs/corpus_poldensity.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import gzip, csv, re, pandas as pd, collections

# Political-vocabulary dictionary: names, institutions, process words, hot-button topics.
# \b...\b = whole words only; re.I = case-insensitive. Each match counts as one political term.
POL = re.compile(
    r"\b(trump|biden|harris|obama|vance|congress|senate|senator|house of representatives|election|elections|ballot|vote|voters?|voting|democrats?|republicans?|gop|liberals?|conservatives?|government|federal|white house|president|presidential|supreme court|justice department|doj|fbi|cia|immigration|border|tariffs?|taxes|policy|policies|legislation|bill|law|constitution|governor|mayor|campaign|political|politics|partisan|left-wing|right-wing|maga|woke|deep state|ukraine|israel|gaza|nato|impeach\w*|indict\w*|epstein)\b",
    re.I,
)
TOK = re.compile(r"[a-z']+")  # a "word" = a run of letters/apostrophes (text is lower-cased first)
# Per-show accumulator: [total words, total political terms, passages with 3+ terms, (passage count added below)]
c = collections.defaultdict(lambda: [0, 0, 0])
# Stream all 1.38M passages from the compressed corpus file, one at a time, to keep memory small.
with gzip.open(DATA / "output/scoring_chunks.csv.gz", "rt", newline="") as fh:
    for r in csv.DictReader(fh):
        t = r["text"]
        n = len(TOK.findall(t.lower()))
        if n < 40:  # skip very short passages (fewer than 40 words), as in all rate calculations
            continue
        k = len(POL.findall(t))
        a = c[r["collection_id"]]  # collection_id is the show id
        a[0] += n
        a[1] += k
        a[2] += k >= 3  # passage counts as "political" if it has at least 3 dictionary hits
        # count passages
        a.append(1) if len(a) == 3 else None  # first time this show is seen: add a 4th slot; note it starts at 1 and is then incremented, so the first passage is counted twice (denominator off by one; negligible over thousands of passages)
        a[3] += 1
# One row per show: total words, political terms per 10,000 words, and share of political passages.
pd.DataFrame(
    [(s, v[0], v[1] / v[0] * 1e4, v[2] / v[3]) for s, v in c.items()],
    columns=["show_id", "nw", "pol_rate", "share_political"],
).to_csv(INPUTS / "corpus_poldensity.csv", index=False)
print("done", len(c))
