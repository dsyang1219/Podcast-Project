"""Word-level breakdown of the R&P populism rate per show (which stems carry it). Exploratory record F.3.4. Writes inputs/rp_words_by_show.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import gzip, csv, re, glob, os, sys, random, collections, pandas as pd, importlib.util

# Fixed seed for the per-show passage sample; run from the repo root so relative corpus paths resolve.
random.seed(3)
os.chdir(ROOT)
sys.path.insert(0, ".")
SP = str(INPUTS)
shows = set(pd.read_pickle(INPUTS / "pooled_resp_features.pkl").show_id.unique())  # only shows that appear in the audience matches
# Same R&P dictionary as rp_antimedia.py; STEM maps a matched word back to its dictionary stem label.
RP = re.compile(
    r"\b(elit\w*|consensus\w*|undemocratic\w*|referend\w*|corrupt\w*|propagand\w*|politici\w*|\w*deceit\w*|\w*deceiv\w*|\w*betray\w*|shame\w*|scandal\w*|truth\w*|dishonest\w*|establishm\w*|ruling\w*)\b",
    re.I,
)
STEM = {
    "elit": "elite",
    "consensus": "consensus",
    "undemocratic": "undemocratic",
    "referend": "referendum",
    "corrupt": "corrupt",
    "propagand": "propaganda",
    "politici": "politician",
    "deceit": "deceit",
    "deceiv": "deceive",
    "betray": "betray",
    "shame": "shame",
    "scandal": "scandal",
    "truth": "truth",
    "dishonest": "dishonest",
    "establishm": "establishment",
    "ruling": "ruling",
}


# Map a matched word to its stem label (first stem found as a substring); "other" should not occur.
def stem(w):
    w = w.lower()
    for k, v in STEM.items():
        if k in w:
            return v
    return "other"


TOK = re.compile(r"[a-z']+")
# Collect passages per show from the scoring chunks (matched shows only, passages of >= 120 characters).
by = collections.defaultdict(list)
with gzip.open(DATA / "output/scoring_chunks.csv.gz", "rt", newline="") as fh:
    for r in csv.DictReader(fh):
        if r["collection_id"] in shows and len(r["text"]) >= 120:
            by[r["collection_id"]].append(r["text"])
# Some matched shows (hold-out shows and one extra id) are not in scoring_chunks.csv.gz; re-chunk their
# raw transcripts with the corpus's own chunker so they are scored on the same footing.
spec = importlib.util.spec_from_file_location("bsc", "step4_ideology/build_scoring_chunks.py")
bsc = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(bsc)
except SystemExit:  # the chunker script may call sys.exit() when imported; ignore it
    pass
for p in glob.glob("data/transcripts/HOLDOUT_*/*.json") + glob.glob("data/transcripts/402306412/*.json"):
    by[p.split("/")[2]] += [t for t in bsc.split_750(bsc.transcript_text(p)) if len(t) >= 120]  # path segment 2 = show folder name = show id
rows = []
# Per show: sample up to 600 passages (for speed), count words and hits by stem, then rates per 10k words.
for s, texts in by.items():
    texts = random.sample(texts, min(600, len(texts)))  # cap at 600 passages per show
    nw = 0
    c = collections.Counter()
    for t in texts:
        nw += len(TOK.findall(t.lower()))
        for m in RP.findall(t):
            c[stem(m)] += 1
    rows.append(
        {"show_id": s, "nw": nw, "passages": len(texts)}
        | {k + "_rate": v / nw * 1e4 for k, v in c.items()}  # one column per stem, e.g. corrupt_rate
    )
pd.DataFrame(rows).fillna(0).to_csv(INPUTS / "rp_words_by_show.csv", index=False)  # stems never seen in a show -> 0
print("done", len(rows))
