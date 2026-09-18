"""Rooduijn & Pauwels (2011) populism dictionary and Fawzi-style anti-media references scored over the corpus. Paper section 2.3. Writes inputs/show_rp_antimedia.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import gzip, csv, re, pandas as pd, numpy as np, warnings

warnings.filterwarnings("ignore")
csv.field_size_limit(10_000_000)
# Rooduijn & Pauwels (2011), English populism dictionary, exact stems (Table A1). Anti-elitist core.
# \w* after each stem matches every inflection (elite, elites, elitist; corrupt, corruption, ...);
# stems with a leading \w* (deceit, deceiv, betray) also catch prefixed forms.
RP = r"\b(elit\w*|consensus\w*|undemocratic\w*|referend\w*|corrupt\w*|propagand\w*|politici\w*|\w*deceit\w*|\w*deceiv\w*|\w*betray\w*|shame\w*|scandal\w*|truth\w*|dishonest\w*|establishm\w*|ruling\w*)\b"
# people-centrism, as used in R&P replications (Pauwels 2011; Rooduijn 2014): "the people" and equivalents
PC = r"\b(the people|the american people|ordinary (?:people|americans|citizens)|citizens|taxpayers|voters|the public|everyday americans|working (?:people|families))\b"
# anti-media populism (Fawzi 2019): references to the media as an out-group, scored for negativity below
MEDIA = r"\b(the media|mainstream media|news media|the press|journalists?|reporters?|msm|legacy media|corporate media|fake news|cnn|msnbc|fox news|new york times|washington post|the times|the post|npr|abc news|cbs news|nbc news)\b"
# Compile case-insensitively; TOK is the word tokenizer used for the per-10k-word denominator.
R = re.compile(RP, re.I)
P = re.compile(PC, re.I)
M = re.compile(MEDIA, re.I)
TOK = re.compile(r"[a-z']+")
# VADER gives a "neg" share (0-1) per passage; it weights each media reference by how negative the
# surrounding passage is, so neutral mentions of CNN count for little.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

V = SentimentIntensityAnalyzer()
rows = []
# Stream all passages once. Per passage: show id, word count, R&P hits, people-centrism hits, media
# references, and media references x passage negativity (the Fawzi-style anti-media score).
with gzip.open(DATA / "output/scoring_chunks.csv.gz", "rt", newline="") as fh:
    for r in csv.DictReader(fh):
        t = r["text"]
        n = len(TOK.findall(t.lower()))
        if n < 40:  # skip very short passages
            continue
        m = len(M.findall(t))
        neg = V.polarity_scores(t)["neg"] if m else 0.0  # only run VADER when the passage mentions the media (it is slow)
        rows.append((r["collection_id"], n, len(R.findall(t)), len(P.findall(t)), m, neg * m))
D = pd.DataFrame(rows, columns=["show_id", "nw", "rp", "pc", "media", "media_negw"])
# Sum within show, so show rates are automatically weighted by passage word count.
S = (
    D.groupby("show_id")
    .agg(
        nw=("nw", "sum"),
        rp=("rp", "sum"),
        pc=("pc", "sum"),
        media=("media", "sum"),
        media_negw=("media_negw", "sum"),
    )
    .reset_index()
)
# Rates per 10,000 words.
S["rp_rate"] = S.rp / S.nw * 1e4
S["pc_rate"] = S.pc / S.nw * 1e4
S["media_rate"] = S.media / S.nw * 1e4
S["media_neg"] = S.media_negw / S.media.replace(
    0, np.nan
)  # mean VADER negativity of media-referencing passages, weighted by references
S["antimedia_rate"] = (
    S.media_negw / S.nw * 1e4
)  # negative media references per 10k words (Fawzi-style anti-media populism)
S["show_id"] = S.show_id.astype(str)  # string ids for merging with directive_final.csv
S.to_csv(INPUTS / "show_rp_antimedia.csv", index=False)
print(
    f"{len(D):,} passages, {len(S)} shows. per-10k: R&P populism median {S.rp_rate.median():.1f}, people-centrism {S.pc_rate.median():.1f}, media refs {S.media_rate.median():.1f}, anti-media {S.antimedia_rate.median():.2f}"
)
