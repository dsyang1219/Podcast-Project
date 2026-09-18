"""Split second person into guest-directed vs listener-directed cues (questions, named persons, 'you know', deontic) on holdout passages and a corpus sample; the interrogative-you check in paper section 3.2. Writes inputs/corpus_you_split.csv and inputs/holdout_you_split.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS

"""Split second-person into guest-directed vs listener-directed cues, on holdout passages AND a corpus sample. No outcomes."""
import sys, os, glob, re, random, collections, numpy as np, pandas as pd, spacy, gzip, csv

# Run from the repo root; fixed seed for the corpus sampling below.
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
random.seed(5)
import importlib.util

# Reuse the corpus chunker (split_750, transcript_text) from step4_ideology/build_scoring_chunks.py.
spec = importlib.util.spec_from_file_location("bsc", "step4_ideology/build_scoring_chunks.py")
bsc = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(bsc)
except SystemExit:
    pass
nlp = spacy.load("en_core_web_sm")  # NER on, for person names
TOK = re.compile(r"[a-z']+")  # word tokenizer for nw
DEONT = {"need", "have", "should", "must", "gotta", "got", "ought", "better", "want", "can't", "cannot"}  # words after 'you' that mark a directive reading: you need / should / gotta ...
# Explicit plural or audience address, which can only be aimed at listeners, not a guest.
PLURAL = re.compile(
    r"\b(you guys|you all|y'all|everybody|everyone|folks|listeners?|audience|if you're (?:listening|watching)|those of you)\b",
    re.I,
)


# Count second-person cues per label (a show), then convert to rates per 10k words and shares of all 'you'.
def feats(texts, labels):
    per = collections.defaultdict(collections.Counter)
    for lab, doc in zip(labels, nlp.pipe(texts, batch_size=400, n_process=6)):
        n = len(TOK.findall(doc.text.lower()))  # word count
        per[lab]["nw"] += n
        per[lab]["plural"] += len(PLURAL.findall(doc.text))  # audience-address phrases
        for sent in doc.sents:
            q = sent.text.strip().endswith("?")  # sentence is a question
            names = any(e.label_ == "PERSON" for e in sent.ents)  # sentence names a person (guest-directed cue)
            toks = list(sent)
            for k, t in enumerate(toks):
                if t.pos_ == "PRON" and "Person=2" in str(t.morph):  # any second-person pronoun
                    per[lab]["you_all"] += 1
                    nxt = toks[k + 1].text.lower() if k + 1 < len(toks) else ""
                    if nxt == "know":
                        per[lab]["you_know"] += 1  # 'you know' filler: counted, then dropped from the split
                        continue
                    # Mutually exclusive split: question > named person > declarative.
                    if q:
                        per[lab]["you_question"] += 1
                    elif names:
                        per[lab]["you_named"] += 1
                    else:
                        per[lab]["you_declar"] += 1
                        # Deontic subset of declarative 'you': followed by need/should/must ... or 'you're going to / gonna'.
                        if nxt in DEONT or (
                            nxt in ("'re", "are")
                            and k + 2 < len(toks)
                            and toks[k + 2].text.lower() in ("going", "gonna")
                        ):
                            per[lab]["you_deontic"] += 1
    D = pd.DataFrame.from_dict(per, orient="index").fillna(0)
    for c in [c for c in D.columns if c != "nw"]:
        D[c + "_rate"] = D[c] / D.nw * 1e4  # per 10k words
    D["q_share"] = D.you_question / D.you_all.replace(0, np.nan)  # share of all 'you' that sits in questions
    D["decl_share"] = D.you_declar / D.you_all.replace(0, np.nan)
    return D


# --- holdout ---
rows = []
for p in sorted(glob.glob("data/transcripts/HOLDOUT_*/*.json")):
    show = p.split("/")[2]
    for t in bsc.split_750(bsc.transcript_text(p)):  # same 750-char passages as the scoring pipeline
        if len(t) >= 120:  # drop fragments under 120 characters
            rows.append((show, t))
H = feats([t for _, t in rows], [s for s, _ in rows])
# Holdout table, most second-person show first.
print("=== HOLDOUT: where does each show's 'you' live? (rates per 10k words; shares of all 'you') ===")
print(
    H[
        [
            "you_all_rate",
            "you_question_rate",
            "you_named_rate",
            "you_declar_rate",
            "you_deontic_rate",
            "plural_rate",
            "q_share",
            "decl_share",
        ]
    ]
    .round(2)
    .sort_values("you_all_rate", ascending=False)
    .to_string()
)
# --- corpus sample: 150 passages/show for the 6 landmark shows + 40 random others ---
by = collections.defaultdict(list)
with gzip.open(DATA / "output/scoring_chunks.csv.gz", "rt", newline="") as fh:
    for r in csv.DictReader(fh):
        if len(r["text"]) >= 120:
            by[r["collection_id"]].append(r["text"])  # all corpus passages grouped by show (collection_id)
# The 194-show reference table supplies show names, dir_z, side and DIME scores.
ref = pd.read_csv(HANDOFF / "prereg/dirz_reference_scale.csv")
ref["show_id"] = ref.show_id.astype(str)
# Landmark shows: ten named shows plus one by id (1043245989), chosen to span the dir_z range.
land = {"1043245989"} | set(
    ref[
        ref.show.isin(
            [
                "Louder with Crowder",
                "The MeidasTouch Podcast",
                "Pod Save America",
                "Letters from an American",
                "The Bulwark Podcast",
                "Life, Liberty & Levin Podcast",
                "The Young Turks",
                "Breaking Points with Krystal and Saagar",
                "Tangle",
                "The David Pakman Show",
            ]
        )
    ].show_id
)
pick = list(land) + random.sample([s for s in by if s not in land], 40)  # landmarks + 40 random other corpus shows (50 total)
rows = [(s, t) for s in pick if s in by for t in random.sample(by[s], min(150, len(by[s])))]  # up to 150 passages per show
# Same feature split on the corpus sample, joined to show metadata.
C = (
    feats([t for _, t in rows], [s for s, _ in rows])
    .reset_index()
    .rename(columns={"index": "show_id"})
    .merge(ref[["show_id", "show", "dir_z", "side", "avg_host_cfscore"]], on="show_id")
)
# Top 12 most directive corpus shows and where their 'you' lives.
print("\n=== CORPUS sample (50 shows x <=150 passages): same split ===")
print(
    C.sort_values("dir_z", ascending=False)[
        [
            "show",
            "dir_z",
            "you_all_rate",
            "you_question_rate",
            "you_declar_rate",
            "you_deontic_rate",
            "q_share",
        ]
    ]
    .head(12)
    .round(2)
    .to_string(index=False)
)
# Key contrast: Rogan's question share vs the corpus median (if the Rogan folder is present).
print(
    "  ... corpus median q_share =",
    round(C.q_share.median(), 3),
    "| holdout Rogan q_share =",
    round(H.loc["HOLDOUT_rogan", "q_share"], 3) if "HOLDOUT_rogan" in H.index else "n/a",
)
print("\n=== does the refined measure keep the corpus validation? (50-show sample, vs DIME) ===")
# Does each 'you' variant keep the ideological validation? Correlate show rates with the DIME host score and with dir_z.
c = C.dropna(subset=["avg_host_cfscore"])
for x, lab in (
    ("you_all_rate", "all 'you' (current)"),
    ("you_declar_rate", "declarative 'you' (drop questions, names, 'you know')"),
    ("you_deontic_rate", "deontic 'you'"),
    ("you_question_rate", "interrogative 'you' (guest-directed cue)"),
):
    print(
        f"  {lab:<58} vs DIME r={np.corrcoef(c[x],c.avg_host_cfscore)[0,1]:+.3f}   vs dir_z r={np.corrcoef(C[x],C.dir_z)[0,1]:+.3f}   (n={len(c)})"
    )
# Save both tables; corpus_you_split.csv is the reference distribution score_holdout_final.py uses for the declarative-you z.
H.to_csv(INPUTS / "holdout_you_split.csv")
C.to_csv(INPUTS / "corpus_you_split.csv", index=False)
