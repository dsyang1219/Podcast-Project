"""The same imperative / second-person decomposition for the out-of-frame shows and The Majority Report. Writes inputs/holdout_imp_decomp.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import glob, importlib.util, sys, os, collections, re, spacy, pandas as pd

# Run from the repo root so the relative transcript and corpus paths below resolve.
os.chdir(ROOT)
sys.path.insert(0, ".")
SP = str(INPUTS)
# Same verb buckets as imp_decomp.py (kept identical so the holdout rates are comparable).
EPIST = {
    "understand",
    "realize",
    "know",
    "think",
    "remember",
    "consider",
    "imagine",
    "notice",
    "recognize",
    "believe",
    "ask",
    "wonder",
    "forget",
    "bear",
    "keep",
    "trust",
    "picture",
    "assume",
    "note",
    "realise",
}
ATTN = {
    "look",
    "listen",
    "see",
    "watch",
    "hear",
    "check",
    "wait",
    "hold",
    "stop",
    "hang",
    "guess",
    "mark",
    "behold",
}
ACTION = {
    "go",
    "get",
    "call",
    "vote",
    "donate",
    "subscribe",
    "share",
    "sign",
    "join",
    "buy",
    "support",
    "fight",
    "stand",
    "take",
    "make",
    "do",
    "give",
    "send",
    "read",
    "visit",
    "use",
    "try",
    "help",
    "tell",
    "write",
    "download",
    "follow",
    "click",
    "order",
    "pay",
    "come",
    "let",
    "put",
    "start",
    "turn",
    "show",
    "find",
    "bring",
    "leave",
    "pick",
    "stay",
    "run",
    "push",
    "demand",
    "protest",
    "register",
    "contact",
}
# Load the corpus chunker (split_750, transcript_text) from step4_ideology/build_scoring_chunks.py without running its main block.
spec = importlib.util.spec_from_file_location("bsc", "step4_ideology/build_scoring_chunks.py")
bsc = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(bsc)
except SystemExit:  # the module calls sys.exit() when imported without arguments; ignore that
    pass
# Chunk every holdout transcript (and The Majority Report, show 402306412, as a corpus reference point) into 750-char passages.
rows = []
for p in sorted(glob.glob("data/transcripts/HOLDOUT_*/*.json")) + sorted(
    glob.glob("data/transcripts/402306412/*.json")
):
    s = p.split("/")[2]  # show folder name = show id
    rows += [(s, t) for t in bsc.split_750(bsc.transcript_text(p)) if len(t) >= 120]  # skip very short passages
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
TOK = re.compile(r"[a-z']+")
# Parse and count imperatives by bucket, exactly as in imp_decomp.py (no second-person breakdown here).
per = collections.defaultdict(collections.Counter)
for (s, t), doc in zip(rows, nlp.pipe([t for _, t in rows], batch_size=800, n_process=8)):
    per[s]["nw"] += len(TOK.findall(t.lower()))  # words per show, for the rates
    for sent in doc.sents:
        toks = list(sent)
        for k, tok in enumerate(toks):
            if (
                tok.pos_ == "VERB"
                and tok.tag_ == "VB"
                and tok.dep_ in ("ROOT", "conj", "ccomp", "advcl")
            ):
                if any(c.dep_ in ("nsubj", "nsubjpass", "expl", "aux", "auxpass") for c in tok.children):  # has a subject or auxiliary -> not an imperative
                    continue
                if k > 0 and toks[k - 1].text.lower() in ("to", "not", "never"):  # infinitives and bare negations excluded
                    continue
                v = tok.text.lower()
                b = (
                    "epistemic"
                    if v in EPIST
                    else "attention" if v in ATTN else "action" if v in ACTION else "other"
                )
                per[s]["imp_" + b] += 1
                per[s]["imp_all"] += 1
# Counts -> rates per 10,000 words; one row per show.
D = (
    pd.DataFrame.from_dict(per, orient="index")
    .fillna(0)
    .reset_index()
    .rename(columns={"index": "show_id"})
)
for c in ("imp_epistemic", "imp_attention", "imp_action", "imp_other", "imp_all"):
    D[c + "_rate"] = D[c] / D.nw * 1e4
D.to_csv(INPUTS / "holdout_imp_decomp.csv", index=False)
print("done", len(D))
