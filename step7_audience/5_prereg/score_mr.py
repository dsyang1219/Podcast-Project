"""Score The Majority Report from its existing corpus transcripts with the same pinned pipeline, for the within-frame test. Writes inputs/mr_show_dirz.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS

"""CONFIRMATORY scoring of holdout transcripts. Pinned procedure (prereg 'Scoring procedure'):
(1) build_scoring_chunks split_750; (2) remeasure2 features verbatim; (3) nw-weighted show rates;
(4) z vs 194-show reference (population SD); (5) dir_z = mean of 3 z. Adds, for Amendments 1-2 only:
per-passage position (chunk_ix/n_chunks) and declarative-you count. Touches no survey outcomes."""
import sys, glob, os, re, numpy as np, pandas as pd, importlib.util, spacy, time

# Run from the repo root so the corpus module's relative paths resolve.
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
OUT = str(INPUTS)
# Load step4_ideology/build_scoring_chunks.py as a module to reuse its split_750 and transcript_text; it may call sys.exit on import, which we swallow.
spec = importlib.util.spec_from_file_location("bsc", "step4_ideology/build_scoring_chunks.py")
bsc = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(bsc)
except SystemExit:
    pass
t0 = time.time()
# Chunk The Majority Report's already-downloaded corpus transcripts (show_id 402306412), per the Majority Report clause of prereg_withinframe.md; the pipeline below is otherwise identical to score_holdout_final.py.
rows = []
for p in sorted(glob.glob("data/transcripts/402306412/*.json")):
    show = p.split("/")[2]  # folder name = show id (402306412)
    ep = os.path.basename(p).split("_")[0]  # episode id = filename prefix before the first underscore
    try:
        text = bsc.transcript_text(p)
    except Exception as e:
        print("skip", p, e)
        continue
    ch = [t for t in bsc.split_750(text) if len(t) >= 120]  # same 750-char split as the corpus; drop fragments under 120 characters
    n = len(ch)
    for i, t in enumerate(ch):
        rows.append((show, ep, i, n, t))
CH = pd.DataFrame(rows, columns=["show_id", "episode_id", "chunk_ix", "n_chunks", "text"])
# Progress line: episodes and passages per show.
print(
    f"chunked: {CH.episode_id.nunique()} episodes, {len(CH):,} passages",
    {k.replace("HOLDOUT_", ""): v for k, v in CH.groupby("show_id").episode_id.nunique().items()},  # the HOLDOUT_ replace is a no-op here (inherited from the holdout script)
    flush=True,
)
W = lambda a: re.compile(r"\b(?:" + "|".join(a) + r")\b", re.I)  # whole-word, case-insensitive alternation regex
# The frozen dictionary features (OLD = the original remeasure2 lists): second-person pronouns and the imperative cue phrases.
OLD = {
    "you": W(["you", "your", "yours", "yourself"]),
    "imper": W(
        [
            "look",
            "listen",
            "think about it",
            "consider",
            "remember",
            "imagine",
            "ask yourself",
            "understand",
            "realize",
            "let me tell you",
        ]
    ),
}
TOK = re.compile(r"[a-z']+")  # word tokenizer for nw (word count): runs of letters/apostrophes
# spaCy pipeline: dependency parse for imper_syn, NER only for the declarative-you variant.
nlp = spacy.load(
    "en_core_web_sm", disable=["lemmatizer"]
)  # NER kept ON only for the declarative-you variant (viii); tagger/parser identical


# Syntactic imperatives: a bare-form verb (tag VB) heading a clause (ROOT/conj/ccomp/advcl) with no subject or auxiliary among its children, and not preceded by to/not/never (infinitives and negations are not commands).
def syn_feats(doc):
    imp = 0
    for sent in doc.sents:
        for tok in sent:
            if (
                tok.pos_ == "VERB"
                and tok.tag_ == "VB"
                and tok.dep_ in ("ROOT", "conj", "ccomp", "advcl")  # clause-heading positions
            ):
                if any(c.dep_ in ("nsubj", "nsubjpass", "expl", "aux", "auxpass") for c in tok.children):  # has a subject or auxiliary -> not an imperative
                    continue
                if tok.i > sent.start and sent[tok.i - sent.start - 1].text.lower() in (  # previous word rules out 'to go', 'not go', 'never go'
                    "to",
                    "not",
                    "never",
                ):
                    continue
                imp += 1
    return imp


# Declarative 'you' (Amendment 2, variant viii): count second-person pronouns but skip questions, sentences naming a PERSON (likely guest-directed), and the filler 'you know'.
def declar_you(doc):
    k = 0
    for sent in doc.sents:
        if sent.text.strip().endswith("?") or any(e.label_ == "PERSON" for e in sent.ents):  # question, or names a person -> skip the whole sentence
            continue
        toks = list(sent)
        for i, t in enumerate(toks):
            if t.pos_ == "PRON" and "Person=2" in str(t.morph):  # spaCy morphology flags 2nd person: you/your/yours/yourself
                if i + 1 < len(toks) and toks[i + 1].text.lower() == "know":  # 'you know' filler
                    continue
                k += 1
    return k


# Per-passage features. nlp.pipe parses in parallel (8 processes); passages under 40 words are skipped as in the corpus; each count becomes a rate per 10,000 words.
out = []
for r, doc in zip(CH.itertuples(index=False), nlp.pipe(CH.text.tolist(), batch_size=400, n_process=8)):
    n = len(TOK.findall(r.text.lower()))  # word count nw
    if n < 40:  # corpus rule: skip passages under 40 words
        continue
    per = lambda c: 10000.0 * c / n  # count -> rate per 10k words
    out.append(
        (
            r.show_id,
            r.episode_id,
            r.chunk_ix,
            r.n_chunks,
            n,
            per(syn_feats(doc)),
            per(len(OLD["imper"].findall(r.text))),
            per(len(OLD["you"].findall(r.text))),
            per(declar_you(doc)),
        )
    )
F = pd.DataFrame(
    out,
    columns=[
        "show_id",
        "episode_id",
        "chunk_ix",
        "n_chunks",
        "nw",
        "imper_syn",
        "imper_rx",
        "you_rx",
        "you_declar",
    ],
)
F["pos"] = F.chunk_ix / F.n_chunks.clip(lower=1)  # relative position in the episode: 0 = opening, ~1 = end
F.to_csv(INPUTS / "mr_features.csv", index=False)  # passage-level features for The Majority Report
print(f"features: {len(F):,} passages in {time.time()-t0:.0f}s", flush=True)
# Show-level rates: mean of passage rates weighted by passage word count (nw), plus size bookkeeping.
S = (
    F.groupby("show_id")
    .apply(
        lambda g: pd.Series(
            {
                c: np.average(g[c], weights=g.nw)
                for c in ("imper_syn", "imper_rx", "you_rx", "you_declar")
            }
            | {"nw": g.nw.sum(), "episodes": g.episode_id.nunique(), "passages": len(g)}
        ),
        include_groups=False,
    )
    .reset_index()
)
# Put each rate on the corpus scale: z against the mean and population SD (ddof=0) of the 194 reference shows; dir_z = mean of the three z's.
R = pd.read_csv(HANDOFF / "prereg/dirz_reference_scale.csv")
for c in ("imper_syn", "imper_rx", "you_rx"):
    mu, sd = R[c + "_rate"].mean(), R[c + "_rate"].std(ddof=0)  # population SD, as the reference scale was built
    S[c + "_z"] = (S[c] - mu) / sd
S["dir_z"] = S[["imper_syn_z", "imper_rx_z", "you_rx_z"]].mean(axis=1)


# (vii) position-based format score: log(you_rx opening 10% / you_rx body 30-100%), nw-weighted within show
def fmt(g):
    o = g[g.pos < 0.10]  # opening 10% of the episode
    b = g[g.pos >= 0.30]  # body: 30% onward, skipping the transition zone
    return np.log(np.average(o.you_rx, weights=o.nw) / np.average(b.you_rx, weights=b.nw))  # > 0 means address concentrates in the open (monologue fingerprint)


S = S.merge(
    F.groupby("show_id").apply(fmt, include_groups=False).rename("fmt_pos").reset_index(), on="show_id"
)
# (viii) declarative-you z: reference = 50-show corpus sample distribution of the same quantity (only corpus estimate available)
C = pd.read_csv(INPUTS / "corpus_you_split.csv")
mu, sd = C.you_declar_rate.mean(), C.you_declar_rate.std(ddof=0)
S["you_declar_z"] = (S.you_declar - mu) / sd
S["dir_z_declar"] = S[["imper_syn_z", "imper_rx_z", "you_declar_z"]].mean(axis=1)  # alternative exposure: declarative-you replaces all-you
# Save the show table, most directive first.
S = S.sort_values("dir_z", ascending=False)
S.to_csv(INPUTS / "mr_show_dirz.csv", index=False)  # the one-row show score that feeds WITHINFRAME_SCORES.csv for withinframe_test.py
# Print the final scores: small values with sign and two decimals, counts as integers.
print("\n=== FINAL holdout show scores (corpus scale) ===")  # header text inherited from the holdout script; this is The Majority Report
print(
    S[
        [
            "show_id",
            "episodes",
            "passages",
            "imper_syn",
            "imper_rx",
            "you_rx",
            "you_declar",
            "dir_z",
            "dir_z_declar",
            "fmt_pos",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:+.2f}" if abs(x) < 10 else f"{x:.0f}")
)
