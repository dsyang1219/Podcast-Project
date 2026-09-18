#!/usr/bin/env python3
"""A second, wider battery of register measures.

The first battery found four discriminating features (vulgarity, hedging,
concreteness, direct address) and a wall of nulls on hostility. Four features is
thin evidence for a claim about "register", and the ones that worked cluster
around informality and audience orientation -- which may mean register is really
one dimension, or may mean the battery only looked in one place.

This adds sixteen measures spanning dimensions the first battery missed:

  spontaneity     fillers, contractions, discourse markers, false starts
                  -- scripted monologue vs unscripted talk
  complexity      type-token ratio, mean word length, long-word share,
                  words per sentence -- lexical and syntactic density
  stance          imperatives, superlatives, quantification, reported speech
                  -- how claims are delivered and sourced
  temporal        past / present / future orientation
  register-mark   religious vocabulary, institutional vocabulary

Each is a rate per 10k words (or a ratio), computed over full episode
transcripts, so nothing here depends on chunking or on the ideology scoring.
"""
import re, os, csv, glob, math
from multiprocessing import Pool

W = lambda *a: re.compile(r"\b(?:" + "|".join(a) + r")\b", re.I)
FILLER  = W("um","uh","erm","hmm","like","y'know","basically","literally","actually","right")
DISC    = W("so","well","now","anyway","look","listen","okay","alright","see")
IMPER   = W("look","listen","think about it","consider","remember","imagine","ask yourself",
            "understand","realize","picture this","let me tell you","here's the thing")
SUPER   = W("best","worst","greatest","biggest","most","least","never","always","every",
            "everyone","nobody","all","none","huge","massive","enormous")
QUANT   = re.compile(r"\b\d[\d,.]*\s*(?:percent|%|million|billion|trillion|thousand)?\b")
REPORT  = W("said","says","claimed","claims","told","argued","argues","wrote","stated",
            "according to","reportedly","allegedly")
PAST    = W("was","were","had","did","went","came","took","made","got","said","thought")
FUTURE  = W("will","going to","gonna","shall","would","could","might","may")
RELIG   = W("god","lord","jesus","christ","faith","pray","prayer","church","biblical",
            "scripture","sin","soul","holy","blessed","evil","moral")
INSTIT  = W("congress","senate","house","court","agency","department","administration",
            "committee","federal","legislation","bill","policy","regulation","statute")
CONTRA  = re.compile(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.I)
FALSE   = re.compile(r"\b(\w+)[,\s]+\1\b", re.I)          # word repeated = restart
TOK     = re.compile(r"[a-z']+")
SENT    = re.compile(r"[.!?]+")


def feats(path):
    try:
        with open(path, errors="replace") as fh: t = fh.read()
    except OSError: return None
    toks = TOK.findall(t.lower()); n = len(toks)
    if n < 500: return None
    per = lambda c: 10000.0 * c / n
    sents = [s for s in SENT.split(t) if s.strip()]
    uniq = len(set(toks))
    long_w = sum(1 for w in toks if len(w) >= 7)
    past, fut = len(PAST.findall(t)), len(FUTURE.findall(t))
    return dict(
        episode_id=os.path.basename(path).split("_")[0],
        show_id=os.path.basename(os.path.dirname(path)),
        n_words2=n,
        filler=per(len(FILLER.findall(t))),
        discourse=per(len(DISC.findall(t))),
        contraction=per(len(CONTRA.findall(t))),
        false_start=per(len(FALSE.findall(t))),
        imperative=per(len(IMPER.findall(t))),
        superlative=per(len(SUPER.findall(t))),
        quantification=per(len(QUANT.findall(t))),
        reported=per(len(REPORT.findall(t))),
        religious=per(len(RELIG.findall(t))),
        institutional=per(len(INSTIT.findall(t))),
        ttr=uniq / math.sqrt(n),                       # root TTR, length-robust
        word_len=sum(len(w) for w in toks) / n,
        long_word=100.0 * long_w / n,
        words_per_sent=n / max(len(sents), 1),
        past_future=math.log((past + 1) / (fut + 1)),
    )


def main():
    files = glob.glob("data/transcripts/*/*.txt")
    print(f"{len(files):,} transcripts", flush=True)
    out = "data/output/register2.csv"
    with Pool(12) as p, open(out, "w", newline="") as fh:
        w = None
        for i, r in enumerate(p.imap_unordered(feats, files, chunksize=64), 1):
            if r is None: continue
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(r)); w.writeheader()
            w.writerow(r)
            if i % 6000 == 0: print(f"  {i:,}", flush=True)
    print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
