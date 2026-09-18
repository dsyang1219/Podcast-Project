#!/usr/bin/env python3
"""Side-agnostic style measures, for the attention question.

WHY THESE AND NOT MORE OUTRAGE
------------------------------
The out-group test needed to know which side a show was on before it could
count anything, and side assignment was where it kept breaking. Every measure
here is computed without reference to a show's politics, so nothing depends on
Brookings, DIME, or an LLM classifier.

  othering        log((they+them+their) / (we+us+our)).  Pennebaker's pronoun
                  work: group boundaries show up in function words, which
                  speakers do not consciously manage. "They" marks an out-group
                  without the analyst having to know who "they" are.
  direct_address  you / your per 10k. Podcasts' distinguishing feature as a
                  medium is intimacy with an absent audience -- the parasocial
                  channel. This is the medium-matched construct, the way
                  Sobieraj & Berry's codebook was matched to talk radio.
  certainty       definitely / obviously / no question. Overclaiming.
  hedge           maybe / perhaps / I think. The complement.
  question_rate   '?' per 10k -- rhetorical engagement device.
  negation        not / never / nothing.
  first_person    I / me / my -- self-focus vs topic-focus.
  concreteness    mean Brysbaert concreteness over known tokens. Abstract talk
                  about principles vs concrete talk about events.
  moral_emo       Brady et al. (2017): moral-emotional words drive diffusion.

Counts are per 10k words so episode length cannot drive them.
"""
import re, os, csv, sys, glob, hashlib
from multiprocessing import Pool

W = lambda *a: re.compile(r"\b(?:" + "|".join(a) + r")\b", re.I)
THEY  = W("they", "them", "their", "theirs", "themselves")
WE    = W("we", "us", "our", "ours", "ourselves")
YOU   = W("you", "your", "yours", "yourself", "you're", "youre")
I1    = W("i", "me", "my", "mine", "myself")
CERT  = W("definitely","certainly","obviously","clearly","undeniably","absolutely",
          "unquestionably","without a doubt","no question","of course","literally")
HEDGE = W("maybe","perhaps","might","possibly","probably","seems","apparently",
          "sort of","kind of","i think","i guess","arguably","somewhat")
NEG   = W("not","never","no","nothing","nobody","none","cannot","can't","won't","don't")
MORAL = W("evil","wrong","right","wicked","corrupt","betray","betrayal","shame","shameful",
          "disgrace","disgraceful","outrage","outrageous","immoral","injustice","unjust",
          "hypocrite","hypocrisy","coward","cowardly","destroy","attack","fight","war",
          "hate","hatred","greed","greedy","liar","lie","lies","fraud","criminal","abuse")
TOK = re.compile(r"[a-z']+")

CONC = {}
def load_conc(path="data/output/brysbaert_concreteness.csv"):
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try: CONC[r["word"]] = float(r["conc_mean"])
            except (TypeError, ValueError): pass

def feats(path):
    try:
        with open(path, errors="replace") as fh: t = fh.read()
    except OSError: return None
    toks = TOK.findall(t.lower())
    n = len(toks)
    if n < 500: return None            # too short to estimate rates from
    per10k = lambda c: 10000.0 * c / n
    they, we = len(THEY.findall(t)), len(WE.findall(t))
    cs = [CONC[w] for w in toks if w in CONC]
    import math
    return dict(
        episode_id=os.path.basename(path).split("_")[0],
        show_id=os.path.basename(os.path.dirname(path)),
        n_words=n,
        othering=math.log((they + 1.0) / (we + 1.0)),
        they_per10k=per10k(they), we_per10k=per10k(we),
        direct_address=per10k(len(YOU.findall(t))),
        first_person=per10k(len(I1.findall(t))),
        certainty=per10k(len(CERT.findall(t))),
        hedge=per10k(len(HEDGE.findall(t))),
        question_rate=per10k(t.count("?")),
        negation=per10k(len(NEG.findall(t))),
        moral_emo=per10k(len(MORAL.findall(t))),
        concreteness=(sum(cs) / len(cs)) if cs else float("nan"),
        conc_coverage=len(cs) / n)

def main():
    load_conc()
    files = glob.glob("data/transcripts/*/*.txt")
    print(f"{len(files):,} transcripts", flush=True)
    out = "data/output/style_measures.csv"
    with Pool(12) as p, open(out, "w", newline="") as fh:
        w = None
        for i, r in enumerate(p.imap_unordered(feats, files, chunksize=64), 1):
            if r is None: continue
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(r)); w.writeheader()
            w.writerow(r)
            if i % 5000 == 0: print(f"  {i:,}/{len(files):,}", flush=True)
    print("done ->", out, flush=True)

if __name__ == "__main__":
    main()
