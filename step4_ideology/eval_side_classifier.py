#!/usr/bin/env python3
"""Can the content ideology measure assign each show a side, better than DIME?

    ./eval_side_classifier.py

The gate this answers
---------------------
Defining "out-group" requires knowing each show's side. Today that comes from
Brookings partisan lean: 95 shows, only 41 of which also have chart coverage,
which is why the out-group test runs at n=300. Substituting DIME doubles the
sample and makes the test WORSE -- measured, not assumed:

    Brookings (41 shows, n=300)   std beta 0.258   t 3.30
    DIME target (84, n=561)       std beta 0.164   t 2.76
    guest DIME  (79, n=539)       std beta 0.168   t 2.74

A misassigned show does not add noise, it subtracts signal: its out-group terms
become in-group terms and it contributes with the sign reversed. So a broader
anchor only helps if it is also a BETTER one.

Three rules, all evaluated leave-one-out
----------------------------------------
    A  threshold on the net index      the naive rule
    B  logistic on the 7-bin shape     a show that is half strongly_conservative
                                       and half strongly_liberal has a net index
                                       near zero and looks identical to a show
                                       that is uniformly moderate. It is not.
    C  DIME target / guest DIME        the benchmark

The DIME accuracies quoted above (66.7%, 68.3%) were best-achievable thresholds
picked on the same data they were scored on -- an upper bound. Everything here is
leave-one-out so the comparison is fair in both directions, which means the
honest DIME number is LOWER than the bar previously quoted.

The circularity condition
-------------------------
If the anchor is derived from a show's text and the out-group measure counts
party references in the SAME text, the two are not independent: a left show that
constantly attacks Republicans could have its side inferred from those very
mentions, and the correlation is partly built in. Brookings is external and does
not have this problem, which is why the 41-show test is CLEANER even though it is
smaller.

So the anchor is also scored on PARTY-FREE chunks only -- chunks containing no
party, politician, or ideological-label term. If accuracy holds there, the
circularity is broken and the larger sample is legitimate. If it collapses, the
anchor was riding on party mentions and the bigger test is not valid.
"""
import re, sys
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # repository root (this script lives one folder down)

PARTY = re.compile(r"\b(?:democrat|democrats|democratic party|liberal|liberals|progressives|"
                   r"biden|kamala|pelosi|schumer|aoc|republican|republicans|gop|conservative|"
                   r"conservatives|trump|maga|desantis|mcconnell)\b", re.I)
T = {"strongly_conservative":1,"moderately_conservative":1,"slightly_conservative":1,
     "moderate":0,"slightly_liberal":-1,"moderately_liberal":-1,"strongly_liberal":-1}
BINS = list(T)


def show_features(chunks, scored, party_free=False):
    d = chunks.merge(scored, on="chunk_id")
    d = d[d.politics == "Yes"].copy()
    if party_free:
        d = d[~d.text.fillna("").str.contains(PARTY)]
    d["t"] = d.ideology.map(T)
    d = d.dropna(subset=["t"])
    g = d.groupby("show_id")
    net = g.apply(lambda x: ((x.t > 0) * x.n_words).sum()/x.n_words.sum()
                          - ((x.t < 0) * x.n_words).sum()/x.n_words.sum(),
                  include_groups=False).rename("net")
    shape = (d.groupby(["show_id","ideology"]).size().unstack(fill_value=0)
               .reindex(columns=BINS, fill_value=0))
    shape = shape.div(shape.sum(1).replace(0, np.nan), axis=0)
    n = g.size().rename("n_pol")
    return pd.concat([net, shape, n], axis=1).dropna(subset=["net"])


def loo_threshold(x, y):
    """Leave-one-out: threshold chosen without the held-out show."""
    ok = 0
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        cands = np.unique(x[m])
        best_th, best = 0.0, -1
        for th in cands:
            a = (((x[m] >= th).astype(int)) == y[m]).mean()
            if a > best: best, best_th = a, th
        ok += int((x[i] >= best_th) == y[i])
    return ok/len(x)


def loo_logistic(X, y):
    ok = 0
    for i in range(len(X)):
        m = np.ones(len(X), bool); m[i] = False
        Xt, yt = X[m], y[m]
        Xt = np.column_stack([np.ones(len(Xt)), Xt])
        b = np.zeros(Xt.shape[1])
        for _ in range(60):                       # Newton with ridge
            p = 1/(1+np.exp(-Xt@b)); W = p*(1-p)+1e-6
            H = Xt.T@(Xt*W[:,None]) + 1e-2*np.eye(Xt.shape[1])
            try: b = b + np.linalg.solve(H, Xt.T@(yt-p))
            except np.linalg.LinAlgError: break
        xi = np.concatenate([[1.0], X[i]])
        ok += int((1/(1+np.exp(-xi@b)) >= .5) == y[i])
    return ok/len(X)


def auc(x, y):
    pos, neg = x[y==1], x[y==0]
    if not len(pos) or not len(neg): return np.nan
    return float(np.mean([(a>b)+0.5*(a==b) for a in pos for b in neg]))


def main():
    sc = ROOT/"data/output/ideology_sideclass.csv"
    if not sc.exists():
        sys.exit("scoring not finished — data/output/ideology_sideclass.csv missing")
    chunks = pd.read_csv(ROOT/"data/output/scoring_chunks_sideclass.csv")
    chunks["show_id"] = chunks.show_id.astype(str)
    scored = pd.read_csv(sc)
    lean = pd.read_csv(ROOT/"data/output/lean_validation.csv")
    lean = lean[lean.match_status == "matched"]   # exact-title matches only; fuzzy rows are wrong
    lean = lean[lean.brookings_partisan_leaning.isin(["More Conservative","More Liberal"])]
    truth = {str(r.collection_id): int(r.brookings_partisan_leaning=="More Conservative")
             for r in lean.itertuples()}

    print(f"{'rule':<34}{'shows':>7}{'LOO acc':>10}{'AUC':>8}")
    for label, pf in (("all chunks", False), ("PARTY-FREE chunks only", True)):
        f = show_features(chunks, scored, party_free=pf)
        f = f[f.index.isin(truth)]
        y = np.array([truth[s] for s in f.index])
        if len(y) < 30 or y.sum() in (0, len(y)):
            print(f"  content anchor · {label:<18} insufficient"); continue
        print(f"  ── content anchor · {label}   (median {f.n_pol.median():.0f} political chunks/show)")
        a = loo_threshold(f.net.values, y)
        print(f"    A  threshold on net index{'':<9}{len(y):>7}{a:>10.1%}{auc(f.net.values,y):>8.3f}")
        b = loo_logistic(f[BINS].values, y)
        print(f"    B  logistic on 7-bin shape{'':<8}{len(y):>7}{b:>10.1%}{'':>8}")

    # benchmark, scored the same way
    print("  ── benchmark, same LOO treatment")
    t = pd.read_csv(ROOT/"data/output/ideology_targets_204.csv")
    dm = {str(r.show_id): r.ideology_primary for r in t.itertuples() if pd.notna(r.ideology_primary)}
    common = [s for s in truth if s in dm]
    x = np.array([dm[s] for s in common]); y = np.array([truth[s] for s in common])
    print(f"    C  DIME target{'':<20}{len(y):>7}{loo_threshold(x,y):>10.1%}{auc(x,y):>8.3f}")
    print("\n  GATE: the content anchor must beat DIME on PARTY-FREE chunks by enough")
    print("  that applying it to 174 shows beats staying on Brookings' validated 41.")


if __name__ == "__main__":
    main()
