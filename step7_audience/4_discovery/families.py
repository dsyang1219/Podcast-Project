"""Benjamini-Hochberg survival of the discovery-sample results under alternative outcome-family definitions. Disclosure section G.1."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, warnings, re
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
rng = np.random.default_rng(21)
from common import *

g = st  # the strict (exact-title) match, the sample of record


# Local re-definition of common.num (same behaviour: negative missing codes -> NaN).
def num(g, c):
    s = pd.to_numeric(g[c], errors="coerce")
    return s.where(s > 0)


X0 = design(g, ())  # controls only (party, attention, age, education); dir_z is added per item


# Wild cluster bootstrap p for dir_z on one survey item y: observed show-clustered t, then t-stats from B refits
# with each show's null-model residuals sign-flipped. Also returns the observed t and the SD of the bootstrap t's
# (how much wider than N(0,1) the null distribution is with ~50 clusters).
def wildp(y, B=1000):
    Xf = X0.copy()
    Xf["dir_z"] = g.dir_z
    m = pd.concat([Xf, y.rename("y"), g[["show_id"]]], axis=1).dropna().reset_index(drop=True)
    c0 = [c for c in Xf.columns if c != "dir_z"]
    ids = m.show_id.values
    u = np.unique(ids)
    t0 = sm.OLS(m.y, m[Xf.columns]).fit(cov_type="cluster", cov_kwds={"groups": ids}).tvalues["dir_z"]
    r0 = sm.OLS(m.y, m[c0]).fit()
    res = r0.resid.values
    f0 = r0.fittedvalues.values
    sg = pd.Series(index=u, dtype=float)
    tb = np.empty(B)
    for i in range(B):
        sg[:] = rng.choice([-1.0, 1.0], len(u))  # one coin per show
        yb = f0 + res * sg.reindex(ids).values
        tb[i] = (
            sm.OLS(yb, m[Xf.columns]).fit(cov_type="cluster", cov_kwds={"groups": ids}).tvalues["dir_z"]
        )
    return (np.abs(tb) >= abs(t0)).mean(), t0, np.std(tb)


# Family C: every democratic-attitude item, defined by question block, NOT by result
DEM = (
    ["Q22", "Q23", "Q29", "Q31"]
    + [f"Q30{c}" for c in "ABC"]
    + [f"Q32{c}" for c in "ABCDEFGHIJ"]
    + [f"Q33{c}" for c in "ABCDEFGHIJ"]
    + [f"Q34{c}" for c in "ABCD"]
    + [f"Q35{c}" for c in "ABCDE"]
    + [f"Q36{c}" for c in "ABCDEFGHIJK"]
)
# Run the wild bootstrap on every family item that exists in the frame, has >= 300 answers and actually varies.
rows = []
for v in DEM:
    if v not in g:
        continue
    y = num(g, v)
    if y.notna().sum() < 300 or y.nunique() < 2:
        continue
    p, t, sd = wildp(y)
    rows.append((v, t, p, sd))
F = pd.DataFrame(rows, columns=["var", "t", "wild_p", "boot_t_sd"]).sort_values("wild_p")  # best item first
F.to_csv(OUTPUTS / "wild_dem_family.csv", index=False)
print(f"REAL wild bootstrap on {len(F)} democratic-attitude items (1000 reps each)")
print(
    f"  bootstrap t-sd across items: median={F.boot_t_sd.median():.2f}  range=[{F.boot_t_sd.min():.2f},{F.boot_t_sd.max():.2f}]  (my earlier deflation used 1.53)"
)
print(F.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# BH within one family: q-values, the items surviving q < .05, and the rank-1 threshold .05/k
# (the p the single best item must beat).
def bh(sub, label):
    q = multipletests(sub.wild_p, method="fdr_bh")[1]
    surv = sub[q < 0.05]["var"].tolist()
    print(
        f"  {label:<52} k={len(sub):3d}  first-rank threshold={.05/len(sub):.5f}  survivors: {surv if surv else 'NONE'}   min q={q.min():.3f}"
    )


# Families C-F are nested subsets of the democratic-attitude items, defined by question block, not by result.
print("\nDOES ANYTHING SURVIVE BH? depends entirely on the family:")
bh(F, "C. all democratic-attitude items (Q22,23,29-36)")
bh(F[F["var"].str.match(r"Q3[345]")], "D. institutional confidence + efficacy (Q33-35)")
bh(F[F["var"].str.match(r"Q33")], "E. institutional confidence only (Q33)")
bh(F[F["var"].str.match(r"Q3[2-6]|Q29|Q31")], "F. Q29,Q31,Q32-36 (democracy batteries)")
# full 113 family: wild p where computed, deflated-analytic elsewhere (conservative for the top items, see below)
S = pd.read_csv(OUTPUTS / "outcome_scan.csv")
import scipy.stats as st_  # aliased so it does not shadow the strict frame `st`

S["p_use"] = 2 * st_.norm.sf(np.abs(st_.norm.isf(S.p / 2)) / F.boot_t_sd.median())  # deflated analytic p: shrink the z-score by the median bootstrap t-SD, recompute two-sided p
S = S.set_index("var")
S.loc[F["var"], "p_use"] = F.set_index("var").wild_p  # where a real wild p was computed, use it instead
S = S.reset_index().rename(columns={"p_use": "wild_p"})
bh(S, "A. ALL 113 screened outcomes")
bh(S[~S["var"].str.match(r"Q2[01]")], "B. 98 non-media-use outcomes")
# Arithmetic of the failure: with k = 113 outcomes the rank-1 BH threshold is .05/113 = .00044.
print("\nWHY the best item fails in the big family:")
top = F.iloc[0]
print(
    f"  best item {top['var']}: wild p={top.wild_p:.4f}.  BH rank-1 threshold = .05/k.  k=113 -> .00044 (fails, {top.wild_p/.00044:.1f}x too large).  k={len(F)} -> {.05/len(F):.5f}."
)
