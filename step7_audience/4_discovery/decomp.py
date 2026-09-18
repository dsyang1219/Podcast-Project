"""Decompose the register composite: which register measures carry the association with listener cynicism, with ideology controls and interaction tests. Full record F.5."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, warnings
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
rng = np.random.default_rng(31)
from common import *

g = st  # strict (exact-title) match, the frame of record
IDEO = ("share_right", "rw_plat")  # ideology controls: share of named shows leaning right, right-wing platform count
# The 14 dictionary/register measures scored per show. The first two (imperatives, second_person)
# are the ingredients of dir_z; the rest are rival dimensions of register.
MEAS = [
    "imperatives",
    "second_person",
    "out_group_pron",
    "in_group_pron",
    "religious",
    "profanity",
    "hedging",
    "assertion",
    "america",
    "named_actors",
    "insult",
    "catastrophe",
    "fillers",
    "valence",
]
SM = pd.read_csv(INPUTS / "show_measures.csv")[["show_id"] + MEAS]
# z-score each measure across shows so coefficients are comparable (effect per 1 SD of the show measure).
for c in MEAS:
    SM[c] = (SM[c] - SM[c].mean()) / SM[c].std()
# Respondent-level exposure to each measure = mean over the corpus shows the respondent named
# (mirrors how dir_z is built in common.frame), then attach to the respondent frame.
M = pd.read_csv(INPUTS / "match_strict.csv").merge(SM, on="show_id", how="left")
R = M.groupby("resp")[MEAS].mean().reset_index()
g = g.merge(R, on="resp", how="left")


# Same wild cluster bootstrap as common.wild(), but for an arbitrary exposure column `xcol`
# instead of dir_z. Returns (coefficient, wild-bootstrap p).
def wild_for(xcol, y="inst_cyn", extra=IDEO, B=1000):
    X = design(g, extra)
    Xf = X.copy()
    Xf["x"] = g[xcol]
    m = pd.concat([Xf, g[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    c0 = [c for c in Xf.columns if c != "x"]  # the null model: controls only, exposure removed
    ids = m.show_id.values
    u = np.unique(ids)
    full = sm.OLS(m[y], m[Xf.columns]).fit(cov_type="cluster", cov_kwds={"groups": ids})
    b, t0 = full.params["x"], full.tvalues["x"]  # observed coefficient and cluster t-statistic
    r0 = sm.OLS(m[y], m[c0]).fit()  # restricted fit under H0: x has no effect
    res = r0.resid.values
    f0 = r0.fittedvalues.values
    sg = pd.Series(index=u, dtype=float)
    tb = np.empty(B)
    for i in range(B):  # B = 1000 bootstrap draws
        sg[:] = rng.choice([-1.0, 1.0], len(u))  # one Rademacher coin per show, flipping that show's residuals
        yb = f0 + res * sg.reindex(ids).values  # fake outcome that obeys the null
        tb[i] = sm.OLS(yb, m[Xf.columns]).fit(cov_type="cluster", cov_kwds={"groups": ids}).tvalues["x"]
    return b, (np.abs(tb) >= abs(t0)).mean()  # p = share of null t-stats at least as extreme as the observed one


# --- A. Run each of the 14 measures, one at a time, as the exposure predicting institutional cynicism.
print("=== A. WHICH REGISTER DIMENSION carries institutional cynicism? (ideology controls, wild p) ===")
rows = [wild_for(c) + (c,) for c in MEAS]
D = pd.DataFrame(rows, columns=["b", "wild_p", "measure"]).sort_values("wild_p")
D["q_bh14"] = multipletests(D.wild_p, method="fdr_bh")[1]  # Benjamini-Hochberg q across the family of 14 tests
print(
    D[["measure", "b", "wild_p", "q_bh14"]].to_string(
        index=False, float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.3f}"
    )
)
# Reference line: the dir_z composite itself under the same model and bootstrap.
b, p = wild_for("dir_z")
print(f"\n  [reference] dir_z composite   b={b:+.3f}  wild p={p:.4f}")
print("  dir_z = imperatives + second_person (syntactic) — check those two are what's loading.")

# --- B. Does the dir_z -> cynicism association differ by party? Fit the model separately within
# Democrats, Independents and Republicans (each with ideology controls), CR1 clustered by show.
print("\n=== B. PARTY HETEROGENEITY on institutional cynicism (ideology controls) ===")
for lab, mask in (
    ("Democrats (D+leanD)", g.pid.isin(["D", "leanD"])),
    ("Independents", g.pid == "I"),
    ("Republicans (R+leanR)", g.pid.isin(["R", "leanR"])),
):
    gg = g[mask]
    if len(gg) < 40:  # too few respondents in the subgroup for a clustered regression to mean anything
        print(f"  {lab:<24} n={len(gg)} too few")
        continue
    # Party dummies are constant (or nearly so) within a party subgroup, so drop them from the design.
    X = design(gg, IDEO).drop(columns=[c for c in design(gg, IDEO).columns if c.startswith("pid_")])
    X["dir_z"] = gg.dir_z
    m = pd.concat([X, gg[["inst_cyn", "show_id"]]], axis=1).dropna()
    r = sm.OLS(m.inst_cyn, m[X.columns]).fit(cov_type="cluster", cov_kwds={"groups": m.show_id})
    print(
        f"  {lab:<24} n={len(m):3d}  b={r.params['dir_z']:+.3f}  cluster p={r.pvalues['dir_z']:.4f}  ({m.show_id.nunique()} shows)"
    )
# interaction test
# Pooled model with dir_z x party-dummy interaction terms; a joint F-test asks whether the
# slope of dir_z differs across parties at all.
X = design(g, IDEO)
X["dir_z"] = g.dir_z
for c in [c for c in X.columns if c.startswith("pid_")]:
    X[f"dir_x_{c}"] = X[c] * g.dir_z  # one interaction per (non-reference) party dummy
m = pd.concat([X, g[["inst_cyn", "show_id"]]], axis=1).dropna()
r = sm.OLS(m.inst_cyn, m[X.columns]).fit(cov_type="cluster", cov_kwds={"groups": m.show_id})
ints = [c for c in X.columns if c.startswith("dir_x_")]
print(
    f"  joint test of dir_z x party interactions: F p = {r.f_test(' = '.join(ints)+' = 0' if False else [f'{c} = 0' for c in ints]).pvalue:.4f}"  # the `if False` branch is dead code; the list of 'term = 0' restrictions is what is tested
)
