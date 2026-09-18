"""Show-level permutation test for the population-profile items in d_pop.pkl."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, scipy.stats as st, warnings

warnings.filterwarnings("ignore")
rng = np.random.default_rng(11)  # seed for the permutation draws
# Respondent frame for the population-profile items (built upstream).
d = pd.read_pickle(INPUTS / "d_pop.pkl")
# Controls only (no dir_z): party dummies, attention, age, education.
X = pd.get_dummies(d["pid"], prefix="pid", drop_first=True).astype(float)
X["attn"] = d.attn
X["age"] = d.age
X["edu"] = d.edu
X = sm.add_constant(X, has_constant="add")
# Residualise ap_gap on the controls, then average the residuals within each show: one number per show.
m = pd.concat([X, d[["ap_gap", "dir_z", "show_id"]]], axis=1).dropna().reset_index(drop=True)
m["r"] = sm.OLS(m.ap_gap, m[X.columns]).fit().resid
g = m.groupby("show_id").agg(dir_z=("dir_z", "first"), r=("r", "mean"), n=("r", "size")).reset_index()  # n = matched listeners per show (precision weight)


# Show-level WLS of mean residual on dir_z, weighted by listener count.
def wls(gg):
    w = gg.n.values
    Z = sm.add_constant(gg.dir_z.values)
    return sm.WLS(gg.r.values, Z, weights=w).fit()


# Repeat for increasingly strict minimum-listener thresholds; singleton shows contribute only one person's noise.
print("SHOW-LEVEL WLS (precision-weighted by listener count) — the correct show-level test")
for k in (1, 2, 3, 5):
    gg = g[g.n >= k]  # keep shows with at least k matched listeners
    r = wls(gg)
    b = r.params[1]
    # permutation: shuffle dir_z across shows
    null = []
    for _ in range(5000):
        h = gg.copy()
        h["dir_z"] = rng.permutation(h.dir_z.values)  # refit on shuffled exposure
        null.append(wls(h).params[1])
    null = np.array(null)
    p = (np.abs(null) >= abs(b)).mean()  # two-sided permutation p
    print(
        f"  shows n>={k:<2} ({len(gg):2d} shows, {gg.n.sum():3d} resp)  b={b:+.3f}  permutation p={p:.4f}"
    )

# How many shows are one-listener shows, and how few respondents they actually carry.
print("\nWhat the singleton shows contribute:")
print(
    f"  {(g.n==1).sum()} shows have exactly 1 listener; their show-mean residual is one person's noise."
)
print(
    f"  they are {(g.n==1).sum()/len(g):.0%} of shows but {g[g.n==1].n.sum()/g.n.sum():.0%} of respondents."
)
