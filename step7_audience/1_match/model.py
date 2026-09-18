"""Build inputs/kett.pkl from the Kettering microdata (derived party, attention, age, education, weights) and fit the first respondent-level model of the favourability gap. Entry point for everything downstream."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm

# Raw Kettering Y1 microdata; the row order gives the respondent id used by every match table.
B = str(ROOT) + "/"
K = pd.read_csv(
    B + "data/external/kettering/KETTERING_DATA_Y1_WEIGHTED.csv", low_memory=False, encoding="cp1252"
)
K = K.reset_index().rename(columns={"index": "resp"}).copy()  # resp = original row number


def clean(s):  # Kettering uses -98/-99 style missing
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s > 0)  # negative codes -> NaN


# Party favourability: Q39A/Q39B recentred so 0 = neutral; ap_gap = absolute affective-polarisation gap.
K["repfav"] = clean(K.Q39A) - 6
K["demfav"] = clean(K.Q39B) - 6
K["ap_gap"] = (K.repfav - K.demfav).abs()
# Controls: political attention, education, age.
K["attn"] = clean(K.Q19)
K["edu"] = clean(K.EDU)
K["age"] = clean(K.AGE)
# Party id: Q41 gives Republican/Democratic identifiers; Q43 sorts the rest into leaners; everyone else is I.
q41 = clean(K.Q41)
q43 = clean(K.Q43)
K["pid"] = np.where(
    q41 == 1, "R", np.where(q41 == 2, "D", np.where(q43 == 2, "leanR", np.where(q43 == 1, "leanD", "I")))
)
# panel = probability-sample respondents (SAMP_TYPE 1/2); w = survey weight.
K["panel"] = clean(K.SAMP_TYPE).isin([1, 2])
K["w"] = clean(K.WEIGHT)
K.to_pickle(INPUTS / "kett.pkl")  # kett.pkl is the input of record for every downstream script

# First model: strict (exact-title) matches joined to show-level dir_z and lean.
M = pd.read_csv(INPUTS / "match_strict.csv")
S = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "dir_z", "lean"]]
M = M.merge(S, on="show_id", how="left")
# respondent-level exposure = mean dir_z across named corpus shows
R = (
    M.groupby("resp")
    .agg(
        dir_z=("dir_z", "mean"),
        share_right=("lean", lambda x: (x == "R").mean()),
        n_shows=("show_id", "nunique"),
    )
    .reset_index()
)
# Respondent frame = exposure + survey variables.
d = R.merge(K, on="resp", how="left")
print("matched:", len(d), "| ap_gap present:", d.ap_gap.notna().sum())


# OLS of ap_gap on dir_z + party dummies, attention, age, education (plain SEs; clustering by show comes later in common.py).
def fit(df, y="ap_gap", extra=()):
    X = pd.get_dummies(df["pid"], prefix="pid", drop_first=True).astype(float)
    X["dir_z"] = df.dir_z
    X["attn"] = df.attn
    X["age"] = df.age
    X["edu"] = df.edu
    for e in extra:
        X[e] = df[e]
    X = sm.add_constant(X, has_constant="add")
    m = pd.concat([X, df[y]], axis=1).dropna()
    if len(m) < 30:  # too few complete cases to fit
        return None
    r = sm.OLS(m[y], m[X.columns]).fit()
    return r, len(m)


# Baseline, then add the share of named shows leaning right to separate register from show ideology.
r, n = fit(d)
print(f"\nBASELINE strict n={n}")
print(f"  dir_z  b={r.params['dir_z']:+.3f}  t={r.tvalues['dir_z']:+.2f}  p={r.pvalues['dir_z']:.4f}")
r2, n2 = fit(d, extra=("share_right",))
print(
    f"  + share_right (n={n2}): dir_z b={r2.params['dir_z']:+.3f} p={r2.pvalues['dir_z']:.4f} | share_right p={r2.pvalues['share_right']:.4f}"
)
print("  corr(dir_z, share_right) =", round(d.dir_z.corr(d.share_right), 3))
