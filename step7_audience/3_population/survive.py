"""Which population-level composites survive the full control ladder; builds d_pop.pkl for perm.py and scan.py."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, re, warnings

warnings.filterwarnings("ignore")
# Full Kettering respondent file, the long table of Q17 verbatims (one row per named source), and the strict-match respondent ids.
K = pd.read_pickle(INPUTS / "kett.pkl")
L = pd.read_csv(INPUTS / "verbatims_long.csv")
strict = set(pd.read_csv(INPUTS / "match_strict.csv").resp)


# Survey item as numbers; Kettering codes missing as negatives.
def num(df, c):
    s = pd.to_numeric(df[c], errors="coerce")
    return s.where(s > 0)


# 'Names a podcast' regex: big podcast hosts/shows plus the generic words podcast/pod. With the strict corpus matches this defines the podcast-namer group.
BIG = r"joe rogan|rogan|daily wire|shapiro|tucker|megyn kelly|charlie kirk|tim pool|hasan|piker|glenn beck|bongino|theo von|lex fridman|pod save|meidas|midas|bulwark|breaking points|young turks|tyt|pakman|crowder|candace|benny johnson|heather cox|matt walsh|knowles|klavan|jordan peterson|sam harris|ezra klein|the daily|up first|npr politics|\bpodcast|\bpod\b"
L["pod"] = L.v.str.contains(BIG, regex=True, na=False) | L.resp.isin(strict)  # any verbatim matches, or the respondent is a strict corpus match
R = L.groupby("resp").agg(pod=("pod", "max"), nsrc=("v", "nunique")).reset_index()  # per respondent: podcast-namer flag and number of distinct sources named (diet breadth)
D = K.merge(R, on="resp")
D = D[D.w.notna()].copy()  # keep only respondents with a survey weight
# ---- composites
# Composites: the 4-item discovery version (core) and the 8-item pre-registered version (broad).
z = lambda s: (s - s.mean()) / s.std()
core = ["Q35E", "Q35C", "Q34B", "Q33H"]
broad = core + ["Q31", "Q33E", "Q35A", "Q35D"]


# Cronbach's alpha on complete cases: k/(k-1) * (1 - sum of item variances / variance of the item sum).
def alpha(cols):
    Z = pd.DataFrame({c: num(D, c) for c in cols}).dropna()
    k = len(cols)
    return (k / (k - 1)) * (1 - Z.var().sum() / Z.sum(axis=1).var()), len(Z)


# Average of z-scored items, negated because the items ask whether things work well; higher = more cynical.
D["cyn4"] = -sum(z(num(D, c)) for c in core) / 4
D["cyn8"] = -sum(z(num(D, c)) for c in broad) / 8
a4, n4 = alpha(core)
a8, n8 = alpha(broad)
print(
    f"SCALE RELIABILITY at full n:  4-item cyn alpha={a4:.3f} (n={n4:,})   8-item alpha={a8:.3f} (n={n8:,})"
)
# Re-standardise so slopes read in SD units.
D["cyn4"] = z(D.cyn4)
D["cyn8"] = z(D.cyn8)
# ---- extra controls
# Extra controls for the ladder: gender, income, race, social-media hours, platforms used, survey mode, urban-rural code.
D["female"] = (num(D, "GENDER") == 2).astype(float)  # GENDER 2 = female
D["income"] = num(D, "Q49")
D["white"] = (pd.to_numeric(D.RACE_1, errors="coerce") == 1).astype(float)  # RACE_1 1 = white
D["sm_hours"] = num(D, "Q21")
D["nplat"] = sum((num(D, f"Q20{c}") <= 2).astype(float) for c in "ABCDEFGHIJKLMN")  # how many of the 14 Q20 platforms are used regularly (code <= 2)
D["mail"] = (num(D, "MODE") == 2).astype(float)  # MODE 2 = mail questionnaire
D["urc"] = pd.to_numeric(D.Urban_Rural_Continuum_Code, errors="coerce")
D["pod"] = D.pod.astype(float)


# Survey-weighted least squares of outcome y on the podcast-namer flag plus controls; prints and returns the 'pod' slope.
def fit(g, y, extra, label):
    X = pd.get_dummies(g.pid, prefix="pid", drop_first=True).astype(float)
    X["attn"] = g.attn
    X["age"] = g.age
    X["edu"] = g.edu
    for e in extra:
        X[e] = g[e]
    X["pod"] = g.pod
    X = sm.add_constant(X, has_constant="add")
    m = pd.concat([X, g[[y]], g.w], axis=1).dropna()
    r = sm.WLS(m[y], m[X.columns], weights=m.w).fit(cov_type="HC1")  # weights = WEIGHT; HC1 robust SEs (population model, nothing to cluster on)
    print(
        f"  {label:<46} b={r.params['pod']:+.3f} SD  [{r.conf_int().loc['pod',0]:+.3f},{r.conf_int().loc['pod',1]:+.3f}]  p={r.pvalues['pod']:.2e}  n={len(m):,}"
    )
    return r.params["pod"]


print(
    "\n=== 1. CONTROL LADDER (podcast-namer vs everyone else; outcome = 4-item cynicism, SD units) ==="
)
# Section 1: add control blocks one at a time and watch how much the podcast-namer gap shrinks.
b0 = fit(D, "cyn4", [], "base: party+attention+age+edu")
fit(D, "cyn4", ["female", "white", "income"], "+ gender, race, income")
fit(D, "cyn4", ["female", "white", "income", "sm_hours", "nplat"], "+ social-media hours, # platforms")
fit(
    D,
    "cyn4",
    ["female", "white", "income", "sm_hours", "nplat", "urc", "mail"],
    "+ urban-rural, survey mode",
)
bf = fit(
    D,
    "cyn4",
    ["female", "white", "income", "sm_hours", "nplat", "urc", "mail", "nsrc"],
    "+ diet breadth (# sources named)  [FULL]",
)
print(f"  attenuation base -> full: {100*(1-bf/b0):.0f}%")  # share of the base gap explained away by the full control set
# Section 2: the same full model on the 8-item composite.
print("\n=== 2. 8-item broad composite, FULL controls ===")
fit(
    D,
    "cyn8",
    ["female", "white", "income", "sm_hours", "nplat", "urc", "mail", "nsrc"],
    "8-item cynicism, full controls",
)
FULL = ["female", "white", "income", "sm_hours", "nplat", "urc", "mail", "nsrc"]  # the full control set, reused below
# Section 3: does the gap hold inside each party group?
print("\n=== 3. WITHIN PARTY (full controls) ===")
for lab, mask in (
    ("Democrats+leaners", D.pid.isin(["D", "leanD"])),
    ("Independents", D.pid == "I"),
    ("Republicans+leaners", D.pid.isin(["R", "leanR"])),
):
    fit(D[mask], "cyn4", FULL, lab)
# Section 4: probability-panel respondents vs opt-in (panel flag from SAMP_TYPE).
print("\n=== 4. SAMPLE TYPE (full controls) ===")
fit(D[D.panel], "cyn4", FULL, "Gallup Panel (probability) only")
fit(D[~D.panel], "cyn4", FULL, "opt-in only")
# Section 5: each of the eight items on its own, plus two non-cynicism items (political violence, loneliness) as contrasts.
print("\n=== 5. ITEM-BY-ITEM under FULL controls (SD units) ===")
for c, lab in (
    ("Q31", "leaders accountable"),
    ("Q35E", "laws uphold justice"),
    ("Q33E", "elections run well"),
    ("Q33H", "press freedom"),
    ("Q35C", "govt includes me"),
    ("Q34B", "reach officials"),
    ("Q35A", "govt reflects majority"),
    ("Q35D", "govt serves citizens"),
    ("Q36K", "political violence"),
    ("Q4", "loneliness"),
):
    D["_y"] = z(num(D, c))  # z-score the single item so b is in SD units
    fit(D, "_y", FULL, lab)
D.to_pickle(INPUTS / "D_pop.pkl")  # frame with composites and controls, reused by perm.py and scan.py
