"""The non-institutional-diet gradient on institutional cynicism: podcast-only, platform-only and mixed diets vs mainstream-only, plus per-source dose terms and the equivalence test. Paper finding 4."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, re, warnings

warnings.filterwarnings("ignore")
# Full survey (all weighted respondents), typed source verbatims, and the ids of strict corpus-show listeners.
K = pd.read_pickle(INPUTS / "kett.pkl")
L = pd.read_csv(INPUTS / "verbatims_long.csv")
strict = set(pd.read_csv(INPUTS / "match_strict.csv").resp)


# Negative Kettering missing codes -> NaN.
def num(df, c):
    s = pd.to_numeric(df[c], errors="coerce")
    return s.where(s > 0)


z = lambda s: (s - s.mean()) / s.std()  # z-score helper
# Three source classifiers applied to each verbatim: BIG = big-name podcasts/hosts or the word 'podcast'/'pod';
# MAIN = mainstream outlets; PLAT = a bare platform name (anchored ^...$: the whole verbatim must be the platform).
BIG = r"joe rogan|rogan|daily wire|shapiro|tucker|megyn kelly|charlie kirk|tim pool|hasan|piker|glenn beck|bongino|theo von|lex fridman|pod save|meidas|midas|bulwark|breaking points|young turks|tyt|pakman|crowder|candace|benny johnson|heather cox|matt walsh|knowles|klavan|jordan peterson|sam harris|ezra klein|the daily|up first|npr politics|\bpodcast|\bpod\b"
MAIN = r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|washington post|wall street|wsj|reuters|associated press|\bap\b|usa today|local news|newspaper|the hill|politico|axios|bloomberg|cnbc|news ?nation|newsmax"
PLAT = r"^(facebook|fb|youtube|you tube|tiktok|tik tok|instagram|ig|x|twitter|reddit|snapchat|threads|bluesky|truth social|rumble|telegram|discord|twitch|google|internet|social media|online|apple news|google news|yahoo|msn|newsbreak|smartnews)$"
L["pod"] = L.v.str.contains(BIG, regex=True, na=False) | L.resp.isin(strict)  # podcast source: matches the big-show list OR the respondent is a strict corpus-show listener
L["main"] = L.v.str.contains(MAIN, regex=True, na=False)
L["plat"] = L.v.str.match(PLAT)
L["pod_word"] = L.v.str.contains(r"\bpodcast|\bpod\b", regex=True, na=False)  # narrowest: literally wrote 'podcast'/'pod'
L["corpus"] = L.resp.isin(strict)
# Collapse to respondents: counts of podcast / mainstream / platform sources named (0-3), any-flags, distinct sources.
R = (
    L.groupby("resp")
    .agg(
        npod=("pod", "sum"),
        nmain=("main", "sum"),
        nplat=("plat", "sum"),
        pod=("pod", "max"),
        main=("main", "max"),
        plat=("plat", "max"),
        pod_word=("pod_word", "max"),
        corpus=("corpus", "max"),
        nsrc=("v", "nunique"),
    )
    .reset_index()
)
# Join to the survey; keep respondents with a survey weight.
D = K.merge(R, on="resp")
D = D[D.w.notna()].copy()
# Institutional cynicism from the 4 discovery items, reversed and standardised (higher = more cynical), re-z'd.
core = ["Q35E", "Q35C", "Q34B", "Q33H"]
D["cyn"] = z(-sum(z(num(D, c)) for c in core) / 4)
# Extra demographic controls: sex, income, race, social-media hours, number of platforms used regularly,
# mail-mode response, and the USDA urban-rural continuum code.
D["female"] = (num(D, "GENDER") == 2).astype(float)
D["income"] = num(D, "Q49")
D["white"] = (pd.to_numeric(D.RACE_1, errors="coerce") == 1).astype(float)
D["sm_hours"] = num(D, "Q21")
D["nplatform"] = sum((num(D, f"Q20{c}") <= 2).astype(float) for c in "ABCDEFGHIJKLMN")  # regular use (<= 2) of each of the 14 platforms Q20A-N
D["mail"] = (num(D, "MODE") == 2).astype(float)
D["urc"] = pd.to_numeric(D.Urban_Rural_Continuum_Code, errors="coerce")
FULL = ["female", "white", "income", "sm_hours", "nplatform", "urc", "mail", "nsrc"]  # the FULL control set added to party / attention / age / education


# Control matrix: party dummies + attention, age, education + the FULL demographics.
def base(g):
    X = pd.get_dummies(g.pid, prefix="pid", drop_first=True).astype(float)
    X["attn"] = g.attn
    X["age"] = g.age
    X["edu"] = g.edu
    for e in FULL:
        X[e] = g[e]
    return X


# Weighted least squares of y on the controls plus the columns of interest, survey weight w, HC1 robust SEs.
# (No show clusters here: most respondents in this national frame name no corpus show.)
def run(g, cols, y="cyn"):
    X = base(g)
    for c in cols:
        X[c] = g[c]
    X = sm.add_constant(X, has_constant="add")
    m = pd.concat([X, g[[y]], g.w], axis=1).dropna()
    return sm.WLS(m[y], m[X.columns], weights=m.w).fit(cov_type="HC1"), len(m)


# ---- 4-way segments, reference = mainstream-only
# Diet segments: podcast-only, podcast + mainstream, mainstream-only (the omitted reference), platform-only,
# and a residual 'other-only' (named only unclassified sources).
D["seg"] = np.select(
    [D.pod & ~D.main, D.pod & D.main, ~D.pod & D.main, ~D.pod & ~D.main & D.plat],
    ["pod_only", "pod_plus", "main_only", "platform_only"],
    "other_only",
)
for s in ("pod_only", "pod_plus", "platform_only", "other_only"):
    D[s] = (D.seg == s).astype(float)
# All four segment dummies vs mainstream-only, with weighted segment shares reported alongside.
r, n = run(D, ["pod_only", "pod_plus", "platform_only", "other_only"])
print("=== 1. FOUR-WAY DIET SEGMENTS vs mainstream-only (cynicism, SD; FULL controls; n=%d) ===" % n)
print("   weighted shares:", (D.groupby("seg").w.sum() / D.w.sum()).mul(100).round(1).to_dict())
for s, lab in (
    ("pod_only", "podcast-ONLY"),
    ("pod_plus", "podcast + mainstream"),
    ("platform_only", "platform-ONLY (facebook/youtube/x...)"),
    ("other_only", "other-only (unclassified)"),
):
    ci = r.conf_int().loc[s]
    print(f"   {lab:<40} b={r.params[s]:+.3f}  [{ci[0]:+.3f},{ci[1]:+.3f}]  p={r.pvalues[s]:.2e}")
# Contrasts: is the podcast-only effect distinguishable from platform-only and from podcast + mainstream?
print("   test pod_only = platform_only:  p =", f"{r.t_test('pod_only = platform_only').pvalue:.4f}")
print("   test pod_only = pod_plus:       p =", f"{r.t_test('pod_only = pod_plus').pvalue:.4f}")
# ---- continuous dose
print("\n=== 2. CONTINUOUS DOSE: # podcasts named (0-3), # mainstream named (0-3), interaction ===")
# Per-source dose: each additional podcast named vs each additional mainstream source named, then their interaction.
r, n = run(D, ["npod", "nmain"])
print(
    f"   npod b={r.params['npod']:+.3f} per podcast (p={r.pvalues['npod']:.2e}) | nmain b={r.params['nmain']:+.3f} per mainstream source (p={r.pvalues['nmain']:.2e})"
)
D["npod_x_nmain"] = D.npod * D.nmain
r, n = run(D, ["npod", "nmain", "npod_x_nmain"])
print(
    f"   with interaction: npod {r.params['npod']:+.3f} | nmain {r.params['nmain']:+.3f} | npod x nmain {r.params['npod_x_nmain']:+.3f} (p={r.pvalues['npod_x_nmain']:.3f})"
)
# Weighted cell means of raw cynicism by (podcasts named x mainstream named), counts capped at 2+.
print("   cell means (weighted, raw cynicism SD) by npod x nmain:")
ct = (
    D.groupby([D.npod.clip(upper=2), D.nmain.clip(upper=2)])
    .apply(lambda g: np.average(g.cyn.dropna(), weights=g.w[g.cyn.notna()]))
    .unstack()
    .round(2)
)
ct.index = ["0 pod", "1 pod", "2+ pod"]
ct.columns = ["0 main", "1 main", "2+ main"]
print(ct.to_string())
# ---- alternative podcast definitions
print("\n=== 3. ALTERNATIVE PODCAST DEFINITIONS (each vs everyone else; FULL controls) ===")
# Each alternative definition of 'podcast listener' vs everyone else, broadest to narrowest.
for c, lab in (
    ("pod", "broad: corpus OR big-show list OR 'podcast' word"),
    ("corpus", "corpus shows only (strict, 385)"),
    ("pod_word", "literally wrote 'podcast'/'pod' (no names)"),
):
    D["_x"] = D[c].astype(float)
    r, n = run(D, ["_x"])
    ci = r.conf_int().loc["_x"]
    print(
        f"   {lab:<48} n_exposed={int(D[c].sum()):5d}  b={r.params['_x']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  p={r.pvalues['_x']:.2e}"
    )
# ---- party x segment
print("\n=== 4. GRADIENT WITHIN PARTY (pod_only / pod_plus vs main_only; FULL controls) ===")
# Same segment model within Democrats + leaners and within Republicans + leaners.
for lab, mask in (
    ("Democrats+leaners", D.pid.isin(["D", "leanD"])),
    ("Republicans+leaners", D.pid.isin(["R", "leanR"])),
):
    r, n = run(D[mask], ["pod_only", "pod_plus", "platform_only", "other_only"])
    print(
        f"   {lab:<22} pod_only {r.params['pod_only']:+.3f} (p={r.pvalues['pod_only']:.1e}) | pod_plus {r.params['pod_plus']:+.3f} (p={r.pvalues['pod_plus']:.1e}) | platform_only {r.params['platform_only']:+.3f}  n={n:,}"
    )
