"""Profile all usable Kettering items on the news-diet segments (podcast-only, podcast+mainstream, platform-only vs mainstream-only), BH-corrected. Paper finding 4 (engaged anti-institutionalism). Writes outputs/pop_profile.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, re, warnings
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
# Survey (K), typed sources (L, one row per source), strict corpus-show matches, and the Kettering
# codebook so each output row can carry a human-readable item label.
K = pd.read_pickle(INPUTS / "kett.pkl")
L = pd.read_csv(INPUTS / "verbatims_long.csv")
strict = set(pd.read_csv(INPUTS / "match_strict.csv").resp)
cb = (
    pd.read_excel(
        DATA / "external/kettering/KETTERING_CODEBOOK.xlsx", sheet_name="VARIABLES", header=None
    )
    .fillna("")
    .astype(str)
)
lab = {r[0]: r[2] for r in cb.values.tolist() if r[0]}  # variable name -> label text (codebook columns 0 and 2)


# Item as numbers with negative missing codes dropped. Reads from D, the merged frame built below.
def num(c):
    s = pd.to_numeric(D[c], errors="coerce")
    return s.where(s > 0)


# Regexes that classify each typed source. BIG = big political podcasts/hosts (corpus or not) or the
# bare word "podcast"; MAIN = mainstream TV/print/wire/web outlets; PLAT = the whole answer is a social
# platform or a generic "internet"-type source (anchored ^...$, so "fox on youtube" is not a platform).
BIG = r"joe rogan|rogan|daily wire|shapiro|tucker|megyn kelly|charlie kirk|tim pool|hasan|piker|glenn beck|bongino|theo von|lex fridman|pod save|meidas|midas|bulwark|breaking points|young turks|tyt|pakman|crowder|candace|benny johnson|heather cox|matt walsh|knowles|klavan|jordan peterson|sam harris|ezra klein|the daily|up first|npr politics|\bpodcast|\bpod\b"
MAIN = r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|washington post|wall street|wsj|reuters|associated press|\bap\b|usa today|local news|newspaper|the hill|politico|axios|bloomberg|cnbc|news ?nation|newsmax"
PLAT = r"^(facebook|fb|youtube|you tube|tiktok|tik tok|instagram|ig|x|twitter|reddit|snapchat|threads|bluesky|truth social|rumble|telegram|discord|twitch|google|internet|social media|online|apple news|google news|yahoo|msn|newsbreak|smartnews)$"
# Flag each typed source. A strict corpus-show match counts as a podcast even if the regex misses it.
L["pod"] = L.v.str.contains(BIG, regex=True, na=False) | L.resp.isin(strict)
L["main"] = L.v.str.contains(MAIN, regex=True, na=False)
L["plat"] = L.v.str.match(PLAT)
# Collapse to one row per respondent: named any podcast / any mainstream / any platform, and how
# many distinct sources they typed (nsrc, later a control for diet breadth).
R = (
    L.groupby("resp")
    .agg(pod=("pod", "max"), main=("main", "max"), plat=("plat", "max"), nsrc=("v", "nunique"))
    .reset_index()
)
# Merge onto the survey; keep only respondents with a survey weight (the weighted models need it).
D = K.merge(R, on="resp")
D = D[D.w.notna()].copy()
# Diet segments, checked in order: podcast without mainstream; podcast plus mainstream; mainstream
# without podcast (the reference group); neither but a platform; everything else -> "other".
D["seg"] = np.select(
    [D.pod & ~D.main, D.pod & D.main, ~D.pod & D.main, ~D.pod & ~D.main & D.plat],
    ["pod_only", "pod_plus", "main_only", "plat_only"],
    "other",
)
# Segment dummies. main_only gets no dummy, so it is the omitted reference category in every model.
for s in ("pod_only", "pod_plus", "plat_only", "other"):
    D[s] = (D.seg == s).astype(float)
# Extra controls beyond the core ladder: gender, income, race, social-media hours, platform count, rurality.
D["female"] = (num("GENDER") == 2).astype(float)  # GENDER 2 = female
D["income"] = num("Q49")
D["white"] = (pd.to_numeric(D.RACE_1, errors="coerce") == 1).astype(float)  # RACE_1 1 = white
D["sm_hours"] = num("Q21")
D["nplat"] = sum((num(f"Q20{c}") <= 2).astype(float) for c in "ABCDEFGHIJKLMN")  # number of the 14 platforms used regularly (<= 2 on the frequency scale)
D["urc"] = pd.to_numeric(D.Urban_Rural_Continuum_Code, errors="coerce")  # USDA rural-urban continuum code (1 = most urban)
# derived theory items never examined
D["strong_partisan"] = np.where(
    D.pid.isin(["R", "D"]), ((num("Q42") == 1) | (num("Q42.0") == 1)).astype(float), np.nan
)  # among identifiers: strong vs not
# Columns never profiled: ids/admin fields, the controls themselves, the party-ID and favourability
# items that feed pid/ap_gap, and the open-text items (Q17 sources, Q28 democracy).
EXCL = {
    "uig",
    "EMPLOYEE_KEY_VALUE",
    "MODE",
    "SAMP_TYPE",
    "AGE",
    "EDU",
    "GENDER",
    "WEIGHT",
    "WEIGHT_PROJ",
    "DEMO_DIVISION",
    "DEMO_REGION",
    "Urban_Rural_Continuum_Code",
    "Q19",
    "Q41",
    "Q42",
    "Q42.0",
    "Q43",
    "Q39A",
    "Q39B",
    "Q49",
    "Q51",
    "Q48",
    "Q45",
    "Q17_1",
    "Q17_2",
    "Q17_3",
    "Q17_all_modes",
    "Q28",
}
# Auto-select profilable items: every Q*/RACE* column not excluded, not a free-text "_TEXT" field, with
# 2-12 distinct valid values (Likert-style, not continuous or constant) and at least 3000 valid answers.
items = [
    c
    for c in D.columns
    if c not in EXCL
    and not c.endswith("_TEXT")
    and c.startswith(("Q", "RACE"))
    and 2 <= num(c).nunique() <= 12
    and num(c).notna().sum() >= 3000
]
items += ["strong_partisan", "ap_gap"]  # plus the two derived items (built above / upstream in kett.pkl)
print(
    f"profiling {len(items)} items × 3 diet segments vs mainstream-only; n≈{len(D):,}; weighted; controls party/attn/age/edu/gender/race/income/sm-hours/platforms/urban-rural/nsrc\n"
)
rows = []
# One weighted regression per item: z-scored item on the segment dummies + the full control ladder.
for c in items:
    y = D[c] if c in ("strong_partisan", "ap_gap") else num(c)
    y = (y - y.mean()) / y.std()  # SD units, so effects are comparable across items
    X = pd.get_dummies(D.pid, prefix="pid", drop_first=True).astype(float)  # party-ID dummies, reference category dropped
    for e in ("attn", "age", "edu", "female", "white", "income", "sm_hours", "nplat", "urc", "nsrc"):
        X[e] = D[e]
    for s in ("pod_only", "pod_plus", "plat_only", "other"):
        X[s] = D[s]
    X = sm.add_constant(X, has_constant="add")
    m = pd.concat([X, y.rename("y"), D.w], axis=1).dropna()
    if len(m) < 3000:  # skip items that lose too many rows to listwise deletion
        continue
    r = sm.WLS(m.y, m[X.columns], weights=m.w).fit(cov_type="HC1")  # survey-weighted, HC1 robust SEs (respondent level)
    rows.append(
        (
            c,
            lab.get(c, c)[:64],
            len(m),
            r.params["pod_only"],
            r.pvalues["pod_only"],
            r.params["pod_plus"],
            r.pvalues["pod_plus"],
            r.params["plat_only"],
            r.pvalues["plat_only"],
        )
    )
# Assemble the results table; BH-correct p-values within each segment across all items tested.
P = pd.DataFrame(
    rows,
    columns=[
        "var",
        "label",
        "n",
        "b_podonly",
        "p_podonly",
        "b_podplus",
        "p_podplus",
        "b_platonly",
        "p_platonly",
    ],
)
# One BH family per segment (all items tested against that segment).
for s in ("podonly", "podplus", "platonly"):
    P["q_" + s] = multipletests(P["p_" + s], method="fdr_bh")[1]
P.to_csv(OUTPUTS / "pop_profile.csv", index=False)
print(
    f"items surviving BH (q<.05): podcast-only {(P.q_podonly<.05).sum()} | podcast+mainstream {(P.q_podplus<.05).sum()} | platform-only {(P.q_platonly<.05).sum()}  of {len(P)}\n"
)
# Headline: podcast-only effects that survive BH, largest |b| first. The sign follows the raw item
# coding (e.g. 1 = strongly agree), so read each row with its label.
print(
    "=== PODCAST-ONLY vs mainstream-only: largest |b| among BH survivors (SD units; sign is on the raw item scale — see label) ==="
)
s = P[P.q_podonly < 0.05].reindex(
    P[P.q_podonly < 0.05].b_podonly.abs().sort_values(ascending=False).index
)
print(
    s.head(22)[["var", "label", "b_podonly", "q_podonly", "b_podplus", "b_platonly"]].to_string(  # top 22 by |b|
        index=False, float_format=lambda x: f"{x:+.3f}"
    )
)
# Divergence: where podcast-only and platform-only differ most. Items where both segments move the
# same way just reflect a non-mainstream diet; large gaps point to something podcast-specific.
print(
    "\n=== items where PODCAST-ONLY and PLATFORM-ONLY DIVERGE most (podcast-specific, not just 'non-institutional') ==="
)
P["dvg"] = P.b_podonly - P.b_platonly
d = P[(P.q_podonly < 0.05)].reindex(P[(P.q_podonly < 0.05)].dvg.abs().sort_values(ascending=False).index)
print(
    d.head(10)[["var", "label", "b_podonly", "b_platonly", "dvg"]].to_string(
        index=False, float_format=lambda x: f"{x:+.3f}"
    )
)
