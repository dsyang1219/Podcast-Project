"""Classify every Kettering verbatim (institution / person-branded / platform / podcast / local / other) and build the news-diet segments for the full sample. Population finding, finding 4."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, re, collections, statsmodels.api as sm, warnings
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
# Inputs: the full survey (all ~20k respondents, not just podcast listeners), every typed
# news-source mention, and the corpus show/host list (used to recognise host names).
K = pd.read_pickle(INPUTS / "kett.pkl")
L = pd.read_csv(INPUTS / "verbatims_long.csv")
sh = pd.read_csv(DATA / "output/shows_205_host_dime_scores.csv")


# A survey item as numbers, with Kettering's negative missing codes (-98/-99) set to NaN.
def num(c):
    s = pd.to_numeric(K[c], errors="coerce")
    return s.where(s > 0)


# ---- classify each verbatim: institution / person-branded / platform / podcast-ish / local / other
# Three regex dictionaries. INST = news organisations and aggregators; PLAT = bare platform or
# generic answers ("facebook", "tv", "the news"), anchored ^...$ so only the whole answer counts;
# PERS = named hosts, pundits and personalities (matched anywhere in the answer).
INST = r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|ny times|washington post|wall street|wsj|reuters|associated press|\bap\b|usa today|bloomberg|the hill|politico|axios|newsmax|oan|one america|the guardian|economist|atlantic|time magazine|newsweek|huffpost|daily mail|epoch|breitbart|vice|vox\b|slate|salon|forbes|cnbc|news ?nation|local news|newspaper|c-span|cspan|apple news|google news|yahoo|msn|smartnews|newsbreak|flipboard|the daily wire|daily wire|blaze"
PLAT = r"^(facebook|fb|youtube|you tube|tiktok|tik tok|instagram|ig|x|twitter|reddit|snapchat|threads|bluesky|truth social|rumble|telegram|discord|twitch|google|internet|social media|online|radio|tv|television|news|podcasts?|the news|nextdoor|substack|linkedin|whatsapp)$"
# Build a set of full corpus host names (first + last), splitting co-host strings like "A, B and C".
hosts = set()
for h in sh.host.dropna():
    for p in re.split(r",| and ", str(h)):
        p = p.strip().lower()
        if len(p.split()) >= 2:  # keep two-word names only, so a lone surname cannot match by accident
            hosts.add(p)
PERS = r"joe rogan|rogan|tucker|shapiro|megyn kelly|charlie kirk|tim pool|glenn beck|bongino|hasan|piker|theo von|lex fridman|heather cox|aaron parnas|parnas|brian tyler|pakman|crowder|candace|benny johnson|matt walsh|knowles|jordan peterson|sam harris|ezra klein|rachel maddow|maddow|hannity|jesse watters|watters|laura ingraham|ingraham|anderson cooper|jake tapper|lawrence o'?donnell|nicolle wallace|chris hayes|joy reid|ari melber|don lemon|jimmy dore|kyle kulinski|david pakman|meidas|midas|ben meiselas|mark levin|levin|dave rubin|stephen colbert|colbert|jon stewart|john oliver|bill maher|maher|trevor noah|seth meyers|kimmel|fallon|lester holt|david muir|norah|scott pelley|gutfeld|bret baier|baier|shawn ryan|andrew schulz|schulz|jesse kelly|deace|steve bannon|bannon|alex jones|infowars|dan crenshaw|aoc|ocasio|bernie|trump|elon|musk|robert reich|reich|russell brand|sam seder|majority report|olbermann|tim miller|bulwark|hugh hewitt|carville|katie phang|jim acosta|acosta|chris cuomo|cuomo|kaitlan|erin burnett|wolf blitzer|fareed|zakaria|george stephanopoulos|stephanopoulos|dana bash|abby phillip|philip defranco|defranco|v spehar|under the desk|carlos eduardo|espina|philip|krystal|saagar|breaking points|young turks|tyt|cenk|ana kasparian|kasparian"


# Assign one category per typed answer. Order matters: a whole-answer platform word wins first,
# then a person's name (so "fox - hannity" is person-branded), then an institution, then
# unnamed podcasts, then local-TV cues (call signs like KABC/WGN), else "other".
def cls(v):
    v = str(v).lower().strip()
    if re.fullmatch(PLAT, v):
        return "platform/generic"
    if re.search(PERS, v) or v in hosts:
        return "person-branded"
    if re.search(INST, v):
        return "institution"
    if re.search(r"podcast|\bpod\b", v):
        return "podcast(unnamed)"
    if re.search(r"local|channel \d|wxyz|kron|kabc|wgn|\bk[a-z]{3}\b|\bw[a-z]{3}\b", v):  # US station call signs start with K or W + 3 letters
        return "local"
    return "other"


# --- A. Population description: how the whole sample's mentions split across categories,
# how concentrated the answers are on a few strings, and which people are named most.
L["cls"] = L.v.apply(cls)
print("=== A. WHAT 20,464 AMERICANS NAME AS THEIR TOP-3 NEWS SOURCES (51,082 mentions) ===")
print(L.cls.value_counts(normalize=True).mul(100).round(1).to_string())
print(
    "\nconcentration: top 10 strings =",
    f"{L.v.value_counts().head(10).sum()/len(L):.1%} of mentions; top 50 =",
    f"{L.v.value_counts().head(50).sum()/len(L):.1%}; distinct strings = {L.v.nunique():,}",
)
print("\ntop 15 PERSON-branded sources named:")
print(L[L.cls == "person-branded"].v.value_counts().head(15).to_string())

# ---- respondent-level: person-branded diet share
# Collapse mentions to one row per respondent: how many sources named, and the share that are
# person-branded / institutional / platform. any_person = named at least one personality.
R = (
    L.groupby("resp")
    .agg(
        n=("v", "size"),
        person=("cls", lambda x: (x == "person-branded").mean()),
        inst=("cls", lambda x: (x == "institution").mean()),
        plat=("cls", lambda x: (x == "platform/generic").mean()),
    )
    .reset_index()
)
R["any_person"] = (R.person > 0).astype(float)
D = K.merge(R, on="resp", how="inner")  # inner: keep respondents who typed at least one source
print(
    f"\nrespondents naming >=1 PERSON as a top news source: {D.any_person.mean():.1%}  (n={int(D.any_person.sum()):,} of {len(D):,})"
)
print("by party:", D.groupby("pid").any_person.mean().mul(100).round(1).to_dict())
print(
    "by age band:",
    D.assign(ab=pd.cut(D.age, [17, 29, 44, 64, 120], labels=["18-29", "30-44", "45-64", "65+"]))  # bin edges are exclusive on the left: 18-29, 30-44, ...
    .groupby("ab", observed=True)
    .any_person.mean()
    .mul(100)
    .round(1)
    .to_dict(),
)

# --- C. Exploratory: is a person-branded news diet associated with democracy attitudes across the
# whole sample? One weighted regression per outcome, any_person as the exposure, base controls.
print(
    "\n=== C. PERSON-BRANDED DIET vs democracy attitudes (exploratory; weighted; controls party/attn/age/edu; vs institution-only) ==="
)
# Outcome items and short labels. "(rev)" marks items whose scale runs the other way; nothing is
# reversed here, so read the sign of those two rows accordingly.
OUT = {
    "Q31": "leaders held accountable",
    "Q35E": "laws uphold justice",
    "Q33E": "elections run well",
    "Q33H": "press freedom working",
    "Q35C": "govt includes people like me",
    "Q34B": "know how to reach officials",
    "Q32F": "radicals may protest",
    "Q32C": "assume fraud if surprised",
    "Q36K": "political violence",
    "Q29": "democracy doing",
    "Q22": "feel valued",
    "Q23": "citizen power",
    "ap_gap": "party favorability gap",
    "Q4": "loneliness(rev)",
    "Q8": "info overload(rev)",
}
rows = []
for v, lab in OUT.items():
    # ap_gap is already numeric; raw items are converted and negative missing codes dropped.
    # (The `if False` branch is dead code; the last expression is what runs.)
    y = (
        D[v]
        if v == "ap_gap"
        else (
            num(v).reindex(D.index)
            if False
            else pd.to_numeric(D[v], errors="coerce").where(lambda s: s > 0)
        )
    )
    y = (y - y.mean()) / y.std()  # standardise so b is in SD units of the outcome
    # Same control set as design() in common.py: party dummies, attention, age, education.
    X = pd.get_dummies(D.pid, prefix="pid", drop_first=True).astype(float)
    X["attn"] = D.attn
    X["age"] = D.age
    X["edu"] = D.edu
    X["any_person"] = D.any_person
    X = sm.add_constant(X, has_constant="add")
    m = pd.concat([X, y.rename("y"), D.w], axis=1).dropna()
    r = sm.WLS(m.y, m[X.columns], weights=m.w).fit(cov_type="HC1")  # survey-weighted; HC1 robust SEs (no show clusters here)
    rows.append((lab, len(m), r.params["any_person"], r.pvalues["any_person"]))
T = pd.DataFrame(rows, columns=["outcome", "n", "b_SD", "p"])
T["q"] = multipletests(T.p, method="fdr_bh")[1]  # Benjamini-Hochberg q across the 15 outcomes
print(
    T.sort_values("p").to_string(
        index=False, float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.0f}"
    )
)
