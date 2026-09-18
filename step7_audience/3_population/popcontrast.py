"""Population contrasts by breadth of the podcast definition (corpus shows only, any political podcast, the word 'podcast'), for the podcast-namer comparison in finding 4."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, re, warnings
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
# Load the survey (K), the one-row-per-typed-source verbatims (L), and the ids of respondents
# whose typed source exactly matched a corpus show title (strict match = tier 1, "corpus_show").
K = pd.read_pickle(INPUTS / "kett.pkl")
L = pd.read_csv(INPUTS / "verbatims_long.csv")
strict = set(pd.read_csv(INPUTS / "match_strict.csv").resp)


# Survey item as numbers; Kettering codes missing answers as negatives, so those become NaN.
def num(c):
    s = pd.to_numeric(K[c], errors="coerce")
    return s.where(s > 0)


# tier 2: named ANY political podcast/creator — corpus OR big non-corpus shows OR the word podcast
BIG = (
    r"joe rogan|rogan|daily wire|ben shapiro|shapiro|tucker carlson|megyn kelly|charlie kirk|tim pool|"
    r"hasan|piker|glenn beck|dan bongino|bongino|theo von|lex fridman|call her daddy|pod save|meidas|midas|"
    r"bulwark|breaking points|young turks|tyt|pakman|crowder|candace|benny johnson|heather cox|matt walsh|"
    r"michael knowles|andrew klavan|jordan peterson|sam harris|ezra klein|the daily|up first|npr politics|"
    r"\bpodcast|\bpod\b"  # or the respondent simply typed the word "podcast"/"pod"
)
# Flag each typed source against BIG, then collect every respondent who named at least one such podcast.
L["anypod"] = L.v.str.contains(BIG, regex=True, na=False)
anypod = set(L[L.anypod].resp)
# Three mutually exclusive tiers, narrowest first: corpus show > other podcast > no podcast.
# A corpus-show namer stays "corpus_show" even if they also typed "podcast".
K["tier"] = np.where(
    K.resp.isin(strict), "corpus_show", np.where(K.resp.isin(anypod), "other_podcast", "no_podcast")
)
# Restrict to respondents who typed at least one news source (the population these contrasts describe).
K = K[K.resp.isin(set(L.resp))]  # named >=1 source
K = K[K.panel | (~K.panel)]  # keep all; malformed rows lack pid anyway
print("tiers among respondents naming >=1 source:")
print(K.tier.value_counts().to_string())

# Outcomes: institutional-trust and democratic-attitude items, plus ap_gap (the party favourability
# gap from Q39A/Q39B, built upstream when kett.pkl was made). Labels are for the printout.
OUT = {
    "ap_gap": "party favorability gap",
    "Q33E": "elections administered well",
    "Q34B": "know how to reach officials",
    "Q35E": "laws uphold freedom/justice",
    "Q33H": "freedom of press working",
    "Q35C": "govt includes people like me",
    "Q32C": "assume fraud when surprised",
    "Q32F": "radicals may protest",
    "Q36K": "political violence",
    "Q31": "leaders held accountable",
    "Q23": "citizen power",
    "Q22": "feel valued/respected",
    "Q29": "democracy doing",
}


# z-score so every coefficient is read in SD units of the outcome.
def z(s):
    return (s - s.mean()) / s.std()


rows = []
# One control matrix reused for every outcome: party-ID dummies (one category dropped as reference),
# political attention, age, education, plus the two tier dummies. The omitted tier is "no_podcast",
# so each b is the gap between podcast namers and people who named only TV/print/web sources.
X0 = pd.get_dummies(K.pid, prefix="pid", drop_first=True).astype(float)
X0["attn"] = K.attn
X0["age"] = K.age
X0["edu"] = K.edu
X0["other_podcast"] = (K.tier == "other_podcast").astype(float)
X0["corpus_show"] = (K.tier == "corpus_show").astype(float)
X0 = sm.add_constant(X0, has_constant="add")
# One survey-weighted regression per outcome with HC1 robust SEs (respondent-level model, so no
# show clustering is needed here).
for v, lab in OUT.items():
    y = K[v] if v == "ap_gap" else num(v)  # ap_gap is already numeric; survey items need missing codes dropped
    y = z(y)
    m = pd.concat([X0, y.rename("y"), K.w], axis=1).dropna()  # listwise-drop missing rows; keep the weight column
    r = sm.WLS(m.y, m[X0.columns], weights=m.w).fit(cov_type="HC1")  # WLS with survey weight w; HC1 heteroskedasticity-robust SEs
    rows.append(
        (
            lab,
            len(m),
            r.params["other_podcast"],
            r.pvalues["other_podcast"],
            r.params["corpus_show"],
            r.pvalues["corpus_show"],
        )
    )
# Collect the results and BH-correct the 13 p-values separately for each tier.
R = pd.DataFrame(rows, columns=["outcome", "n", "b_otherpod", "p_otherpod", "b_corpus", "p_corpus"])
R["q_otherpod"] = multipletests(R.p_otherpod, method="fdr_bh")[1]
R["q_corpus"] = multipletests(R.p_corpus, method="fdr_bh")[1]
print("\n=== C. POPULATION CONTRAST, weighted, full sample, controls: pid attn age edu ===")
print("    reference = named only TV/print/web sources.  b in SD units of outcome.")
print(R.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if abs(x) < 50 else f"{x:.0f}"))  # print n as an integer, b and p signed to 3 dp
