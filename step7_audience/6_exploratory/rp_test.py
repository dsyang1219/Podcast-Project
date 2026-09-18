"""Address vs the R&P populism and anti-media measures in joint audience models (CR2 + wild). Full record F.5.2."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, scipy.stats as st, warnings
from scipy import linalg

warnings.filterwarnings("ignore")
rng = np.random.default_rng(17)
from common import *
import scipy.stats as sst

# Show-level table: R&P / anti-media rates (from rp_antimedia.py) merged with the register scores and
# ideology labels in directive_final.csv, plus the Sobieraj-Berry insult rate from show_measures.csv.
Sh = pd.read_csv(INPUTS / "show_rp_antimedia.csv")
Sh["show_id"] = Sh.show_id.astype(str)
D = pd.read_csv(INPUTS / "directive_final.csv")
D["show_id"] = D.show_id.astype(str)
ref = D.merge(Sh, on="show_id")
c = ref.dropna(subset=["avg_host_cfscore"])  # DIME correlation only for shows with a host donor score
ins = pd.read_csv(INPUTS / "show_measures.csv")
ins["show_id"] = ins.show_id.astype(str)
ref = ref.merge(ins[["show_id", "insult"]], on="show_id")
print("=== A. Validated instruments at the show level (194 shows) ===")
# Part A: how each content measure correlates with address (dir_z), host DIME score, LLM side, and insult.
for x, lab in (
    ("rp_rate", "Rooduijn-Pauwels populism"),
    ("pc_rate", "people-centrism"),
    ("antimedia_rate", "anti-media (neg. media refs)"),
    ("media_rate", "media references (any)"),
):
    print(
        f"  {lab:<30} vs dir_z r={sst.pearsonr(ref.dir_z,ref[x])[0]:+.3f} (p={sst.pearsonr(ref.dir_z,ref[x])[1]:.4f}) | vs DIME r={sst.pearsonr(c[x],c.avg_host_cfscore)[0]:+.3f} | vs LLM side r={sst.pearsonr(ref[x],ref.side)[0]:+.3f} | vs insult r={sst.pearsonr(ref[x],ref.insult)[0]:+.3f}"
    )
# Which shows sit at the top of each measure, with their lean label (names truncated to 20 chars).
t = ref.sort_values("rp_rate")
print(
    "  highest R&P populism:",
    ", ".join(f"{s[:20]}({l})" for s, l in zip(t.show.tail(6), t.lean.tail(6))),
)
t = ref.sort_values("antimedia_rate")
print(
    "  highest anti-media  :",
    ", ".join(f"{s[:20]}({l})" for s, l in zip(t.show.tail(6), t.lean.tail(6))),
)
# Respondent frame for the audience models: strict matches (st) plus the "exclusive diet" outcome
# (named no mainstream outlet), built here the same way common.R is.
L = pd.read_csv(INPUTS / "verbatims_long.csv")
MAIN = r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|washington post|wall street|wsj|reuters|associated press|\bap\b|usa today|local news|newspaper|the hill|politico|axios|bloomberg|cnbc|news ?nation|newsmax"
L["main"] = L.v.str.contains(MAIN, regex=True, na=False)
R = L.groupby("resp").agg(any_main=("main", "max")).reset_index()
g = st.copy()
g["show_id"] = g.show_id.astype(str)
g = g.merge(R, on="resp")
g["excl"] = 1 - g.any_main  # 1 = named only non-mainstream sources
g = g.merge(Sh[["show_id", "rp_rate", "pc_rate", "antimedia_rate"]], on="show_id", how="left")
# Attach each respondent's show content rates and standardise them on the show-level distribution
# (population SD, ddof=0), so each b is per show-SD, on the same footing as dir_z.
for x in ("rp_rate", "pc_rate", "antimedia_rate"):
    g[x.replace("_rate", "_z")] = (g[x] - ref[x].mean()) / ref[x].std(ddof=0)
# Exclude NBC-branded shows, matching the main strict-sample specification.
nb = D[D.show.str.contains("NBC")].show_id
g = g[~g.show_id.isin(nb)]


# Fit y on the controls (party, attention, age, education, share_right, rw_plat, LLM side) plus the
# listed content regressors, and return CR1 / CR2 / wild-bootstrap inference for coefficient j.
def run(y, xs, j):
    X = design(g, IDEO + ("side",))
    for x in xs:
        X[x] = g[x]
    m = pd.concat([X, g[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    return cr_all(m[y].values, m[X.columns].values, m.show_id.values, list(X.columns).index(j))


print(
    "\n=== B. Does validated populism / anti-media content explain the directive -> audience results? (strict, NBC excl., ideology controls) ==="
)
print(f"{'model':<58}{'b/SD':>8}{'CR2 p':>8}{'df':>6}{'WILD p':>8}")
# Part B: horse race. For each outcome: each measure alone, then address and one rival jointly
# (reporting each coefficient in turn), then address against all three rivals at once.
# "|" in a label means "controlling for". b is per SD of the regressor.
for y, lab in (("inst_cyn", "CYNICISM"), ("excl", "EXCLUSIVITY")):
    for xs, j, d in (
        (["dir_z"], "dir_z", "directive alone"),
        (["rp_z"], "rp_z", "R&P populism alone"),
        (["antimedia_z"], "antimedia_z", "anti-media alone"),
        (["dir_z", "rp_z"], "dir_z", "directive | R&P populism"),
        (["dir_z", "rp_z"], "rp_z", "  R&P populism | directive"),
        (["dir_z", "antimedia_z"], "dir_z", "directive | anti-media"),
        (["dir_z", "antimedia_z"], "antimedia_z", "  anti-media | directive"),
        (
            ["dir_z", "rp_z", "pc_z", "antimedia_z"],
            "dir_z",
            "directive | populism + people + anti-media",
        ),
    ):
        r = run(y, xs, j)
        print(f"{lab+': '+d:<58}{r['b']:+8.3f}{r['p2']:8.4f}{r['df']:6.1f}{r['pw']:8.4f}")  # CR2 p with Satterthwaite df, then wild-bootstrap p
    print()
