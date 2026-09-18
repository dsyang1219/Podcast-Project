"""The 113-item outcome scan of the discovery sample against address exposure. Disclosure section G.1. Writes outputs/outcome_scan.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, warnings

warnings.filterwarnings("ignore")
# Discovery-sample frame (d_pop.pkl): matched respondents with their show's dir_z, the standard controls, and every raw Kettering item.
d = pd.read_pickle(INPUTS / "d_pop.pkl")
# Kettering codebook, VARIABLES sheet: column 0 = variable name, column 2 = question wording (used only to label the output).
cb = (
    pd.read_excel(
        DATA / "external/kettering/KETTERING_CODEBOOK.xlsx", sheet_name="VARIABLES", header=None
    )
    .fillna("")
    .astype(str)
)
lab = {r[0]: r[2] for r in cb.values.tolist() if r[0]}  # variable name -> question label

# Never treated as outcomes: identifiers, weights, design/mode fields, the controls themselves (age, edu, attention, party Q41-43, favourability Q39), the open-text source items Q17, and demographics.
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
    "Q17_1",
    "Q17_2",
    "Q17_3",
    "Q17_all_modes",
    "Q28",
    "Q45",
    "Q49",
    "Q51",
    "Q48",
}
# Candidate outcomes: every remaining item with >= 300 valid answers and 2-12 distinct values (a Likert-type scale, not an ID or continuous field).
cand = []
for c in d.columns:
    if (
        c in EXCL
        or c.endswith("_TEXT")  # open-text follow-ups
        or c
        in (
            "resp",
            "dir_z",
            "share_right",
            "pid",
            "ap_gap",
            "attn",
            "age",
            "edu",
            "w",
            "panel",
            "repfav",
            "demfav",
            "show_id",
            "n_listeners",
            "charted",
            "best_rank",
            "hours",
            "log_listeners",
            "log_hours",
        )
    ):
        continue
    s = pd.to_numeric(d[c], errors="coerce")
    s = s.where(s > 0)  # negative codes = missing
    if s.notna().sum() >= 300 and 2 <= s.nunique() <= 12:  # Likert-style item with enough answers
        cand.append(c)
print(f"screening {len(cand)} outcome variables\n")

# One fixed design for every outcome: party dummies, attention, age, education and the exposure dir_z (the discovery-model controls, no ideology terms).
X0 = pd.get_dummies(d["pid"], prefix="pid", drop_first=True).astype(float)
X0["attn"] = d.attn
X0["age"] = d.age
X0["edu"] = d.edu
X0["dir_z"] = d.dir_z
X0 = sm.add_constant(X0, has_constant="add")
rows = []
# Regress each candidate on the design with show-clustered (CR1) SEs; keep the dir_z slope, its SD-unit version, p and n.
for c in cand:
    y = pd.to_numeric(d[c], errors="coerce")
    y = y.where(y > 0)
    m = pd.concat([X0, y.rename("y"), d[["show_id"]]], axis=1).dropna()
    if len(m) < 300:  # skip outcomes with too few complete cases
        continue
    r = sm.OLS(m.y, m[X0.columns]).fit(cov_type="cluster", cov_kwds={"groups": m.show_id})
    sd = m.y.std()  # outcome SD, so beta_sd reads in SD units
    rows.append(
        (c, lab.get(c, "")[:60], r.params["dir_z"], r.params["dir_z"] / sd, r.pvalues["dir_z"], len(m))
    )
R = pd.DataFrame(rows, columns=["var", "label", "b", "beta_sd", "p", "n"]).sort_values("p")  # most significant first
from statsmodels.stats.multitest import multipletests

# Benjamini-Hochberg false-discovery-rate adjustment across all tests in the scan.
R["q_bh"] = multipletests(R.p, method="fdr_bh")[1]
R.to_csv(OUTPUTS / "outcome_scan.csv", index=False)
# Headline: nominal p<.05 hits versus the ~5% expected by chance, and how many survive FDR.
print(
    f"tests={len(R)}  p<.05: {(R.p<.05).sum()}  (chance expects {0.05*len(R):.1f})  BH survivors: {(R.q_bh<.05).sum()}\n"
)
print(R.head(16).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
