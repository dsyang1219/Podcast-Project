"""Build the Kettering verbatim table and the strict (exact-title) respondent-show match. Writes inputs/verbatims_long.csv and inputs/match_strict.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, re, collections

# --- Load the raw Kettering-Gallup survey file and give every respondent a row-number id (`resp`).
B = str(ROOT) + "/"
K = pd.read_csv(
    B + "data/external/kettering/KETTERING_DATA_Y1_WEIGHTED.csv", low_memory=False, encoding="cp1252"
)  # cp1252: the vendor file is Windows-encoded
K = K.reset_index().rename(columns={"index": "resp"}).copy()  # resp = row position; used as the respondent key everywhere else
dirf = pd.read_csv(INPUTS / "directive_final.csv")  # dir_z per show_id
sh = pd.read_csv(DATA / "output/shows_205_host_dime_scores.csv")  # show titles, hosts and DIME scores for the 205 corpus shows
strict = pd.read_csv(INPUTS / "kettering_strict.csv")  # resp, verbatim, show

# ---- show-level frame
# Attach the show title (and host) to the scored shows; fill titles missing from directive_final.
S = dirf.merge(sh[["show_id", "show", "host"]], on="show_id", how="left", suffixes=("", "_y"))
S["show"] = S["show"].fillna(S.get("show_y"))
S = S[["show_id", "show", "dir_z", "lean", "side", "avg_host_cfscore"]].drop_duplicates("show_id")
print("shows with dir_z:", len(S))

# ---- verbatim long frame
# Stack the three open-text "top news source" answers (Q17_1..3) into one row per mention.
L = []
for c in ("Q17_1", "Q17_2", "Q17_3"):
    t = K[["resp", c]].dropna()
    t.columns = ["resp", "v"]
    L.append(t)
L = pd.concat(L)
L["v"] = L.v.astype(str).str.strip().str.lower()  # normalise: lower-case, trim whitespace
L = L[(L.v != "") & (L.v != "nan")]  # drop blanks and the string "nan" left by missing cells
L.to_csv(INPUTS / "verbatims_long.csv", index=False)

# ---- STRICT matches -> show_id
# kettering_strict.csv already holds the hand-checked exact-title matches (resp -> show title);
# here we translate the title into the corpus show_id by case-insensitive lookup.
name2id = dict(zip(S.show.str.lower(), S.show_id))
strict["show_id"] = strict.show.str.lower().map(name2id)
print(
    "strict mentions:",
    len(strict),
    "| with dir_z:",
    strict.show_id.notna().sum(),  # matches whose show actually has a dir_z score
    "| respondents:",
    strict.resp.nunique(),
)
strict[["resp", "verbatim", "show", "show_id"]].to_csv(INPUTS / "match_strict.csv", index=False)  # the match of record
