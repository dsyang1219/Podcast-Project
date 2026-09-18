"""Label validation against listeners: correlate each show's LLM ideology label with the party identification of its own Kettering listeners (expanded match, validation only). Paper finding 1, audience row."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, scipy.stats as st, warnings

warnings.filterwarnings("ignore")
# --- Inputs: the survey, the respondent-show match, and show-level attributes.
# The expanded (host-name) match is used here because more listeners per show makes the
# audience composition estimates steadier; it is NOT used for the outcome tests.
K = pd.read_pickle(INPUTS / "kett.pkl")
M = pd.read_csv(INPUTS / "match_expanded.csv").drop(
    columns=["show"], errors="ignore"
)  # expanded: VALIDATION of labels, not the outcome test
S = pd.read_csv(INPUTS / "directive_final.csv")[
    ["show_id", "show", "lean", "side", "avg_host_cfscore", "dir_z"]
]  # LLM label (lean/side), DIME host score, and the address composite per show
sm_ = pd.read_csv(INPUTS / "show_measures.csv")[
    ["show_id", "religious", "profanity", "second_person", "imperatives"]
]  # dictionary register rates per show (only 'religious' is used below)
S = S.merge(sm_, on="show_id", how="left")
# One row per respondent-show pair, carrying the respondent's demographics and the show's attributes.
J = M.merge(
    K[["resp", "pid", "age", "edu", "Q44", "Q46", "GENDER", "Urban_Rural_Continuum_Code"]], on="resp"
).merge(S, on="show_id")
# Respondent-level binary indicators (leaners are folded into their party).
J["dem"] = J.pid.isin(["D", "leanD"]).astype(float)
J["rep"] = J.pid.isin(["R", "leanR"]).astype(float)
J["church_weekly"] = (pd.to_numeric(J.Q44, errors="coerce") <= 2).astype(float)  # Q44 <= 2: attends at least weekly
J["born_again"] = (pd.to_numeric(J.Q46, errors="coerce") == 1).astype(float)  # Q46 == 1: yes, born-again/evangelical
J["female"] = (pd.to_numeric(J.GENDER, errors="coerce") == 2).astype(float)  # GENDER == 2: female
# Collapse to one row per show: audience composition (means of the indicators) next to the show's own labels.
A = (
    J.groupby("show_id")
    .agg(
        show=("show", "first"),
        n=("resp", "nunique"),  # number of distinct listeners matched to the show
        dem=("dem", "mean"),
        rep=("rep", "mean"),
        age=("age", "mean"),
        church=("church_weekly", "mean"),
        born=("born_again", "mean"),
        female=("female", "mean"),
        lean=("lean", "first"),
        side=("side", "first"),
        cf=("avg_host_cfscore", "first"),
        religious=("religious", "first"),
        dir_z=("dir_z", "first"),
    )
    .reset_index()
)
A["aud_lean"] = A.rep - A.dem  # audience lean: share Republican minus share Democratic (-1..+1)
print("=== B. AUDIENCE COMPOSITION as an independent validation of show labels ===")
# Run the validation twice, requiring at least 5 and then at least 10 matched listeners per show,
# to check the correlations are not driven by shows with a handful of listeners.
for k in (5, 10):
    a = A[A.n >= k]
    r1, p1 = st.pearsonr(a.aud_lean, a.side)  # does the LLM side label track who actually listens?
    b = a.dropna(subset=["cf"])  # DIME score exists only for hosts with a donation record
    r2, p2 = st.pearsonr(b.aud_lean, b.cf)  # same check against the donation-based host ideology
    print(f"\nshows with >={k} listeners (n={len(a)}):")
    print(f"  audience lean (R%-D%) vs LLM side label   r={r1:+.3f} p={p1:.4f}")
    print(f"  audience lean (R%-D%) vs DIME host cfscore r={r2:+.3f} p={p2:.4f}  (n={len(b)})")
    print(
        f"  L-labeled shows: mean {a[a.lean=='L'].dem.mean():.0%} Democratic audience | R-labeled: mean {a[a.lean=='R'].rep.mean():.0%} Republican audience"
    )
    # Face-validity checks on other show measures: religious language should go with religious audiences ...
    r3, p3 = st.pearsonr(a.religious, a.church)
    r4, p4 = st.pearsonr(a.religious, a.born)
    print(
        f"  show RELIGIOUS-language score vs audience weekly church r={r3:+.3f} p={p3:.4f} | vs born-again share r={r4:+.3f} p={p4:.4f}"
    )
    # ... and does listener-directed address (dir_z) relate to audience age or gender at all?
    r5, p5 = st.pearsonr(a.dir_z, a.age)
    r6, p6 = st.pearsonr(a.dir_z, a.female)
    print(
        f"  show DIRECTIVE score vs audience mean age r={r5:+.3f} p={p5:.4f} | vs female share r={r6:+.3f} p={p6:.4f}"
    )
# Show-by-show listing (>= 10 listeners), sorted from most Democratic to most Republican audience,
# flagging shows whose label disagrees with its audience by more than 20 points.
a = A[A.n >= 10].sort_values("aud_lean")
print("\nshows with >=10 listeners — label vs audience (mismatches flagged):")
for _, r in a.iterrows():
    flag = (
        "  <-- MISMATCH"
        if (r.lean == "L" and r.aud_lean > 0.2) or (r.lean == "R" and r.aud_lean < -0.2)  # L show with R-leaning audience, or vice versa
        else ""
    )
    print(
        f"  {r.show[:38]:<38} n={r.n:3d}  label={r.lean}  D={r.dem:.0%} R={r.rep:.0%}  age={r.age:.0f}{flag}"
    )
