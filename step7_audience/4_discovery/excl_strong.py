"""Exclusivity (names no institutional source) on listener-directed address among corpus listeners, with show-level and listener-level checks. Paper finding 6."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, re, warnings

warnings.filterwarnings("ignore")
rng = np.random.default_rng(61)
# Survey respondents, their typed news-source verbatims (one row per respondent x source), and show scores.
K = pd.read_pickle(INPUTS / "kett.pkl")
L = pd.read_csv(INPUTS / "verbatims_long.csv")
# Three nested definitions of an 'institutional' source, narrow to broad:
# TVPRINT (TV networks and newspapers) < MAIN (adds wires and political sites; the primary definition)
# < INST (adds magazines, C-SPAN, etc.). Each is a regex applied to the verbatim text.
MAIN = r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|washington post|wall street|wsj|reuters|associated press|\bap\b|usa today|local news|newspaper|the hill|politico|axios|bloomberg|cnbc|news ?nation|newsmax"
TVPRINT = r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|washington post|wall street|wsj|usa today|local news|newspaper"
INST = (
    MAIN
    + r"|reuters|associated press|the hill|politico|axios|bloomberg|cnbc|the guardian|economist|atlantic|time magazine|newsweek|c-span|cspan"
)
# Flag every verbatim, then collapse to the respondent: did they name ANY outlet of each type? nsrc = distinct sources.
L["main"] = L.v.str.contains(MAIN, regex=True, na=False)
L["tvprint"] = L.v.str.contains(TVPRINT, regex=True, na=False)
L["inst"] = L.v.str.contains(INST, regex=True, na=False)
R = (
    L.groupby("resp")
    .agg(
        any_main=("main", "max"),
        any_tvprint=("tvprint", "max"),
        any_inst=("inst", "max"),
        nsrc=("v", "nunique"),
    )
    .reset_index()
)
# Show table: title, LLM side (-1..+1) and listener-directed address (dir_z).
S = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "show", "side", "dir_z"]]


# Respondent-level frame for a match file: attach show scores, drop NBC-branded shows (a network brand, so
# 'names no institutional source' would be trivially false for them), then add outlet flags and controls.
def build(f):
    M = pd.read_csv(f).drop(columns=["show"], errors="ignore").merge(S, on="show_id")
    M = M[~M.show.str.contains("NBC")]
    return M.merge(R, on="resp").merge(K[["resp", "pid", "age", "attn", "edu"]], on="resp")


# OLS of y on dir_z + show side + party dummies + age + attention + education, clustered by show, with a wild
# cluster bootstrap p: refit B times with each show's null-model residuals flipped by a coin toss (Rademacher).
def wild(J, y, B=2000):
    X = pd.get_dummies(J.pid, prefix="pid", drop_first=True).astype(float)
    X["age"] = J.age
    X["attn"] = J.attn
    X["edu"] = J.edu
    X["side"] = J.side  # show ideology as a control, so dir_z is not just proxying for lean
    Xf = X.copy()
    Xf["dir_z"] = J.dir_z
    Xf = sm.add_constant(Xf, has_constant="add")
    X = sm.add_constant(X, has_constant="add")
    m = pd.concat([Xf, J[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    ids = m.show_id.values
    u = np.unique(ids)
    full = sm.OLS(m[y], m[Xf.columns]).fit(cov_type="cluster", cov_kwds={"groups": ids})
    b, t0, pa = full.params["dir_z"], full.tvalues["dir_z"], full.pvalues["dir_z"]  # coefficient, t-stat and analytic (CR1) p for dir_z
    r0 = sm.OLS(m[y], m[[c for c in Xf.columns if c != "dir_z"]]).fit()  # restricted model with dir_z removed (the null)
    res = r0.resid.values
    f0 = r0.fittedvalues.values
    sg = pd.Series(index=u, dtype=float)
    tb = np.empty(B)
    for i in range(B):
        sg[:] = rng.choice([-1.0, 1.0], len(u))  # one +/-1 coin per show
        tb[i] = (
            sm.OLS(f0 + res * sg.reindex(ids).values, m[Xf.columns])
            .fit(cov_type="cluster", cov_kwds={"groups": ids})
            .tvalues["dir_z"]
        )
    return b, pa, (np.abs(tb) >= abs(t0)).mean(), len(m), len(u)  # b, analytic p, wild p, n respondents, n shows


# --- Listener-level exclusivity: for both match samples and all three definitions, is an exclusive diet
# more common among listeners of more listener-directed shows?
print(
    "=== EXCLUSIVITY, listener-level: P(no institutional source) ~ dir_z + show ideology + party + age + attn + edu; clustered by show; NBC excluded ==="
)
print(
    f"{'sample':<12}{'definition of exclusive':<40}{'b/SD':>8}{'analytic p':>12}{'WILD p':>9}{'n':>6}{'shows':>7}"
)
for f, lab in ((INPUTS / "match_expanded.csv", "expanded"), (INPUTS / "match_strict.csv", "strict")):
    J = build(f)
    for col, dl in (
        ("any_main", "no mainstream outlet (primary)"),
        ("any_tvprint", "no TV/print outlet"),
        ("any_inst", "no institutional outlet (broad)"),
    ):
        J["y"] = 1 - J[col]  # exclusive = 1 - named any outlet of this type
        b, pa, pw, n, k = wild(J, "y")
        print(f"{lab:<12}{dl:<40}{b:+8.3f}{pa:12.4f}{pw:9.4f}{n:6d}{k:7d}")
# --- Descriptive: share of exclusive diets by tercile of the show's dir_z (expanded sample).
J = build(INPUTS / "match_expanded.csv")
J["y"] = 1 - J.any_main
print("\nrate of exclusive diets by directive tercile (expanded, NBC excl.):")
J["terc"] = pd.qcut(J.dir_z, 3, labels=["low-directive", "mid", "high-directive"])  # three equal-size groups of respondents by their show's dir_z
print(
    J.groupby("terc", observed=True)
    .agg(exclusive=("y", "mean"), n=("resp", "size"), shows=("show_id", "nunique"))
    .assign(exclusive=lambda d: (d.exclusive * 100).round(0).astype(int).astype(str) + "%")
    .to_string()
)
