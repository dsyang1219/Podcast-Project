"""Small-sample inference (CR2 + wild bootstrap) for the game-frame vs address models. Full record F.5.1."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, warnings
from scipy import stats, linalg

warnings.filterwarnings("ignore")
rng = np.random.default_rng(9)
from common import *

# Show-level game-frame rates written by horserace.py, and the show table used to standardise them.
Sh = pd.read_csv(INPUTS / "show_gameframe.csv")
Sh["show_id"] = Sh.show_id.astype(str)
D = pd.read_csv(INPUTS / "directive_final.csv")
D["show_id"] = D.show_id.astype(str)
# Exclusivity outcome (no mainstream outlet named), as in horserace.py.
L = pd.read_csv(INPUTS / "verbatims_long.csv")
MAIN = r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|washington post|wall street|wsj|reuters|associated press|\bap\b|usa today|local news|newspaper|the hill|politico|axios|bloomberg|cnbc|news ?nation|newsmax"
L["main"] = L.v.str.contains(MAIN, regex=True, na=False)
R = L.groupby("resp").agg(any_main=("main", "max")).reset_index()
# Strict Kettering frame + exclusivity + the show's game-frame rates.
g = st.copy()
g["show_id"] = g.show_id.astype(str)
g = g.merge(R, on="resp")
g["excl"] = 1 - g.any_main
g = g.merge(Sh[["show_id", "game_rate", "horse_rate", "strat_rate"]], on="show_id", how="left")
ref = D.merge(Sh, on="show_id")
# Standardise each rate on the 194-show corpus (population SD) so coefficients are per SD of the show-level measure.
for c in ("game_rate", "horse_rate", "strat_rate"):
    g[c.replace("_rate", "_z")] = (g[c] - ref[c].mean()) / ref[c].std(ddof=0)
# Drop NBC-branded shows (ambiguous title match).
nb = D[D.show.str.contains("NBC")].show_id
g = g[~g.show_id.isin(nb)]


# Fit y on controls + ideology (+ side label if the frame carries it) + the chosen exposure columns; report coefficient j with CR1, CR2 and wild-bootstrap inference.
def run(y, xcols, j):
    X = design(g, IDEO + ("side",)) if "side" in g else design(g, IDEO)  # side label control only if the frame has it
    for c in xcols:
        X[c] = g[c]
    m = pd.concat([X, g[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    return cr_all(m[y].values, m[X.columns].values, m.show_id.values, list(X.columns).index(j))


print(f"{'model':<52}{'b':>8}{'CR1 p':>8}{'CR2 p':>8}{'Satt df':>9}{'WILD p':>8}{'N':>5}")
# Six models per outcome: each exposure alone, both together (reporting each coefficient in turn), and address with the strategy subframe only.
for y, lab in (("inst_cyn", "CYNICISM"), ("excl", "EXCLUSIVITY")):
    for xc, j, desc in (
        (["dir_z"], "dir_z", "directive alone"),
        (["game_z"], "game_z", "game frame alone"),
        (["dir_z", "game_z"], "dir_z", "both -> directive"),
        (["dir_z", "game_z"], "game_z", "both -> game frame"),
        (["dir_z", "strat_z"], "dir_z", "directive + strategy subframe -> directive"),
        (["dir_z", "strat_z"], "strat_z", "directive + strategy subframe -> strategy"),
    ):
        r = run(y, xc, j)
        print(
            f"{lab+': '+desc:<52}{r['b']:+8.3f}{r['p1']:8.4f}{r['p2']:8.4f}{r['df']:9.1f}{r['pw']:8.4f}{r['N']:5d}"
        )
    print()
# Descriptive: which shows sit at the extremes of the game-frame distribution, and how it correlates with ideology, address and solo format.
print("show-level: which shows are high game-frame? (top/bottom 5 of the 194, with lean)")
t = ref.sort_values("game_rate")
print("  lowest :", ", ".join(f"{s[:22]}({l})" for s, l in zip(t.show.head(5), t.lean.head(5))))
print("  highest:", ", ".join(f"{s[:22]}({l})" for s, l in zip(t.show.tail(5), t.lean.tail(5))))
print(
    f"  corr(game_rate, avg_host_cfscore)={ref.game_rate.corr(ref.avg_host_cfscore):+.3f}   corr(game_rate, dir_z)={ref.game_rate.corr(ref.dir_z):+.3f}   corr(game_rate, solo)={ref.game_rate.corr((ref.solo=='solo').astype(float)) if 'solo' in ref else float('nan'):+.3f}"
)
