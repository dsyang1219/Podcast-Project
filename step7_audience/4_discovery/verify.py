"""Discovery-sample verification table: each outcome on address with clustered, wild-bootstrap and show-permutation p-values, strict and expanded matches. Full record F.3."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
from common import *

# Outcomes to verify: the election-distrust items and composite, the four institutional-cynicism items and composite, and one civic-efficacy item. Labels are for the printout only.
OUT = [
    ("Q33E", "elections administered well (rev: distrust)"),  # entered raw (higher = administered well); read a negative b as more distrust
    ("Q32C", "assume election fraud if surprised"),
    ("elec_distrust", "ELECTION DISTRUST composite"),
    ("Q34B", "know how to reach officials"),
    ("Q35E", "laws uphold freedom/justice"),
    ("Q35C", "govt includes people like me"),
    ("Q33H", "freedom of press working"),
    ("inst_cyn", "INSTITUTIONAL CYNICISM composite"),
    ("Q12D", "unsure how to get involved"),
]
IDEO = ("share_right", "rw_plat")  # ideology controls: share of named shows leaning right + right-wing platform count
# Column header for the verification table.
print(
    f"{'outcome':<38}{'b':>7}{'clus p':>8}{'WILD p':>8} | {'+ideo b':>8}{'+ideo WILD':>11} | {'show b':>7}{'perm p':>8}{'k':>4} | {'EXPANDED b':>11}{'wild p':>8}"
)
# For each outcome: strict-match model with and without ideology controls, the show-level permutation, and the expanded-match model.
for v, lab in OUT:
    for g in (st, ex):
        if v.startswith("Q"):
            g[v] = num(g, v)  # raw Q items -> numbers, negative codes -> NaN (composites already numeric)
    b, p, n = clus(st, v)  # base model: slope, CR1 clustered p, n
    wp = wild(st, v)  # wild cluster bootstrap p (1500 draws)
    bi, pi, _ = clus(st, v, IDEO)  # + share_right and rw_plat
    wpi = wild(st, v, IDEO)
    sb, sp, k = showperm(st, v, IDEO)  # show-level permutation: k = shows with >= 2 listeners
    be, pe, _ = clus(ex, v, IDEO)  # expanded (host-name) match, with ideology controls
    wpe = wild(ex, v, IDEO)
    print(
        f"{lab:<38}{b:+7.3f}{p:8.4f}{wp:8.4f} | {bi:+8.3f}{wpi:11.4f} | {sb:+7.3f}{sp:8.4f}{k:4d} | {be:+11.3f}{wpe:8.4f}"
    )
print(
    "\n(+ideo = share_right + right-wing platform index added; show-level and EXPANDED columns include +ideo)"
)
# How collinear is exposure with the ideology controls? Large values would leave the +ideo model little independent variation.
print(
    f"corr(dir_z, share_right)={st.dir_z.corr(st.share_right):+.3f}  corr(dir_z, rw_plat)={st.dir_z.corr(st.rw_plat):+.3f}"
)
