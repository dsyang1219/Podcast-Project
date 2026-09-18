"""Published-tool address measure from MFTE tags.

Reads the raw tag counts MFTE wrote for every episode (run_mfte_corpus.sh), keeps two tags from Biber's feature
set as implemented by MFTE (Le Foll 2021; Python port by Shakir):

    VIMP   imperative verbs
    PP2    second-person pronouns

turns them into per-10,000-word episode rates, aggregates to shows weighted by episode word count (the same
aggregation as the paper's dir_z), standardises each rate against the 194-show corpus reference set, and averages
the two z-scores into `mfte_z`. No word list of ours is involved.

Writes
    step7_audience/inputs/mfte_episode_rates.csv   one row per episode: show_id, episode_id, words, vimp, pp2
    step7_audience/inputs/mfte_show_scores.csv     one row per show: rates, z-scores, mfte_z, episodes, words
and prints the validation of mfte_z against the paper's dir_z, the host DIME score and the LLM side label.

    .venv/bin/python step6_register/mfte_address.py
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MFTE_DIR = ROOT / "data/output/mfte_corpus"
INPUTS = ROOT / "step7_audience/inputs"
REF = ROOT / "handoff/prereg/dirz_reference_scale.csv"

# ---------------------------------------------------------------- episode rates
frames = []
for f in sorted(glob.glob(str(MFTE_DIR / "shard*_MFTE/Statistics/counts_raw.csv"))):
    d = pd.read_csv(f, usecols=["Filename", "Words", "VIMP", "PP2"])
    frames.append(d)
if not frames:
    sys.exit("no MFTE statistics found; run step6_register/run_mfte_corpus.sh first")
P = pd.concat(frames, ignore_index=True)
# Filename is "<show_id>__<episode_id>__p<k>.txt" (stage_mfte_corpus.py): one row per piece. Recover the ids and
# sum the raw counts back to one row per episode; counts are additive so the pieces lose nothing.
parts = P.Filename.str.split("__", n=2)
P["show_id"] = parts.str[0]
P["episode_id"] = parts.str[1]
E = P.groupby(["show_id", "episode_id"], as_index=False)[["Words", "VIMP", "PP2"]].sum()
E = E[E.Words >= 500].copy()  # same minimum as the register batteries
E["vimp"] = 1e4 * E.VIMP / E.Words
E["pp2"] = 1e4 * E.PP2 / E.Words
E[["show_id", "episode_id", "Words", "vimp", "pp2"]].rename(columns={"Words": "words"}).to_csv(
    INPUTS / "mfte_episode_rates.csv", index=False
)
print(f"episodes tagged: {len(E):,} across {E.show_id.nunique()} shows, {E.Words.sum()/1e6:.0f}M words")

# ---------------------------------------------------------------- show rates (word-count weighted, as dir_z)
S = (
    E.groupby("show_id")
    .apply(
        lambda g: pd.Series(
            {
                "vimp": np.average(g.vimp, weights=g.Words),
                "pp2": np.average(g.pp2, weights=g.Words),
                "episodes": len(g),
                "words": g.Words.sum(),
            }
        ),
        include_groups=False,
    )
    .reset_index()
)

# ---------------------------------------------------------------- standardise against the 194-show reference
R = pd.read_csv(REF)
R["show_id"] = R.show_id.astype(str)
ref = S[S.show_id.isin(R.show_id)]
print(f"reference shows with MFTE scores: {len(ref)} of {len(R)}")
for c in ("vimp", "pp2"):
    mu, sd = ref[c].mean(), ref[c].std(ddof=0)  # population SD, as the dir_z reference scale was built
    S[c + "_z"] = (S[c] - mu) / sd
S["mfte_z"] = S[["vimp_z", "pp2_z"]].mean(axis=1)
S = S.sort_values("mfte_z", ascending=False)
S.to_csv(INPUTS / "mfte_show_scores.csv", index=False)

# ---------------------------------------------------------------- validation against the paper's measure and anchors
D = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "show", "dir_z", "avg_host_cfscore", "side", "lean"]]
D["show_id"] = D.show_id.astype(str)
V = S.merge(D, on="show_id")
print(f"\ncorpus shows with both measures: {len(V)}")


def line(name, x, y, d):
    ok = d[x].notna() & d[y].notna()
    r = stats.pearsonr(d.loc[ok, x], d.loc[ok, y])[0]
    rho = stats.spearmanr(d.loc[ok, x], d.loc[ok, y])[0]
    print(f"  {name:44s} r={r:+.3f}  rho={rho:+.3f}  (n={ok.sum()})")


line("mfte_z vs dir_z (paper composite)", "mfte_z", "dir_z", V)
line("  vimp rate vs paper syntactic imperatives", "vimp_z", "dir_z", V)
line("  pp2 rate vs paper second person", "pp2_z", "dir_z", V)
line("mfte_z vs host DIME score", "mfte_z", "avg_host_cfscore", V)
line("dir_z  vs host DIME score (reference)", "dir_z", "avg_host_cfscore", V)
line("mfte_z vs LLM side label", "mfte_z", "side", V)
sd_ = V.side.notna()
a, b = V.loc[sd_ & (V.side > 0), "mfte_z"], V.loc[sd_ & (V.side < 0), "mfte_z"]
dd = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
t, p = stats.ttest_ind(a, b, equal_var=False)
print(f"  right - left gap in mfte_z: d={dd:+.2f}  (Welch p={p:.1e}, n={len(a)}/{len(b)})")
print("\ntop 8 / bottom 8 shows on mfte_z:")
print(V.sort_values("mfte_z", ascending=False)[["show", "mfte_z", "dir_z", "lean"]].head(8).to_string(index=False))
print(V.sort_values("mfte_z")[["show", "mfte_z", "dir_z", "lean"]].head(8).to_string(index=False))
