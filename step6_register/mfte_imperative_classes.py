"""Split MFTE imperatives (VIMP) by Biber's (2006) semantic verb categories, as assigned by the tagger itself
(ACT activity, COMM communication, MENTAL, CAUSE causative/facilitation, OCCUR occurrence, EXIST existence/relationship,
ASPECT; DOAUX = imperative 'do'; NONE = verb not in Biber's category lists). Also counts the 'you know' bigram
(PP2 token immediately followed by know_VPRT/VB) since MFTE does not tag it as a discourse marker.
Reads data/output/mfte_corpus/shard*_MFTE/MFTE_Tagged/*.txt; writes
  step7_audience/inputs/mfte_imperative_episode_counts.csv (per episode) and
  step7_audience/inputs/mfte_imperative_show_scores.csv (per show, word-weighted rates per 10k words and z on the 194-show reference)."""
import glob, os, re, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; MFTE_DIR = ROOT / "data/output/mfte_corpus"; INPUTS = ROOT / "step7_audience/inputs"
CLASSES = ["ACT", "COMM", "MENTAL", "CAUSE", "OCCUR", "EXIST", "ASPECT", "DOAUX", "NONE"]
def scan(path):
    c = dict.fromkeys(CLASSES, 0); youknow = 0; prev_pp2 = False; lets = 0
    with open(path, errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts: prev_pp2 = False; continue
            tok = parts[0]; tag = tok.rsplit("_", 1)[-1] if "_" in tok else ""
            if tag == "VIMP":
                cls = next((p for p in parts[1:] if p in CLASSES), "NONE"); c[cls] += 1
                if tok.lower().startswith("let_"): lets += 1
            if prev_pp2 and tok.lower().startswith("know_"): youknow += 1
            prev_pp2 = (tag == "PP2")
    name = os.path.basename(path)[:-4]; sid, eid = name.split("__")[:2]
    return dict(show_id=sid, episode_id=eid, youknow=youknow, lets=lets, **{"VIMP_" + k: v for k, v in c.items()})
if __name__ == "__main__":
    files = sorted(glob.glob(str(MFTE_DIR / "shard*_MFTE/MFTE_Tagged/*.txt"))); print(f"{len(files):,} tagged pieces", flush=True)
    with Pool(16) as p: rows = p.map(scan, files, chunksize=200)
    P = pd.DataFrame(rows); E = P.groupby(["show_id", "episode_id"], as_index=False).sum()
    W = pd.read_csv(INPUTS / "mfte_episode_rates.csv", dtype={"show_id": str, "episode_id": str})[["show_id", "episode_id", "words"]]
    E = E.merge(W, on=["show_id", "episode_id"]); E.to_csv(INPUTS / "mfte_imperative_episode_counts.csv", index=False)
    cols = [c for c in E.columns if c.startswith("VIMP_")] + ["youknow", "lets"]
    print(f"episodes {len(E):,}; VIMP by class (share of all imperatives): " + ", ".join(f"{c[5:] if c.startswith('VIMP_') else c} {E[c].sum()/E[[x for x in cols if x.startswith('VIMP_')]].sum().sum():.1%}" for c in cols))
    S = E.groupby("show_id").apply(lambda g: pd.Series({**{c + "_rate": 1e4 * g[c].sum() / g.words.sum() for c in cols}, "episodes": len(g), "words": g.words.sum()}), include_groups=False).reset_index()
    R = pd.read_csv(ROOT / "data/output/dirz_reference_scale.csv"); R["show_id"] = R.show_id.astype(str); ref = S[S.show_id.isin(R.show_id)]
    for c in cols:
        mu, sd = ref[c + "_rate"].mean(), ref[c + "_rate"].std(ddof=0); S[c + "_z"] = (S[c + "_rate"] - mu) / sd
    S.to_csv(INPUTS / "mfte_imperative_show_scores.csv", index=False); print("written", INPUTS / "mfte_imperative_show_scores.csv", f"({len(S)} shows, {len(ref)} reference)")
