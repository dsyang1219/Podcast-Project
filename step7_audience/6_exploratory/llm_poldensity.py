"""Political-content density per corpus show from the ideology instrument's first stage (published tool).

Stage 1 of the Much et al. (2026) rubric asks whether a 750-character passage is political at all. A show's
density is the word-count-weighted share of its sampled passages (up to 40 per show-quarter, step 4) that the
instrument marks "Yes". No word list is involved. This replaces the hand-written political-term dictionary in
corpus_poldensity.py as the frame variable.

The political-podcast frame is density at or above the corpus 10th percentile (0.644 on the 194 reference shows).
Out-of-frame (chart-absent) shows were never scored by the instrument, so they have no value here; they are
outside the paper's corpus frame in any case.

Writes inputs/corpus_poldensity_llm.csv: show_id, llm_density, n_labelled, words_labelled.

    .venv/bin/python step7_audience/6_exploratory/llm_poldensity.py
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import numpy as np, pandas as pd

# Sampled passages (with their show and word count) joined to the instrument's labels.
C = pd.read_csv(DATA / "output/scoring_chunks_cap40.csv", usecols=["chunk_id", "collection_id", "n_words"], low_memory=False)
C["show_id"] = C.collection_id.astype(str)
L = pd.read_csv(DATA / "output/ideology_cap40.csv", usecols=["chunk_id", "politics"], low_memory=False)
D = C.merge(L, on="chunk_id")
D["pol"] = D.politics.astype(str).str.lower().str.startswith("y").astype(float)  # stage-1 "Yes" = political
S = (
    D.groupby("show_id")
    .apply(
        lambda g: pd.Series(
            {"llm_density": np.average(g.pol, weights=g.n_words), "n_labelled": len(g), "words_labelled": g.n_words.sum()}
        ),
        include_groups=False,
    )
    .reset_index()
)
S.to_csv(INPUTS / "corpus_poldensity_llm.csv", index=False)
# Report the frame cut on the 194 reference shows and which of them fall below it.
R = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "show"]]
R["show_id"] = R.show_id.astype(str)
M = R.merge(S, on="show_id")
cut = M.llm_density.quantile(0.10)
print(f"{len(S)} shows scored; reference shows {len(M)}; median density {M.llm_density.median():.3f}; 10th percentile (frame cut) {cut:.3f}")
print("below the cut:")
print(M[M.llm_density < cut].sort_values("llm_density")[["show", "llm_density", "n_labelled"]].to_string(index=False))
