"""Export per-show PC scores so the ideology scatters can be redrawn natively.

The topic-PCA module reports correlations but never persists the scores
themselves, so any external redraw of the PC scatters had to trace pixels. This
writes them out.

Consistency is the whole point: the matrix, the CLR guard and the SVD all come
from nlp/adspan_topic_pca.py itself rather than being reimplemented, so the
exported scores are the same numbers behind topic_pca_ideology_by_pc.csv. The
script re-derives the published PC1/top-PC correlations from the exported
columns and refuses to write if they do not match to 1e-6.

Basis: fit once on all 204 shows (targets restrict rows, not the basis) --
matching step 3 of the source module, NOT the per-target refit of step 6.

    .venv/bin/python -m nlp.export_pc_scores

Writes data/output/adspan/pc_scores_k{30,75}.csv:
    show_id, show, PC1..PC9, ideology_primary, ideology_primary_source,
    ideology_guest_only
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .adspan_phase_c import OUT_DIR
from .adspan_topic_pca import (
    ZERO_GUARD,
    eigenvalues,
    pca_scores,
    show_topic_matrix,
)
from .adspan_target_robustness import build_targets
from .lda_ideology_clr import clr

N_PCS = 9
KS = (30, 75)
TARGETS_CSV = "ideology_targets_204.csv"


def _published_r(scores: np.ndarray, idx: pd.Index, y: pd.Series, pc: int) -> float:
    """Correlation of one PC with a target, over the shows the target covers."""
    common = [s for s in idx if s in y.index]
    pos = [idx.get_loc(s) for s in common]
    return float(np.corrcoef(scores[pos, pc - 1], y.loc[common].to_numpy(float))[0, 1])


def main() -> None:
    targets = build_targets()
    meta = pd.read_csv(OUT_DIR.parent / TARGETS_CSV, dtype={"show_id": str})
    names = dict(zip(meta.show_id, meta.show))

    for k in KS:
        show, diag = show_topic_matrix(k)
        X = clr(np.where(show.to_numpy(dtype=float) <= 0, ZERO_GUARD,
                         show.to_numpy(dtype=float)))
        n_comp = min(N_PCS, X.shape[1] - 1)
        scores, _, _ = pca_scores(X, n_comp)
        # Denominator must be the FULL spectrum, not the retained head, or the
        # shares read far too high (PC5 at K=75: 5.13% true vs 7.76% over top 9).
        full_eig = eigenvalues(X, X.shape[1] - 1)
        pct = 100 * full_eig / full_eig.sum()

        df = pd.DataFrame(
            {f"PC{i+1}": scores[:, i] for i in range(n_comp)},
            index=show.index)
        df.insert(0, "show", [names.get(s, "") for s in show.index])
        df.index.name = "show_id"

        for tname in ("extended_all", "host_only", "guest_only"):
            y = targets[tname]
            col = {"extended_all": "ideology_primary",
                   "host_only": "ideology_host_only",
                   "guest_only": "ideology_guest_only"}[tname]
            df[col] = [y.get(s, np.nan) for s in show.index]
        src = dict(zip(meta.show_id, meta.ideology_primary_source))
        df["ideology_primary_source"] = [src.get(s, "") for s in show.index]

        # --- verify against the published correlations before writing ---
        y_ext = targets["extended_all"]
        r_pc1 = _published_r(scores, show.index, y_ext, 1)
        top_pc = 5 if k == 75 else 2
        r_top = _published_r(scores, show.index, y_ext, top_pc)
        expect_pc1 = -0.03338668582568731 if k == 75 else -0.05383566539629478
        expect_top = -0.2807902537569164 if k == 75 else -0.1454325837064379
        for got, want, label in ((r_pc1, expect_pc1, "PC1"),
                                 (r_top, expect_top, f"PC{top_pc}")):
            if abs(abs(got) - abs(want)) > 1e-6:
                raise SystemExit(
                    f"K={k} {label}: exported scores give r={got:+.9f} but "
                    f"topic_pca_ideology_by_pc.csv reports {want:+.9f}. "
                    "Refusing to write inconsistent scores.")

        out = OUT_DIR / f"pc_scores_k{k}.csv"
        df.to_csv(out)
        print(f"[k={k}] {df.shape[0]} shows x {n_comp} PCs -> {out.name}")
        print(f"       verified PC1 r={r_pc1:+.6f}, PC{top_pc} r={r_top:+.6f} "
              f"(match published)")
        print("       % variance: " +
              ", ".join(f"PC{i+1}={pct[i]:.2f}" for i in range(min(9, n_comp))))


if __name__ == "__main__":
    main()
