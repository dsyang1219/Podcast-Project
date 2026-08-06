"""Export the ad-span-cleaned corpus for the STM arm (R).

Unit of analysis
----------------
PASSAGE level (37,942 docs), not show level. Two reasons: it matches the LDA arm
exactly, so any STM/LDA comparison is about the model rather than the unit; and
204 show-level documents is far too few to fit K=75 topics.

The cost is that passages inherit their SHOW's ideology, so passages are nested
within shows and are not independent draws. estimateEffect treats them as
independent, which makes its confidence intervals ANTI-CONSERVATIVE (too narrow)
-- the effective sample size for an ideology effect is closer to 204 shows than
to 37,942 passages. Every CI reported from this arm has to be read with that
caveat, and the show-clustered check in the R script exists for the same reason.

Text
----
`clean_text` is the token stream the LDA arm consumed: moderate regime,
ad-span-excised, lowercased, stopworded, docfreq-pruned to 38,008 terms. The R
side must therefore run textProcessor with EVERY cleaning step disabled, so it
only splits on whitespace and tallies. Re-stemming or re-stopwording there would
silently give STM a different vocabulary than LDA saw and break the comparison.

Covariates
----------
`ideology`   continuous DIME score (ideology_primary) -- the PREVALENCE covariate.
`ideo_group` tercile factor (Liberal / Moderate / Conservative) -- the CONTENT
             covariate, because stm's `content` argument takes a single
             categorical variable and cannot accept a continuous one.

    .venv/bin/python -m nlp.adspan_stm_export
"""
from __future__ import annotations

import json

import pandas as pd

from .adspan_phase_c import OUT_DIR, paths_for
from .adspan_target_robustness import TARGETS_PATH


def main() -> None:
    P = paths_for("clean")
    df = pd.read_csv(P["tokens"], dtype={"collection_id": str})
    df["clean_text"] = df["clean_text"].fillna("")

    t = pd.read_csv(TARGETS_PATH, dtype={"show_id": str}).set_index("show_id")
    ideo = t["ideology_primary"].dropna()
    host = t[t["ideology_primary_source"] == "host_full"].index

    df = df[df["collection_id"].isin(ideo.index)].copy()
    df["ideology"] = df["collection_id"].map(ideo).astype(float)
    df["is_host_target"] = df["collection_id"].isin(host).astype(int)

    # Terciles computed over SHOWS, not passages -- a passage-level quantile
    # would weight shows by how many passages they happen to have.
    show_ideo = ideo.loc[sorted(set(df["collection_id"]))]
    q = show_ideo.quantile([1 / 3, 2 / 3]).to_list()
    def grp(v: float) -> str:
        return "Liberal" if v <= q[0] else ("Conservative" if v > q[1] else "Moderate")
    df["ideo_group"] = df["ideology"].map(grp)

    out = OUT_DIR / "stm_input.csv"
    df[["chunk_id", "collection_id", "ideology", "ideo_group",
        "is_host_target", "clean_text"]].to_csv(out, index=False)

    meta = {
        "n_passages": int(len(df)),
        "n_shows": int(df["collection_id"].nunique()),
        "tercile_cuts": [float(x) for x in q],
        "group_counts_shows": {
            g: int(show_ideo.map(grp).value_counts().get(g, 0))
            for g in ("Liberal", "Moderate", "Conservative")},
        "group_counts_passages": df["ideo_group"].value_counts().to_dict(),
        "ideology_range": [float(show_ideo.min()), float(show_ideo.max())],
        "unit": "passage (nested within show -- CIs anti-conservative)",
        "text_field": "clean_text (already tokenized/pruned; disable all "
                      "textProcessor cleaning in R)",
    }
    (OUT_DIR / "stm_input_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
