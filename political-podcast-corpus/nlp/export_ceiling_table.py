"""Flatten the Arm 1/2/3 nested-CV JSONs into one tidy CSV for plotting.

The JSONs carry per-show predictions and squared errors, which is what the
paired tests need but not what a bar chart needs. This emits one row per
(arm, target, representation/estimator) with the headline interval.

    .venv/bin/python -m nlp.export_ceiling_table

Writes data/output/adspan/ceiling_summary.csv:
    arm, k, target, model, r2, ci_low, ci_high, n, beats_topics_only, note
"""
from __future__ import annotations

import json

import pandas as pd

from .adspan_phase_c import OUT_DIR

K = 75
FILES = {
    1: "ceiling_nested_arm1_k75.json",
    2: "ceiling_nested_arm2_k75.json",
    "3_wordscores": "ceiling_nested_arm3_wordscores.json",
    "3_transformer": "ceiling_nested_arm3_transformer.json",
}
NOTE = {
    1: "estimator comparison at fixed CLR-topic features",
    2: "representation comparison at fixed ridge -- the claim-relevant arm",
    "3_wordscores": "SUPERVISED recoverability; not comparable to arms 1-2",
    "3_transformer": "SUPERVISED recoverability; not comparable to arms 1-2",
}


def _row(arm, target, model, m, extra=None):
    ci = m.get("ci_95") or [None, None]
    r = {"arm": arm, "k": K, "target": target, "model": model,
         "r2": m.get("outer_r2"), "ci_low": ci[0], "ci_high": ci[1],
         "n": m.get("n"), "beats_topics_only": "", "note": NOTE[arm]}
    if extra:
        r.update(extra)
    return r


def main() -> None:
    rows = []
    for arm, fname in FILES.items():
        p = OUT_DIR / fname
        if not p.exists():
            print(f"[skip] {fname} not present yet")
            continue
        d = json.loads(p.read_text())
        for target, block in d.get("results", {}).items():
            models = block.get("models")
            if models:                                   # arms 1 and 2
                cmp_ = (block.get("paired_vs_topics_only")
                        or block.get("vs_ridge_paired") or {})
                for name, m in models.items():
                    c = cmp_.get(name, {})
                    rows.append(_row(arm, target, name, m, {
                        "beats_topics_only": c.get("beats_topics_only", ""),
                        "delta_r2": c.get("delta_r2", ""),
                        "wilcoxon_p_holm": c.get("wilcoxon_p_holm", "")}))
            elif "model" in block:                       # wordscores
                rows.append(_row(arm, target, "wordscores", block["model"]))
            else:                                        # transformer
                rows.append(_row(arm, target, "transformer_finetune", block, {
                    "note": NOTE[arm] + (
                        f"; mean of seeds {block.get('seeds')}"
                        if block.get("seeds") else "")}))

    df = pd.DataFrame(rows)
    out = OUT_DIR / "ceiling_summary.csv"
    df.to_csv(out, index=False)
    print(f"[done] {len(df)} rows -> {out}")
    print(df.to_string(index=False, max_colwidth=44))


if __name__ == "__main__":
    main()
