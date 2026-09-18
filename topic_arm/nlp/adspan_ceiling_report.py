"""Assemble the Arm 1/2/3 nested-CV results into the reportable tables.

Reads only the JSON written by nlp/adspan_ceiling_nested.py and
nlp/adspan_ceiling_supervised.py -- it does no fitting, so it cannot introduce
selection. Every number it prints is an OUTER-loop quantity.

    .venv/bin/python -m nlp.adspan_ceiling_report
"""
from __future__ import annotations

import json

from .adspan_phase_b import OUT_DIR

TARGETS = ["extended_all", "host_only"]


def _load(name):
    p = OUT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def _ci(m):
    return f"{m['outer_r2']:+.3f} [{m['ci_95'][0]:+.3f}, {m['ci_95'][1]:+.3f}]"


def arm1_table(d):
    if not d:
        return
    dense = d.get("grids") == "dense"
    robust = set(d.get("robustness_rows", []))
    head = "dense grids" if dense else "as-published grids"
    print("\n## Arm 1 -- estimators on CLR topics (K=%d), outer-loop LOSO R2 (%s)\n"
          % (d["k"], head))
    cols = TARGETS + ([f"{t} (coarse grid)" for t in TARGETS] if dense else [])
    print("| estimator | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    names = list(next(iter(d["results"].values()))["models"])
    for n in names:
        cells = []
        for t in TARGETS:
            r = d["results"].get(t, {}).get("models", {}).get(n)
            cells.append(_ci(r) if r else "-")
        if dense:
            for t in TARGETS:
                prev = d["results"].get(t, {}).get("coarse_r2", {}).get(n)
                dr = d["results"].get(t, {}).get("delta_vs_coarse", {}).get(n)
                cells.append(f"{prev:+.3f} ({dr:+.3f})" if prev is not None else "-")
        label = n + (" *" if n in robust else "")
        print(f"| {label} | " + " | ".join(cells) + " |")
    for t in TARGETS:
        if t in d["results"]:
            r = d["results"][t]
            print(f"\n{t}: range [{r['range'][0]:+.3f}, {r['range'][1]:+.3f}], "
                  f"median {r['median']:+.3f}, ridge baseline {r['ridge_baseline']:+.3f}")
    if dense:
        _arm1_dense_notes(d, robust)


def _arm1_dense_notes(d, robust):
    """State plainly what densifying changed and what it does NOT license."""
    print("\nGrids are log-spaced at <=1/4 decade (1/2 for the MLP) and wide enough")
    print("that the modal selection is interior; structural RF/GBM grids are widened")
    print("rather than densified. This replaces the as-published Arm 1 grids, on which")
    print("elasticnet's optimum fell in a GAP between adjacent points and ridge selected")
    print("the grid floor in every fold. Selected hyperparameters per fold are recorded")
    print("in ceiling_nested_arm1_k75_dense.json.")
    if robust:
        print("\n* robustness row, excluded from the range/median so each estimator")
        print("  contributes once. Elastic net's L1 and L2 terms cannot co-scale, so")
        print("  its scaler is a real modelling choice, unlike ridge's; both are shown")
        print("  and neither is selected on its outer R2.")
    for t in TARGETS:
        r = d["results"].get(t)
        if not r:
            continue
        flagged = [(n, m["flags"]) for n, m in r["models"].items() if m.get("flags")]
        if flagged:
            print(f"\n{t} grid flags:")
            for n, fl in flagged:
                print(f"  {n}: {'; '.join(fl)}")
    print("\nThe ridge baseline is still what carries into Arm 2. A higher elasticnet")
    print("point estimate does NOT promote it: choosing the estimator by the same outer")
    print("R2 used to report it is the outcome-selection this harness exists to remove.")


def arm2_table(d):
    if not d:
        return
    print("\n## Arm 2 -- representations at fixed ridge (K=%d), outer-loop LOSO R2\n" % d["k"])
    if not d.get("embeddings_are_cleaned_corpus", True):
        print("!! " + d.get("PLACEHOLDER_WARNING", "") + "\n")
    for t in TARGETS:
        if t not in d["results"]:
            continue
        r = d["results"][t]
        print(f"\n### {t} (n={r['models']['topics_only']['n']})\n")
        print("| representation | outer R2 [95% CI] | dR2 vs topics | Wilcoxon p (Holm) | beats topics-only? |")
        print("|---|---|---|---|---|")
        for name, m in r["models"].items():
            if name == "topics_only":
                print(f"| {name} | {_ci(m)} | baseline | - | - |")
                continue
            c = r["paired_vs_topics_only"][name]
            verdict = "YES" if c["beats_topics_only"] else "no"
            print(f"| {name} | {_ci(m)} | {c['delta_r2']:+.3f} | "
                  f"{c['wilcoxon_p']:.3f} ({c['wilcoxon_p_holm']:.3f}) | {verdict} |")


def arm3_table(ws, tf):
    if not (ws or tf):
        return
    print("\n## Arm 3 -- supervised RECOVERABILITY ceiling (a different question)\n")
    print("These learn features toward the target. NOT comparable head-to-head with")
    print("Arms 1-2 as 'better', and NOT evidence about dominance.\n")
    print("| supervised model | " + " | ".join(TARGETS) + " |")
    print("|---|" + "---|" * len(TARGETS))
    for label, d, key in [("wordscores (LOSO)", ws, "model"),
                          ("transformer fine-tune (grouped 5-fold)", tf, None)]:
        if not d:
            continue
        cells = []
        for t in TARGETS:
            r = d["results"].get(t)
            if not r:
                cells.append("-")
            else:
                m = r[key] if key else r
                cells.append(_ci(m))
        print(f"| {label} | " + " | ".join(cells) + " |")

    spread = []
    for t in TARGETS:
        r = (tf or {}).get("results", {}).get(t) or {}
        if r.get("per_seed_r2"):
            vals = ", ".join(f"{v:+.3f}" for v in r["per_seed_r2"])
            spread.append(f"- {t}: seeds {r.get('seeds')} -> {vals} "
                          f"(sd {r.get('seed_sd_r2', 0.0):.3f})")
    if spread:
        print("\nThe transformer row is the MEAN over seeds -- expected performance")
        print("of a single run -- and its interval is a show-level bootstrap of that")
        print("mean. Per-seed spread, which a single-seed run would hide:\n")
        for line in spread:
            print(line)


def main():
    # Prefer the dense-grid Arm 1 when it exists: the as-published grids
    # under-tuned elasticnet by ~0.12 R2 and searched ridge on the wrong range,
    # so that table compares tuning effort as much as estimators.
    a1 = _load("ceiling_nested_arm1_k75_dense.json") or _load("ceiling_nested_arm1_k75.json")
    a2 = _load("ceiling_nested_arm2_k75.json")
    ws = _load("ceiling_nested_arm3_wordscores.json")
    tf = _load("ceiling_nested_arm3_transformer.json")
    arm2_table(a2)      # claim-relevant arm first, per the brief
    arm1_table(a1)
    arm3_table(ws, tf)
    print("\n---\nRecoverability != dominance. No R2 above changes the PCA result that")
    print("ideology is a low-variance, secondary axis of this discourse.")


if __name__ == "__main__":
    main()
