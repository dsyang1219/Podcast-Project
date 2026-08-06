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
    print("\n## Arm 1 -- estimators on CLR topics (K=%d), outer-loop LOSO R2\n" % d["k"])
    print("| estimator | " + " | ".join(TARGETS) + " |")
    print("|---|" + "---|" * len(TARGETS))
    names = list(next(iter(d["results"].values()))["models"])
    for n in names:
        cells = []
        for t in TARGETS:
            r = d["results"].get(t, {}).get("models", {}).get(n)
            cells.append(_ci(r) if r else "-")
        print(f"| {n} | " + " | ".join(cells) + " |")
    for t in TARGETS:
        if t in d["results"]:
            r = d["results"][t]
            print(f"\n{t}: range [{r['range'][0]:+.3f}, {r['range'][1]:+.3f}], "
                  f"median {r['median']:+.3f}, ridge baseline {r['ridge_baseline']:+.3f}")


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


def main():
    a1 = _load("ceiling_nested_arm1_k75.json")
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
