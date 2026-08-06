"""Does the topic->ideology signal survive dropping REGISTER topics -- the ones
carrying speech-style rather than political content?

Distinct from the ad confound (removed at source in Phase B, worth +0.001) and
from the target confound (guest-inferred scores, worth ~+0.095). This asks a
third question: how much of the R2 is carried by HOW a show talks rather than
WHAT it talks about.

THE CONTROL THIS RUNS THAT THE ORIGINAL DROPOUT PROBE DID NOT
-------------------------------------------------------------
Dropping topics post hoc always lowers R2, because it removes degrees of
freedom along with content. The earlier ad analysis reported 0.451 -> 0.376
from dropping 18 topics and read that as the ads' contribution; Phase B then
showed the true ad contribution is +0.001. The gap was the un-controlled cost
of deleting 18 features.

So every drop condition here is paired with a RANDOM-DROP NULL: the same number
of topics, chosen at random, repeated N_RANDOM times. A drop only means
something if it falls outside that null. Reporting a raw delta without it
repeats the mistake that produced 0.376.

Topic selection is the user's, not the model's; the sets are recorded verbatim.

    .venv/bin/python -m nlp.adspan_register_dropout --k 50
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .adspan_phase_c import paths_for
from .adspan_target_robustness import build_targets, probe

N_RANDOM = 20

# User-selected. T47 (evidence/reporting language) and T49 (sports) were
# considered and explicitly KEPT as genuine content.
DROP_SETS = {
    "register_core": {
        "topics": [15, 32, 34, 39],
        "why": "pure speech register: profanity (15), food/lifestyle chatter (32), "
                "host pleasantries (34), discourse filler (39)",
    },
    "register_plus": {
        "topics": [8, 10, 15, 20, 28, 32, 34, 35, 39],
        "why": "register_core plus non-political content: entertainment (8), "
                "books/writing (10), local incidents+weather (20), news-bulletin "
                "format (28), personal narrative (35)",
    },
}
KEPT_DELIBERATELY = {47: "evidence/reporting language -- genuine content",
                     49: "sports -- genuine content"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--corpus", default="clean", choices=["clean", "preview"])
    args = ap.parse_args()

    P = paths_for(args.corpus)
    K = args.k
    dt = pd.read_csv(P["doctopic_fmt"].format(k=K),
                      dtype={"collection_id": str, "chunk_id": str})
    all_cols = [f"T{i}" for i in range(K)]
    targets = build_targets()

    print(f"[dropout] CORPUS: {P['label']}  K={K}")
    print(f"[dropout] deliberately KEPT: "
          f"{', '.join(f'T{t} ({w})' for t, w in KEPT_DELIBERATELY.items())}")

    rows = []
    base = {}
    for tname, ts in targets.items():
        r = probe(dt, all_cols, ts)
        base[tname] = r["r2"]
        rows.append({"condition": "full", "target": tname, "n_dropped": 0, **r})
        print(f"  full          {tname:<14} N={r['n']:<5} R2={r['r2']:+.4f} "
              f"CI=[{r['ci_95'][0]:+.3f},{r['ci_95'][1]:+.3f}]")

    for cname, spec in DROP_SETS.items():
        drop = [f"T{i}" for i in spec["topics"]]
        keep = [c for c in all_cols if c not in drop]
        print(f"\n  {cname}: dropping {len(drop)} topics {spec['topics']}")
        print(f"    ({spec['why']})")
        for tname, ts in targets.items():
            r = probe(dt, keep, ts)
            rows.append({"condition": cname, "target": tname,
                          "n_dropped": len(drop), **r})
            print(f"    {tname:<14} R2={r['r2']:+.4f} "
                  f"CI=[{r['ci_95'][0]:+.3f},{r['ci_95'][1]:+.3f}]  "
                  f"delta={r['r2'] - base[tname]:+.4f}")

    # ---- random-drop null: is the delta bigger than losing ANY n topics? ----
    print(f"\n[dropout] random-drop null ({N_RANDOM} draws per size, "
          f"target=extended_all)")
    null = {}
    ts = targets["extended_all"]
    for cname, spec in DROP_SETS.items():
        n = len(spec["topics"])
        rng = np.random.default_rng(0)
        vals = []
        for d in range(N_RANDOM):
            drop = set(rng.choice(K, size=n, replace=False).tolist())
            keep = [c for i, c in enumerate(all_cols) if i not in drop]
            vals.append(probe(dt, keep, ts)["r2"])
        vals = np.array(vals)
        obs = next(r["r2"] for r in rows
                    if r["condition"] == cname and r["target"] == "extended_all")
        pct = float((vals <= obs).mean())
        null[cname] = {"n_dropped": n, "random_mean": float(vals.mean()),
                        "random_p5": float(np.percentile(vals, 5)),
                        "random_p95": float(np.percentile(vals, 95)),
                        "observed": obs,
                        "pct_of_random_draws_at_or_below_observed": pct}
        print(f"  drop {n:<2} random: mean R2={vals.mean():+.4f} "
              f"[p5={np.percentile(vals, 5):+.4f}, p95={np.percentile(vals, 95):+.4f}]"
              f"   observed={obs:+.4f}")
        verdict = ("BELOW the random null -- these specific topics carry more "
                    "than n random ones"
                    if obs < np.percentile(vals, 5) else
                    "ABOVE the random null -- dropping these HURTS LESS than "
                    "dropping n random topics"
                    if obs > np.percentile(vals, 95) else
                    "INSIDE the random null -- indistinguishable from losing "
                    "n arbitrary topics")
        print(f"     -> {verdict}")
        null[cname]["verdict"] = verdict

    out = P["sweep"].parent / f"register_dropout_k{K}.json"
    out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus, "corpus_label": P["label"], "k": K,
        "drop_sets": DROP_SETS, "kept_deliberately": KEPT_DELIBERATELY,
        "n_random_draws": N_RANDOM,
        "results": rows, "random_null": null,
    }, indent=2, default=str))
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    print(f"\n[dropout] -> {out}")


if __name__ == "__main__":
    main()
