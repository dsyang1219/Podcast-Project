"""Is the topic->ideology signal an artifact of how the ideology target was built?

The headline target (`ideology_primary`, N=204) is NOT uniformly host-derived.
Read off data/output/ideology_targets_204.csv:

    host_full  84   host coverage complete -> host campaign-finance score
    guest     120   coverage none/partial/no-host-listed -> guest-inferred score

So 120 of 204 shows (59%) are scored from WHO THE SHOW BOOKS. That creates a
booking circularity distinct from the ad confound: the LDA features describe
what a show talks about, and booking a guest generates talk about that guest's
issues. The target and the features can share a cause.

This runs the SAME Ridge/LOSO-CV/permutation harness against three targets so
the size of that circularity is measured instead of argued about:

  host_only     N=84   ideology_primary on host_full shows ONLY.
                       Guest-independent -- no booking circularity. The
                       conservative, defensible number.
  extended_all  N=204  ideology_primary as-is. The headline, and the target the
                       0.451 baseline used, so the before/after chain is
                       like-for-like.
  guest_only    N=202  ideology_guest_only for every show that has it. Maximally
                       exposed to booking circularity -- an UPPER BOUND, not an
                       estimate.

Note host_only and extended_all are not independent: the 84 host shows are a
subset of the 204. host_only is a smaller, cleaner sample, so a lower R2 there
is expected from N alone and is NOT by itself evidence of circularity -- the
permutation null and CI width are what make it interpretable.

    .venv/bin/python -m nlp.adspan_target_robustness --corpus clean
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config as pipeline_config
from .adspan_phase_c import K_GRID, paths_for
from .embed_ideology_nonlinear import (
    bootstrap_r2_ci,
    loso_cv,
    r2_from_predictions,
    ridge_fit_predict,
)
from .lda_ideology_clr import clr

TARGETS_PATH = pipeline_config.OUTPUT_DIR / "ideology_targets_204.csv"


def build_targets() -> dict[str, pd.Series]:
    t = pd.read_csv(TARGETS_PATH, dtype={"show_id": str}).set_index("show_id")
    host = t[t["ideology_primary_source"] == "host_full"]["ideology_primary"].dropna()
    return {
        "host_only": host,
        "extended_all": t["ideology_primary"].dropna(),
        "guest_only": t["ideology_guest_only"].dropna(),
    }


def probe(doc_topic: pd.DataFrame, cols: list[str], y_series: pd.Series) -> dict:
    """Same harness as nlp/lda_regime_refit.probe_ideology -- show-level mean of
    chunk topic proportions, CLR, Ridge with alpha tuned inside each LOSO fold,
    plus a permutation null. Only the target vector differs across calls."""
    show_topics = doc_topic.groupby("collection_id")[cols].mean()
    joined = show_topics.join(y_series.rename("y"), how="inner").dropna(subset=["y"])
    X_raw = joined[cols].to_numpy(dtype=float)
    X = clr(np.where(X_raw <= 0, 1e-12, X_raw))
    y = joined["y"].to_numpy(dtype=float)

    preds = loso_cv(X, y, ridge_fit_predict)
    r2 = r2_from_predictions(y, preds)
    lo, hi = bootstrap_r2_ci(y, preds)
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    perm = r2_from_predictions(y_shuf, loso_cv(X, y_shuf, ridge_fit_predict))
    return {"n": int(len(y)), "r2": float(r2), "ci_95": [float(lo), float(hi)],
            "permutation_r2": float(perm), "y_sd": float(np.std(y, ddof=1))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="clean", choices=["clean", "preview"])
    args = ap.parse_args()

    P = paths_for(args.corpus)
    targets = build_targets()
    print(f"[targets] CORPUS: {P['label']}")
    for name, s in targets.items():
        print(f"  {name:<14} N={len(s):<5} mean={s.mean():+.3f} sd={s.std():.3f}")

    rows = []
    for k in K_GRID:
        path = Path(P["doctopic_fmt"].format(k=k))
        if not path.exists():
            print(f"[skip] K={k}: no doc-topic matrix yet")
            continue
        dt = pd.read_csv(path, dtype={"collection_id": str, "chunk_id": str})
        cols = [f"T{i}" for i in range(k)]
        for name, s in targets.items():
            r = probe(dt, cols, s)
            r.update({"k": k, "target": name})
            rows.append(r)
            print(f"  K={k:<4} {name:<14} N={r['n']:<5} R2={r['r2']:+.4f} "
                  f"CI=[{r['ci_95'][0]:+.3f},{r['ci_95'][1]:+.3f}] "
                  f"perm={r['permutation_r2']:+.4f}", flush=True)

    if not rows:
        raise SystemExit("no doc-topic matrices found -- run the K sweep first")

    df = pd.DataFrame(rows)
    suffix = "" if args.corpus == "clean" else f"_{args.corpus}"
    out_csv = P["sweep"].parent / f"target_robustness{suffix}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{'K':<7}" + "".join(f"{t:<26}" for t in targets))
    for k in sorted(df["k"].unique()):
        line = f"{k:<7}"
        for t in targets:
            sub = df[(df["k"] == k) & (df["target"] == t)]
            line += (f"{sub.iloc[0]['r2']:.3f} (N={sub.iloc[0]['n']})".ljust(26)
                     if len(sub) else "-".ljust(26))
        print(line)

    (P["sweep"].parent / f"target_robustness{suffix}.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus, "corpus_label": P["label"],
        "is_preview": P["is_preview"],
        "target_composition": {
            "host_only": "N=84, ideology_primary where source==host_full; "
                          "guest-independent, no booking circularity",
            "extended_all": "N=204, ideology_primary as-is; 84 host + 120 "
                             "guest-inferred; the headline and the 0.451 baseline target",
            "guest_only": "N=202, ideology_guest_only; UPPER BOUND, maximally "
                           "exposed to booking circularity",
        },
        "results": rows,
    }, indent=2, default=str))
    print(f"\n[targets] -> {out_csv}")


if __name__ == "__main__":
    main()
