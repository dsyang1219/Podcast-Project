"""Characterize the supervised Ridge ideology axis.

    python -m nlp.embed_ideology_axis [--target-words 500]

Follows the nonlinear probe's finding that a Ridge regression on the FULL
768-dim show-level embedding recovers CV R^2~=0.21 for DIME CFscore (vs ~0
from the earlier top-30-PCA analysis). This builds and describes that
1-dimensional supervised direction: which shows sit at its extremes, whether
it survives controlling for length/prolificness confounders, and how much of
the embedding's geometric variance it accounts for.

Two different fits are used deliberately, for two different, non-competing
purposes:
  - The FINAL model, fit on all N=120 matched shows, is used ONLY to describe
    the axis geometrically: which shows are extreme, what fraction of
    embedding variance the direction holds. This is a legitimate use of a
    full-data fit -- it's a descriptive summary, not a claimed predictive
    result.
  - The existing LOSO-CV out-of-sample predictions (from
    nlp.embed_ideology_nonlinear.ridge_fit_predict) are used for the
    confounder check. Using the in-sample fit there would be circular: it's
    fit ON the exact CFscore values being checked, so of course it would
    correlate before any residualization. The CV predictions are honest
    out-of-sample values, so residualizing THEM against confounders and
    checking whether correlation with true CFscore survives is a real test.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import config as pipeline_config
from .embed_ideology import load_cfscore
from .embed_ideology_nonlinear import build_show_level_embeddings, loso_cv, ridge_fit_predict


def load_confounders(target: int, matched_ids: list[str]) -> pd.DataFrame:
    OUT = pipeline_config.OUTPUT_DIR
    chunks_bert = pd.read_csv(OUT / f"chunks_{target}_bert.csv",
                               usecols=["chunk_id", "collection_id", "bert_text"])
    meta = json.loads((OUT / f"chunk_embeddings_{target}_weighted.meta.json").read_text())
    chunk_id_order = meta["chunk_id_order"]
    by_id = chunks_bert.set_index("chunk_id")
    show_ids = by_id.loc[chunk_id_order, "collection_id"].astype(str)
    n_raw_words = by_id.loc[chunk_id_order, "bert_text"].str.split().apply(len)

    df = pd.DataFrame({"show_id": show_ids.to_numpy(), "n_raw_words": n_raw_words.to_numpy()})
    agg = df.groupby("show_id").agg(mean_raw_words_per_chunk=("n_raw_words", "mean"),
                                     n_chunks=("n_raw_words", "size"))
    return agg.loc[matched_ids]


def run(target: int) -> None:
    OUT = pipeline_config.OUTPUT_DIR
    cfscore_df = load_cfscore()
    show_ids_all, X_all = build_show_level_embeddings(target)
    id_to_row = {sid: i for i, sid in enumerate(show_ids_all)}
    matched_ids = [sid for sid in cfscore_df.index if sid in id_to_row]
    n = len(matched_ids)
    X = np.stack([X_all[id_to_row[sid]] for sid in matched_ids])
    y = cfscore_df.loc[matched_ids, "avg_host_cfscore"].to_numpy()
    show_names = cfscore_df.loc[matched_ids, "show"].to_dict()
    print(f"[data] N={n} (same matched shows as linear + nonlinear analyses)")

    # --- geometric characterization: final model fit on ALL matched shows ---
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1000.0))])
    pipe.fit(X, y)
    scaled_X = pipe.named_steps["scale"].transform(X)
    direction = pipe.named_steps["ridge"].coef_
    unit_dir = direction / np.linalg.norm(direction)
    projection = scaled_X @ unit_dir  # descriptive axis: extremes + variance-held only

    axis_variance = float(np.var(projection))
    total_variance = float(np.sum(np.var(scaled_X, axis=0)))
    random_direction_baseline = total_variance / scaled_X.shape[1]
    print(f"[geometry] axis variance={axis_variance:.3f} vs total={total_variance:.1f} "
          f"({100*axis_variance/total_variance:.2f}% of embedding variance) | "
          f"random-direction baseline={random_direction_baseline:.3f} "
          f"({'ABOVE' if axis_variance > random_direction_baseline else 'at/below'} average)")

    order = np.argsort(projection)
    extremes = ([{"show_id": matched_ids[i], "show": show_names[matched_ids[i]],
                  "end": "negative_axis", "cfscore": float(y[i])} for i in order[:3]]
                + [{"show_id": matched_ids[i], "show": show_names[matched_ids[i]],
                    "end": "positive_axis", "cfscore": float(y[i])} for i in order[-3:]])
    print("[geometry] extremes:")
    for e in extremes:
        print(f"    {e['end']:>13}: {e['show']!r} (CFscore={e['cfscore']:+.3f})")

    # --- honest confounder check: LOSO-CV out-of-sample predictions, not the in-sample fit ---
    loso_preds = loso_cv(X, y, ridge_fit_predict)
    r_before, p_before = pearsonr(loso_preds, y)

    confounders_df = load_confounders(target, matched_ids)
    confounders = confounders_df.to_numpy()
    resid_model = LinearRegression().fit(confounders, loso_preds)
    loso_resid = loso_preds - resid_model.predict(confounders)
    r_after, p_after = pearsonr(loso_resid, y)
    print(f"[confounder check, on HONEST LOSO-CV predictions] "
          f"r_before={r_before:.3f} (p={p_before:.4f})  "
          f"r_after={r_after:.3f} (p={p_after:.4f})")

    report = {
        "target": target, "n_matched": n,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "axis_geometry": {
            "axis_variance": axis_variance, "total_variance": total_variance,
            "fraction_of_variance": axis_variance / total_variance,
            "random_direction_baseline_variance": random_direction_baseline,
            "extremes": extremes,
        },
        "confounder_check_on_loso_cv_predictions": {
            "r_before": float(r_before), "p_before": float(p_before),
            "r_after_controlling_length_and_prolificness": float(r_after), "p_after": float(p_after),
            "note": "Uses honest out-of-sample LOSO-CV predictions, not the in-sample-fit "
                    "axis -- residualizing the in-sample fit would be circular since it's "
                    "fit on the exact CFscore values being checked.",
        },
    }
    report_path = OUT / f"embed_ideology_axis_report_{target}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[report] -> {report_path.name}")

    make_figure(target, matched_ids, projection, loso_preds, y, show_names, extremes, OUT)


def make_figure(target, matched_ids, projection, loso_preds, y, show_names, extremes, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].scatter(projection, y, alpha=0.6)
    for j, e in enumerate(extremes):
        i = matched_ids.index(e["show_id"])
        dy = 14 if j % 2 == 0 else -18
        axes[0].annotate(e["show"], (projection[i], y[i]), fontsize=7, xytext=(0, dy),
                          textcoords="offset points", ha="center",
                          arrowprops=dict(arrowstyle="-", alpha=0.4, lw=0.6))
    axes[0].set_xlabel("Ridge axis (in-sample fit, descriptive only)")
    axes[0].set_ylabel("avg_host_cfscore")
    axes[0].set_title("Geometric axis: extremes (in-sample fit)")

    r_loso, _ = pearsonr(loso_preds, y)
    axes[1].scatter(loso_preds, y, alpha=0.6, color="tab:orange")
    axes[1].set_xlabel("Ridge LOSO-CV prediction (honest, out-of-sample)")
    axes[1].set_ylabel("avg_host_cfscore")
    axes[1].set_title(f"Honest CV predictions vs CFscore, r={r_loso:.3f}")

    fig.suptitle(f"Supervised Ridge ideology axis, N={len(y)}")
    fig.tight_layout()
    fig.savefig(out_dir / f"ideology_ridge_axis_{target}.png", dpi=120)
    plt.close(fig)
    print(f"[figure] -> ideology_ridge_axis_{target}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-words", type=int, default=500)
    args = ap.parse_args()
    run(args.target_words)


if __name__ == "__main__":
    main()
