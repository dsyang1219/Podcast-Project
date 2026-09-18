"""Saturation check, best-effort given the data gap.

The nested 5/10/15/20/25/30/40h saturation_ladder design requires each rung's
full episode set to be transcribed per show. As of this pilot only 1 of 58
shows (KQED's Forum) has cleared even the 5h rung, and most shows have a
single episode (median ~0.58h transcribed). The literal per-show ladder check
CANNOT be run yet -- this script instead builds a proxy: for every show with
>=3 transcribed episodes, order its own episodes by H25 draw_order and plot
d_eff vs cumulative hours actually available, so far. This is directional
only, not the final design, and is dominated by n=1 (KQED's Forum) which is
the only show with enough episodes to show a real curve shape.
"""
import sys

sys.path.insert(0, ".")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlp.pilot.common import (
    build_dictionary_and_dtm, correspondence_analysis_d_eff, fit_lda,
    lda_doc_topic, load_tokenized_docs,
)

K = 20
SEED = 0
MIN_EPISODES = 3
MIN_CHUNKS_FOR_CA = 4

df, texts = load_tokenized_docs()
dictionary, bow_corpus, dtm, vocab = build_dictionary_and_dtm(texts)
lda = fit_lda(dtm, K, SEED)
theta = lda_doc_topic(lda, dtm)

manifest = pd.read_csv("data/output/sample_out/sample_manifest_H25.csv")
dur_by_ep = dict(zip(manifest["episode_id"], manifest["duration_hr"]))
draw_by_ep = dict(zip(manifest["episode_id"], manifest["draw_order"]))

eps_per_show = df.groupby("show")["episode_id"].nunique()
shows = eps_per_show[eps_per_show >= MIN_EPISODES].index.tolist()
print(f"Shows with >= {MIN_EPISODES} transcribed episodes: {shows}")

curves = {}
for show in shows:
    show_df = df[df["show"] == show]
    ep_ids = show_df["episode_id"].unique().tolist()
    ep_ids = [e for e in ep_ids if e in draw_by_ep]
    ep_ids.sort(key=lambda e: draw_by_ep[e])

    points = []
    cum_hours = 0.0
    seen_eps = []
    for ep in ep_ids:
        cum_hours += dur_by_ep.get(ep, 0.0)
        seen_eps.append(ep)
        idx = show_df.index[show_df["episode_id"].isin(seen_eps)].to_numpy()
        if len(idx) >= MIN_CHUNKS_FOR_CA:
            d_eff = correspondence_analysis_d_eff(theta[idx])
            points.append((cum_hours, d_eff, len(idx)))
    if points:
        curves[show] = points
        print(f"  {show}: {len(points)} checkpoints, max hours={points[-1][0]:.2f}")

fig, ax = plt.subplots(figsize=(9, 6))
for show, points in curves.items():
    hours = [p[0] for p in points]
    d_effs = [p[1] for p in points]
    ax.plot(hours, d_effs, marker="o", label=show)
ax.set_xlabel("cumulative hours transcribed (this show, actual data so far)")
ax.set_ylabel("d_eff (participation ratio, CA on chunk x topic)")
ax.set_title("Saturation proxy (NOT the full ladder -- see caveat)\nLDA K=20, seed=0")
ax.legend(fontsize=7, loc="best")
ax.axvline(5, color="gray", linestyle="--", alpha=0.4)
ax.axvline(25, color="gray", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig("data/output/saturation_curve.png", dpi=150)
print("\nSaved data/output/saturation_curve.png")

rows = []
for show, points in curves.items():
    for hours, d_eff, n_chunks in points:
        rows.append({"show": show, "cum_hours": hours, "d_eff": d_eff, "n_chunks": n_chunks})
pd.DataFrame(rows).to_csv("data/output/pilot_saturation_proxy.csv", index=False)
print("Saved data/output/pilot_saturation_proxy.csv")

kqed = curves.get("KQED's Forum")
if kqed:
    print("\nKQED's Forum (only show with a real multi-point curve):")
    for h, d, n in kqed:
        print(f"  {h:.2f}h ({n} chunks): d_eff={d:.2f}")
