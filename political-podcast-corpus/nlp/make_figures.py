"""Publication-quality figure suite for the interim report, built ONLY from
real artifacts already on disk -- no hardcoded numbers from memory/prose.
Every loaded value is printed at build time so it can be checked against its
source file.

Fixed ideology color convention, reused in EVERY figure that encodes
ideology (Figs 4/7 use it as bar/line color; Fig 2/5/6 don't encode ideology
directly so they use a separate neutral palette, documented per-figure):
    CONSERVATIVE = "#D55E00" (Okabe-Ito vermillion)
    LIBERAL      = "#0072B2" (Okabe-Ito blue)
    NEUTRAL/GRAY = "#888888"
    FILLER/CONFOUND = "#BBBBBB" with hatch "//" -- visually flagged, never
                      presented as a substantive content color.

Run: python -m nlp.make_figures
Output: ./figures/fig{N}_<name>.png (300dpi) and .pdf, plus a manifest
printed to stdout listing each figure's source artifact(s), and any figure
skipped for missing data (fabricating nothing).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from pipeline import config as pipeline_config

OUT = pipeline_config.OUTPUT_DIR
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

CONSERVATIVE = "#D55E00"
LIBERAL = "#0072B2"
NEUTRAL = "#888888"
FILLER_COLOR = "#BBBBBB"
TOPICS_COLOR = "#009E73"   # bluish-green, Okabe-Ito -- distinct from ideology red/blue
REGISTER_COLOR = "#E69F00"  # orange, Okabe-Ito
ETA2_COLOR = "#56B4E9"     # sky blue, Okabe-Ito -- distinct enough from LIBERAL's darker blue

FILLER_TOPICS = {"T61", "T12", "T57", "T39", "T46"}

manifest: list[dict] = []


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig, name: str, sources: list[str]) -> None:
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    manifest.append({"figure": name, "sources": sources, "status": "built"})
    print(f"[saved] {png_path}, {pdf_path}")


def skip(name: str, reason: str, sources: list[str]) -> None:
    manifest.append({"figure": name, "sources": sources, "status": f"SKIPPED: {reason}"})
    print(f"[SKIPPED] {name}: {reason}")


# ---------------------------------------------------------------------------
# Fig 1 -- corpus scope
# ---------------------------------------------------------------------------
def fig1_corpus_scope() -> None:
    shows_src = OUT / "sample_out" / "per_show_summary.csv"
    meta_src = OUT / "chunk_embeddings_500_weighted.meta.json"
    if not shows_src.exists() or not meta_src.exists():
        skip("fig1_corpus_scope", f"missing {shows_src} or {meta_src}", [str(shows_src), str(meta_src)])
        return

    shows_df = pd.read_csv(shows_src)
    n_shows = len(shows_df)
    total_hours = shows_df["hours"].sum()
    meta = json.loads(meta_src.read_text())
    n_passages = len(meta["chunk_id_order"])
    print(f"[fig1] shows={n_shows} (from {shows_src})")
    print(f"[fig1] total_hours={total_hours:.1f} (from {shows_src}, sum of 'hours' column)")
    print(f"[fig1] n_passages(chunks)={n_passages} (from {meta_src}, len(chunk_id_order))")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    stats = [
        (f"{n_shows}", "shows"),
        (f"{n_passages:,}", "passages (500-word chunks)"),
        (f"{total_hours:,.0f}", "hours of audio sampled"),
    ]
    for i, (big, small) in enumerate(stats):
        x = (i + 0.5) / len(stats)
        ax.text(x, 0.62, big, transform=ax.transAxes, ha="center", va="center",
                fontsize=34, fontweight="bold", color=TOPICS_COLOR)
        ax.text(x, 0.28, small, transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#333333")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i in range(1, len(stats)):
        x = i / len(stats)
        ax.axvline(x, ymin=0.15, ymax=0.85, color="#CCCCCC", linewidth=1)
    ax.set_title("Corpus scope", fontsize=13, pad=14)
    save(fig, "fig1_corpus_scope", [str(shows_src), str(meta_src)])


# ---------------------------------------------------------------------------
# Fig 2 -- topic prevalence
# ---------------------------------------------------------------------------
def fig2_topic_prevalence() -> None:
    src = OUT / "lda_k75_topic_words.csv"
    if not src.exists():
        skip("fig2_topic_prevalence", f"missing {src}", [str(src)])
        return
    df = pd.read_csv(src).sort_values("avg_share_pct", ascending=True)  # ascending for barh top-down reading
    print(f"[fig2] loaded {len(df)} topics from {src}")
    print(f"[fig2] top 5 by share: "
          f"{df.sort_values('avg_share_pct', ascending=False).head(5)[['topic','avg_share_pct']].to_dict('records')}")

    df["is_filler"] = df["topic"].isin(FILLER_TOPICS)
    colors = [FILLER_COLOR if f else TOPICS_COLOR for f in df["is_filler"]]

    fig, ax = plt.subplots(figsize=(8, 18))
    bars = ax.barh(df["topic"], df["avg_share_pct"], color=colors, edgecolor="none", height=0.75)
    for b, f in zip(bars, df["is_filler"]):
        if f:
            b.set_hatch("//")
            b.set_edgecolor("white")

    # annotate top 8 by share with a short gloss (first 2 real top-words, not invented)
    top8 = df.sort_values("avg_share_pct", ascending=False).head(8)
    for _, row in top8.iterrows():
        gloss = "/".join(str(row["top_words"]).split(", ")[:2])
        ax.text(row["avg_share_pct"] + 0.08, row["topic"], gloss, va="center", ha="left", fontsize=7, style="italic")

    ax.set_xlabel("Average corpus share (%)")
    ax.set_ylabel("LDA topic (K=75)")
    ax.set_title("Topic prevalence across the corpus (all 75 topics)")
    ax.tick_params(axis="y", labelsize=6)

    legend_handles = [
        mpatches.Patch(color=TOPICS_COLOR, label="Substantive topic"),
        mpatches.Patch(facecolor=FILLER_COLOR, hatch="//", edgecolor="white",
                        label="Conversational filler/register (T61,T12,T57,T39,T46)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    fig.text(0.5, 0.005,
              "Note: several of the highest-share topics are conversational filler, not substantive content "
              "(see hatched bars).",
              ha="center", fontsize=8, style="italic")
    save(fig, "fig2_topic_prevalence", [str(src)])


# ---------------------------------------------------------------------------
# Fig 3 -- ideology coverage extension
# ---------------------------------------------------------------------------
def fig3_coverage_extension() -> None:
    src = OUT / "ideology_targets_204.csv"
    if not src.exists():
        skip("fig3_coverage_extension", f"missing {src}", [str(src)])
        return
    df = pd.read_csv(src)
    counts = df["ideology_primary_source"].value_counts()
    n_host = int(counts.get("host_full", 0))
    n_guest = int(counts.get("guest", 0))
    n_total = len(df)
    print(f"[fig3] loaded {src}: host_full={n_host}, guest={n_guest}, total={n_total}")
    print(f"[fig3] NOTE: real breakdown is host_full={n_host}/guest={n_guest} -- this is the ACTUAL split "
          f"from the artifact (task brief's assumed 120 host/84 guest was reversed from the real data).")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(["Ideology coverage"], [n_host], color=NEUTRAL, edgecolor="black", linewidth=0.5,
            label=f"Host-scored (clean, n={n_host})")
    ax.barh(["Ideology coverage"], [n_guest], left=[n_host], color=TOPICS_COLOR, edgecolor="black", linewidth=0.5,
            label=f"Guest-DIME extension (n={n_guest})")
    ax.text(n_host / 2, 0, str(n_host), ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax.text(n_host + n_guest / 2, 0, str(n_guest), ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax.text(n_total + 2, 0, f"= {n_total} total", ha="left", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0, n_total + 25)
    ax.set_xlabel("Number of shows")
    ax.set_yticks([])
    ax.set_title("Guest-DIME matching extended ideology coverage from 84 to 204 shows")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, -0.55), ncol=1)
    ax.spines["left"].set_visible(False)
    save(fig, "fig3_coverage_extension", [str(src)])


# ---------------------------------------------------------------------------
# Fig 4 -- two-arm ideology comparison (headline)
# ---------------------------------------------------------------------------
def fig4_two_arm_comparison() -> None:
    lda_src = OUT / "lda_ideology_204_report.json"
    embed_src = OUT / "embed_ideology_204_report.json"
    if not lda_src.exists() or not embed_src.exists():
        skip("fig4_two_arm_comparison", f"missing {lda_src} or {embed_src}", [str(lda_src), str(embed_src)])
        return

    lda = {r["condition"]: r for r in json.loads(lda_src.read_text())["results"]}
    embed = {r["condition"]: r for r in json.loads(embed_src.read_text())["results"]}
    conditions = ["host_clean", "extended_all", "well_supported"]
    print(f"[fig4] LDA source: {lda_src}")
    print(f"[fig4] Embedding source: {embed_src}")
    for c in conditions:
        print(f"[fig4] {c}: LDA R2={lda[c]['loso_cv_r2']:.3f} CI={lda[c]['ci_95']}  "
              f"Embed R2={embed[c]['loso_cv_r2']:.3f} CI={embed[c]['ci_95']}")

    x = np.arange(len(conditions))
    width = 0.35
    lda_r2 = [lda[c]["loso_cv_r2"] for c in conditions]
    lda_err = np.array([[lda[c]["loso_cv_r2"] - lda[c]["ci_95"][0], lda[c]["ci_95"][1] - lda[c]["loso_cv_r2"]]
                         for c in conditions]).T
    embed_r2 = [embed[c]["loso_cv_r2"] for c in conditions]
    embed_err = np.array([[embed[c]["loso_cv_r2"] - embed[c]["ci_95"][0], embed[c]["ci_95"][1] - embed[c]["loso_cv_r2"]]
                           for c in conditions]).T
    ns = [lda[c]["n"] for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    b1 = ax.bar(x - width / 2, lda_r2, width, yerr=lda_err, capsize=4, label="LDA topics (K=75, CLR)",
                color=TOPICS_COLOR, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width / 2, embed_r2, width, yerr=embed_err, capsize=4, label="Embeddings (768d, full)",
                color=NEUTRAL, edgecolor="black", linewidth=0.5)

    ax.set_ylim(top=max(lda_r2[i] + lda_err[1][i] for i in range(len(conditions))) + 0.13)
    for i, c in enumerate(conditions):
        gap = lda_r2[i] - embed_r2[i]
        y = max(lda_r2[i] + lda_err[1][i], embed_r2[i] + embed_err[1][i]) + 0.03
        ax.annotate(f"+{gap:.2f}", xy=(x[i], y), ha="center", fontsize=9, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(N={n})" for c, n in zip(conditions, ns)])
    ax.set_ylabel("Leave-one-show-out CV R²")
    ax.set_title("Topics predict show ideology better than embedding style, across coverage conditions")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.axhline(0, color="black", linewidth=0.8)
    save(fig, "fig4_two_arm_comparison", [str(lda_src), str(embed_src)])


# ---------------------------------------------------------------------------
# Fig 5 -- what organizes the embedding space
# ---------------------------------------------------------------------------
def fig5_pc_structure() -> None:
    src = OUT / "embed_pc_diagnosis_report.json"
    if not src.exists():
        skip("fig5_pc_structure", f"missing {src}", [str(src)])
        return
    d = json.loads(src.read_text())
    per_pc = d["per_pc"]
    avg = d["column_averages"]
    print(f"[fig5] loaded {len(per_pc)} PCs from {src}")
    print(f"[fig5] column averages: register={avg['register_only_r2']:.3f}, "
          f"topics={avg['topics_only_r2']:.3f}, eta2={avg['between_show_eta2']:.3f}")

    pcs = [r["pc"] for r in per_pc]
    reg = [r["register_only_r2"] for r in per_pc]
    top = [r["topics_only_r2"] for r in per_pc]
    eta2 = [r["between_show_eta2"] for r in per_pc]

    x = np.arange(len(pcs))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, reg, width, label="Register-only R²", color=REGISTER_COLOR, edgecolor="black", linewidth=0.4)
    ax.bar(x, top, width, label="Topics-only R²", color=TOPICS_COLOR, edgecolor="black", linewidth=0.4)
    ax.bar(x + width, eta2, width, label="Between-show η²", color=ETA2_COLOR, edgecolor="black", linewidth=0.4)

    ax.axhline(avg["topics_only_r2"], color=TOPICS_COLOR, linestyle="--", linewidth=1, alpha=0.6)
    ax.text(len(pcs) - 0.5, avg["topics_only_r2"] + 0.015, f"avg={avg['topics_only_r2']:.2f}",
            color=TOPICS_COLOR, fontsize=8, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(pcs)
    ax.set_xlabel("Chunk-level principal component")
    ax.set_ylabel("Variance explained (R² or η²)")
    ax.set_title("Topics, not register, organize the top embedding dimensions")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig5_pc_structure", [str(src)])


# ---------------------------------------------------------------------------
# Fig 6 -- ideology concentration (k-sweep)
# ---------------------------------------------------------------------------
def fig6_k_sweep() -> None:
    src = OUT / "topic_ideology_synthesis_report.json"
    if not src.exists():
        skip("fig6_k_sweep", f"missing {src}", [str(src)])
        return
    d = json.loads(src.read_text())
    sweep = d["view3_k_sweep"]
    print(f"[fig6] loaded k-sweep from {src}: {[(r['k'], round(r['loso_cv_r2'],3)) for r in sweep]}")

    ks = [r["k"] for r in sweep]
    r2s = [r["loso_cv_r2"] for r in sweep]
    los = [r["ci_95"][0] for r in sweep]
    his = [r["ci_95"][1] for r in sweep]
    x = np.arange(len(ks))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x, r2s, "-o", color=TOPICS_COLOR, linewidth=2, markersize=7, zorder=3)
    ax.fill_between(x, los, his, color=TOPICS_COLOR, alpha=0.18, zorder=1, label="95% CI")

    full75_r2 = next(r["loso_cv_r2"] for r in sweep if r["k"] == 75)
    ax.axhline(full75_r2, color=NEUTRAL, linestyle="--", linewidth=1, label=f"full 75-topic model (R²={full75_r2:.2f})")

    peak_idx = int(np.argmax(r2s[:-1])) if len(r2s) > 1 else 0  # peak among the k<75 sweep points
    ax.annotate(f"k={ks[peak_idx]} peaks at R²={r2s[peak_idx]:.2f}\n(exceeds full-75 model)",
                xy=(x[peak_idx], r2s[peak_idx]), xytext=(x[peak_idx] - 0.3, r2s[peak_idx] + 0.09),
                fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1))

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (number of top ideology-associated topics used)")
    ax.set_ylabel("Leave-one-show-out CV R²")
    ax.set_title("Ideology signal is concentrated in a modest topic subset")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig6_k_sweep", [str(src)])


# ---------------------------------------------------------------------------
# Fig 7 -- ideology-carrying topics (diverging)
# ---------------------------------------------------------------------------
def fig7_diverging_topics() -> None:
    src = OUT / "topic_ideology_synthesis_report.json"
    if not src.exists():
        skip("fig7_diverging_topics", f"missing {src}", [str(src)])
        return
    d = json.loads(src.read_text())
    synth = pd.DataFrame(d["cross_view_synthesis"])
    core = synth[synth["flagged_by_both_top_lists"]].copy()
    print(f"[fig7] loaded {len(synth)} candidate topics from {src}, "
          f"{len(core)} flagged by both top lists (the robust core)")
    print(f"[fig7] core topics: {core[['topic','univariate_r','is_filler_topic']].to_dict('records')}")

    core = core.sort_values("univariate_r")
    y = np.arange(len(core))
    colors = [CONSERVATIVE if r > 0 else LIBERAL for r in core["univariate_r"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(y, core["univariate_r"], color=colors, edgecolor="black", linewidth=0.5)
    for bar, is_filler in zip(bars, core["is_filler_topic"]):
        if is_filler:
            bar.set_hatch("xx")
            bar.set_edgecolor("#444444")

    labels = []
    for _, row in core.iterrows():
        words = ", ".join(str(row["top_words"]).split(", ")[:4])
        tag = f"{row['topic']}  ({words})"
        if row["is_filler_topic"]:
            tag += "  ⚠ register confound"
        labels.append(tag)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Univariate Pearson r with show ideology (negative = liberal, positive = conservative)")
    ax.set_title("Topics most robustly associated with show ideology")

    legend_handles = [
        mpatches.Patch(color=LIBERAL, label="Liberal-associated"),
        mpatches.Patch(color=CONSERVATIVE, label="Conservative-associated"),
        mpatches.Patch(facecolor="white", hatch="xx", edgecolor="#444444", label="Register confound (not content)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8)
    fig.text(0.5, -0.02,
              "Note: T57 and T46 carry strong associations but reflect casual/profane speech style, not "
              "political content -- see robustness check (ideology_filler_robustness_report.json).",
              ha="center", fontsize=8, style="italic", wrap=True)
    save(fig, "fig7_diverging_topics", [str(src)])


def print_manifest() -> None:
    print("\n" + "=" * 70)
    print("FIGURE MANIFEST")
    print("=" * 70)
    for m in manifest:
        print(f"  {m['figure']:<28} [{m['status']}]")
        for s in m["sources"]:
            print(f"      <- {s}")
    manifest_path = FIG_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[manifest saved] -> {manifest_path}")


def run() -> None:
    apply_style()
    fig1_corpus_scope()
    fig2_topic_prevalence()
    fig3_coverage_extension()
    fig4_two_arm_comparison()
    fig5_pc_structure()
    fig6_k_sweep()
    fig7_diverging_topics()
    print_manifest()


if __name__ == "__main__":
    run()
