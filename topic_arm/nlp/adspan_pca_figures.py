"""Figures for the topic-PCA finding: ideology is not the dominant axis.

The argument these make visible
-------------------------------
fig1  the money plot -- the 204 shows on PC1 vs PC2 (the dominant axes) beside
      the same shows on PC5 vs PC6, both coloured by ideology. If ideology were
      dominant, PC1 would show a clean blue->red gradient. It does not; the
      gradient only appears on the later panel. One glance carries the finding.
fig2  the scree with ideology's position marked -- "1.5% of variance, rank ~14"
      made visual.
fig3  what each leading axis is MADE of -- top +/- topic loadings per PC.
fig4  biplot -- shows and topic loadings together, so the arrangement and its
      cause appear on one plane.

Colour encoding
---------------
Ideology is a SIGNED quantity, so it takes a DIVERGING scale: two hues with a
neutral midpoint, symmetric about zero. Blue (liberal, negative) -> neutral gray
-> red (conservative, positive), which is both the standard diverging
construction and the US political convention. A sequential ramp or a rainbow
would misrepresent zero as an arbitrary point on a magnitude scale rather than
the meaningful centre it is. vmin/vmax are forced symmetric so the neutral
colour lands exactly on 0.

These render on a light surface only -- they are paper/advisor figures, not a
themed web artifact.

    .venv/bin/python -m nlp.adspan_pca_figures [--k 75]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .adspan_phase_c import OUT_DIR
from .adspan_target_robustness import build_targets
from .adspan_topic_pca import load_topic_meta, pca_scores, show_topic_matrix

# --- palette (validated reference instance; diverging pair blue <-> red) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
LIBERAL = "#2a78d6"      # negative pole
NEUTRAL = "#f0efec"      # diverging midpoint -- gray, never a hue
CONSERVATIVE = "#e34948"  # positive pole
ACCENT = "#4a3aa7"       # single-series highlight (violet, slot 7)
IDEO_CMAP = LinearSegmentedColormap.from_list(
    "ideology", [LIBERAL, NEUTRAL, CONSERVATIVE])

# Which PC pair each panel shows. K=75 gives the clean before/after (ideology on
# PC5/PC6, far from the dominant pair). At K=30 ideology sits on PC2, which is
# ITSELF a dominant axis -- so the contrast is genuinely weaker there and the
# panels are chosen to show that honestly rather than manufacture a gap.
PANEL_PAIRS_BY_K = {75: [(1, 2), (5, 9)], 30: [(1, 2), (2, 3)]}
# The PC whose ideology correlation is strongest -- found by scanning ALL
# components, not just the Horn's-retained ones. At K=75 that is PC9, which
# Horn's does NOT retain: a component the dimensionality test calls noise
# carries the strongest single-PC ideological signal (|r|=0.465 extended,
# 0.468 host, 0.506 guest; Spearman 0.442; 0.414 after dropping the 10 most
# extreme shows, so not outlier-driven).
IDEOLOGY_PC = {75: 9, 30: 2}
LOADING_PCS = {75: [1, 2, 3, 4, 5, 6, 9], 30: [1, 2, 3, 4]}
N_TOP_LOADINGS = 7
BAR = "#6b6a64"      # single neutral hue: bar DIRECTION already encodes sign,
                     # so reusing the ideology blue/red here would make one
                     # colour pair mean two different things across the figure set
N_ARROWS = 6


def style(ax) -> None:
    """Recessive chrome: the data should be the darkest thing on the plot."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def topic_label(meta: list[dict], i: int, n: int = 2) -> str:
    return "/".join(meta[i]["top_words"][:n])


def load(k: int):
    show, diag = show_topic_matrix(k)
    X = diag["clr_matrix"]
    n_comp = int(min(X.shape[0] - 1, k - 1))
    scores, loadings, eig = pca_scores(X, n_comp)
    y = build_targets()["extended_all"]
    common = show.index.intersection(y.index)
    pos = show.index.get_indexer(common)
    return (show.index, scores, loadings, eig, pos,
            y.loc[common].to_numpy(dtype=float), load_topic_meta(k))


# ------------------------------------------------------------- fig 1 ----
def fig_scatter_pairs(scores, pos, yv, eig, k) -> None:
    PANEL_PAIRS = PANEL_PAIRS_BY_K[k]
    total = eig.sum()
    # Symmetric limits at the 95th percentile of |ideology|, not the max: a few
    # extreme shows otherwise compress the whole mid-range into near-neutral and
    # hide exactly the variation the plot is about. Colourbar is extended so the
    # clipped shows are visibly clipped rather than silently saturated.
    lim = float(np.percentile(np.abs(yv), 95))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.4), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.55, 1],
                                          "hspace": 0.30, "wspace": 0.22})
    for col, (a, b) in enumerate(PANEL_PAIRS):
        ax = axes[0, col]
        style(ax)
        xa, xb = scores[pos, a - 1], scores[pos, b - 1]
        ra = abs(np.corrcoef(xa, yv)[0, 1])
        rb = abs(np.corrcoef(xb, yv)[0, 1])
        # 2px surface ring so overlapping shows stay individually readable
        sc = ax.scatter(xa, xb, c=yv, cmap=IDEO_CMAP, norm=norm, s=78,
                        edgecolors=SURFACE, linewidths=1.6, zorder=3)
        ax.axhline(0, color=GRID, lw=1, zorder=1)
        ax.axvline(0, color=GRID, lw=1, zorder=1)
        ax.set_xlabel(f"PC{a}  ({eig[a-1]/total*100:.1f}% var)   |r| with ideology = {ra:.3f}",
                      color=INK, fontsize=10)
        ax.set_ylabel(f"PC{b}  ({eig[b-1]/total*100:.1f}% var)   |r| = {rb:.3f}",
                      color=INK, fontsize=10)

        # Binned means below each scatter. A scatter coloured by a variable with
        # |r|=0.28-0.47 CANNOT show a clean gradient -- r^2 is 0.08-0.22, so the
        # eye sees mostly noise and would read "no relationship" on both panels.
        # Binning the shows by PC score and plotting mean ideology per bin (with
        # 95% CIs) shows the real effect size honestly: flat on the dominant
        # axis, monotone on the ideological one. Without this row the figure
        # would understate the finding as badly as a hyped title overstates it.
        axb = axes[1, col]
        style(axb)
        # Left panel bins on the dominant axis; right bins on the STRONGEST
        # ideology axis (PC9), which is the y of the scatter above, not its x.
        bpc = a if col == 0 else IDEOLOGY_PC[k]
        xa = scores[pos, bpc - 1]
        q = np.quantile(xa, np.linspace(0, 1, 6))
        q[-1] += 1e-9
        binid = np.clip(np.digitize(xa, q[1:-1]), 0, 4)
        centres, means, errs = [], [], []
        for g in range(5):
            m = binid == g
            if m.sum() < 2:
                continue
            centres.append(float(xa[m].mean()))
            means.append(float(yv[m].mean()))
            errs.append(1.96 * float(yv[m].std(ddof=1)) / np.sqrt(m.sum()))
        axb.axhline(0, color=GRID, lw=1, zorder=1)
        axb.errorbar(centres, means, yerr=errs, fmt="o-", color=ACCENT,
                     ecolor=ACCENT, elinewidth=1.6, capsize=4, lw=2,
                     markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.6,
                     zorder=3)
        axb.set_xlabel(f"PC{bpc} score (shows binned into quintiles)",
                       color=INK, fontsize=10)
        axb.set_ylabel("mean ideology  (95% CI)", color=INK, fontsize=10)
        axb.set_title(
            "non-monotone: both extremes lean liberal, middle leans conservative\n"
            "— so linear |r|≈0.03 understates it, but there is no left→right order"
            if col == 0 else
            "monotone left→right — this is what an ideological axis looks like",
            color=INK_2, fontsize=10, loc="left")

    axes[0, 0].set_title("Dominant axes — 33% of variance",
                         color=INK, fontsize=13, fontweight="bold", pad=10, loc="left")
    axes[0, 1].set_title("Most ideology-correlated axes — 7.8% of variance",
                         color=INK, fontsize=13, fontweight="bold", pad=10, loc="left")

    cb = fig.colorbar(sc, ax=axes[0, :], fraction=0.028, pad=0.02, extend="both")
    cb.set_label("ideology  (liberal ← 0 → conservative)", color=INK, fontsize=10)
    cb.ax.tick_params(colors=INK_2, labelsize=9)
    cb.outline.set_edgecolor(GRID)
    fig.suptitle(
        f"204 shows in topic-PC space (K={k}, CLR + PCA) — colour is ideology.\n"
        f"Even the most ideological axis reaches only |r|=0.47 (r²=0.22): expect a tilt, not a split.",
        color=INK, fontsize=13.5, fontweight="bold", x=0.02, ha="left", y=1.005)
    p = OUT_DIR / f"fig1_pc_scatter_k{k}.png"
    fig.savefig(p, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p.name}")


# ------------------------------------------------------------- fig 2 ----
def fig_scree(scores, pos, yv, eig, k, retained: int, sup_var: float,
              sup_rank: int) -> None:
    """Two stacked panels sharing the PC axis -- deliberately NOT a dual-axis
    chart. % variance and |r| are different scales; overlaying them on twin y
    axes is the single most misleading thing a chart can do."""
    total = eig.sum()
    n = min(20, len(eig))
    pct = eig[:n] / total * 100
    rs = [abs(np.corrcoef(scores[pos, i], yv)[0, 1]) for i in range(n)]
    top = int(np.argmax(rs))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.4, 7.4), sharex=True, facecolor=SURFACE,
        gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.16})
    style(ax1); style(ax2)
    xs = np.arange(1, n + 1)

    c1 = [ACCENT if i == top else ("#b9b8b2" if i >= retained else "#8c8b85")
          for i in range(n)]
    ax1.bar(xs, pct, color=c1, width=0.68, zorder=3)
    ax1.axhline(sup_var, color=CONSERVATIVE, lw=2, ls="--", zorder=4)
    ax1.annotate(f"supervised ideology direction: {sup_var:.2f}% variance "
                 f"(would rank ~#{sup_rank})",
                 xy=(n * 0.62, sup_var), xytext=(n * 0.62, sup_var + 6.0),
                 color=CONSERVATIVE, fontsize=9.5, ha="center",
                 arrowprops=dict(arrowstyle="->", color=CONSERVATIVE, lw=1.2))
    ax1.set_ylabel("% of variance", color=INK, fontsize=10)
    ax1.text(0.995, 0.93, "dark = retained by Horn's   ·   light = not retained",
             transform=ax1.transAxes, ha="right", color=INK_2, fontsize=9.5)
    ax1.set_title(
        f"Ideology sits far down the variance ranking (K={k}; {retained} components retained by Horn's)",
        color=INK, fontsize=13, fontweight="bold", pad=10, loc="left")

    ax2.bar(xs, rs, color=[ACCENT if i == top else "#b9b8b2" for i in range(n)],
            width=0.68, zorder=3)
    ax2.set_ylabel("|r| with ideology", color=INK, fontsize=10)
    ax2.set_xlabel("principal component", color=INK, fontsize=10)
    ax2.set_xticks(xs)
    ax2.annotate(f"PC{top+1}: |r|={rs[top]:.3f} on {pct[top]:.1f}% of variance",
                 xy=(top + 1, rs[top]), xytext=(top + 2.4, rs[top] + 0.035),
                 color=ACCENT, fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))
    ax2.annotate(f"PC1: |r|={rs[0]:.3f}", xy=(1, rs[0]), xytext=(1.5, rs[0] + 0.09),
                 color=INK_2, fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.1))

    p = OUT_DIR / f"fig2_scree_ideology_k{k}.png"
    fig.savefig(p, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p.name}")


# ------------------------------------------------------------- fig 3 ----
def fig_loadings(loadings, meta, eig, k, readings: dict[int, str]) -> None:
    total = eig.sum()
    pcs = LOADING_PCS[k]
    ncol = 4 if len(pcs) > 4 else len(pcs)
    nrow = int(np.ceil(len(pcs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.6 * nrow),
                             facecolor=SURFACE)
    flat = np.atleast_1d(axes).ravel()
    for ax in flat[len(pcs):]:
        ax.set_visible(False)
    for ax, pc in zip(flat, pcs):
        c = pc - 1
        style(ax)
        ax.grid(axis="y", visible=False)
        w = loadings[:, c]
        o = np.argsort(w)
        idx = np.concatenate([o[:N_TOP_LOADINGS], o[::-1][:N_TOP_LOADINGS][::-1]])
        vals = w[idx]
        ypos = np.arange(len(idx))
        ax.barh(ypos, vals, color=BAR, height=0.72, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels([topic_label(meta, i) for i in idx], fontsize=8.2)
        ax.axvline(0, color=INK_2, lw=1, zorder=4)
        ax.set_title(f"PC{pc} — {eig[c]/total*100:.1f}% var\n{readings.get(pc,'')}",
                     color=INK, fontsize=10.5, fontweight="bold", loc="left")
        ax.tick_params(axis="y", length=0)
    fig.suptitle(
        f"What each axis is made of (K={k}) — bar direction is the loading sign; colour carries no meaning here",
        color=INK, fontsize=14.5, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT_DIR / f"fig3_pc_loadings_k{k}.png"
    fig.savefig(p, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p.name}")


# ------------------------------------------------------------- fig 4 ----
def fig_biplot(scores, loadings, pos, yv, eig, meta, k) -> None:
    PANEL_PAIRS = PANEL_PAIRS_BY_K[k]
    total = eig.sum()
    lim = float(np.percentile(np.abs(yv), 95))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.6), facecolor=SURFACE)

    for ax, (a, b) in zip(axes, PANEL_PAIRS):
        style(ax)
        xa, xb = scores[pos, a - 1], scores[pos, b - 1]
        sc = ax.scatter(xa, xb, c=yv, cmap=IDEO_CMAP, norm=norm, s=46,
                        edgecolors=SURFACE, linewidths=1.1, alpha=0.85, zorder=2)
        mag = np.hypot(loadings[:, a - 1], loadings[:, b - 1])
        pick = np.argsort(mag)[::-1][:N_ARROWS]
        scale = 0.86 * max(np.abs(xa).max(), np.abs(xb).max()) / mag[pick].max()
        for rank, i in enumerate(pick):
            dx, dy = loadings[i, a - 1] * scale, loadings[i, b - 1] * scale
            ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5,
                                        alpha=0.8), zorder=4)
            # Stagger the label radius by rank: near-parallel arrows otherwise
            # land their labels on top of each other, which is the single most
            # common way a biplot becomes unreadable.
            # Radial stagger separates labels on arrows of DIFFERENT length;
            # a perpendicular nudge separates those at nearly the SAME angle,
            # which the radial offset alone cannot fix.
            off = 1.08 + 0.26 * (rank % 3)
            n = np.hypot(dx, dy) or 1.0
            px, py = -dy / n, dx / n
            k_perp = 0.16 * n * (1 if rank % 2 else -1)
            ax.text(dx * off + px * k_perp, dy * off + py * k_perp,
                    topic_label(meta, i), color=INK,
                    fontsize=8.6, fontweight="bold", ha="center", va="center",
                    zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE,
                                        ec=GRID, lw=0.7, alpha=0.9))
        ax.set_xlabel(f"PC{a} ({eig[a-1]/total*100:.1f}% var)", color=INK, fontsize=10)
        ax.set_ylabel(f"PC{b} ({eig[b-1]/total*100:.1f}% var)", color=INK, fontsize=10)
        ax.set_title(f"PC{a} vs PC{b}", color=INK, fontsize=12, fontweight="bold")

    cb = fig.colorbar(sc, ax=axes, fraction=0.026, pad=0.02)
    cb.set_label("ideology", color=INK, fontsize=10)
    cb.ax.tick_params(colors=INK_2, labelsize=9)
    cb.outline.set_edgecolor(GRID)
    fig.suptitle(f"Biplot: shows (dots) and topic loadings (arrows), K={k}",
                 color=INK, fontsize=14.5, fontweight="bold", x=0.045, ha="left")
    p = OUT_DIR / f"fig4_biplot_k{k}.png"
    fig.savefig(p, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p.name}")


READINGS = {
    75: {1: "domestic institutions vs. structural/international",
         2: "geopolitics vs. culture & religion",
         3: "local/tech policy vs. identity flashpoints",
         4: "campaign horserace vs. courtroom",
         5: "constitutional democracy vs. lifestyle  (|r|=0.28)",
         6: "religion/local vs. profanity-banter  (|r|=0.21)",
         9: "NOT retained by Horn's — yet |r|=0.47, the strongest"},
    30: {1: "foreign/war/religion vs. domestic legal-electoral",
         2: "geopolitics vs. religion & register  ← ideology",
         3: "identity flashpoints vs. wonk/tech",
         4: "horserace vs. courtroom/security",
         5: "", 6: ""},
}
SUPERVISED = {75: (1.51, 14), 30: (1.59, 13)}
RETAINED = {75: 7, 30: 4}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=75)
    args = ap.parse_args()
    k = args.k

    _, scores, loadings, eig, pos, yv, meta = load(k)
    print(f"[data] K={k}: {len(pos)} shows with ideology, {len(eig)} components")
    fig_scatter_pairs(scores, pos, yv, eig, k)
    fig_scree(scores, pos, yv, eig, k, RETAINED[k], *SUPERVISED[k])
    fig_loadings(loadings, meta, eig, k, READINGS[k])
    fig_biplot(scores, loadings, pos, yv, eig, meta, k)


if __name__ == "__main__":
    main()
