"""Component-anatomy and show-arrangement figures for the topic PCA.

Five figures, generated for any K:

  A  diverging loadings bars, one panel per leading component -- what each axis
     is MADE of, sorted by |loading| so the strongest contributors read first.
  B  loadings heatmap (topics x PC1-10) -- the whole structure at once; topics
     that anchor several components show up as a dark row.
  C  shows on PC1xPC2 coloured two ways: by ideology (scattered) and by register
     rate (arranged). The contrast IS the argument.
  D  paired scatter: dominant axes beside the most ideology-correlated axes.
  E  same arrangement with extreme shows labelled, to ground abstract axes in
     real podcasts.

Colour rules (they differ per figure by the JOB the colour does)
---------------------------------------------------------------
* ideology is SIGNED -> diverging blue/gray/red, symmetric about zero.
* loadings are SIGNED -> the same diverging construction (B), and in A the sign
  is carried by bar DIRECTION so the bars take one neutral hue instead.
* register rate is a MAGNITUDE, not signed -> SEQUENTIAL single hue, light to
  dark. Using the ideology diverging scale for it would imply a meaningful
  midpoint that does not exist, and would make one palette mean two things.

    .venv/bin/python -m nlp.adspan_component_figures --k 75
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .adspan_phase_c import OUT_DIR
from .adspan_pca_diagnostics import REGISTER_SETS
from .adspan_target_robustness import TARGETS_PATH, build_targets
from .adspan_topic_pca import load_topic_meta, pca_scores, show_topic_matrix

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
LIB, MID, CON = "#2a78d6", "#f0efec", "#e34948"
BAR = "#6b6a64"
ACCENT = "#4a3aa7"
IDEO_CMAP = LinearSegmentedColormap.from_list("ideology", [LIB, MID, CON])
LOAD_CMAP = LinearSegmentedColormap.from_list("loading", [LIB, MID, CON])
# Sequential, single hue, light->dark -- register rate is a magnitude.
REG_CMAP = LinearSegmentedColormap.from_list("register", ["#f2f7f4", "#1baf7a", "#0b4b34"])

N_BARS = 12
RETAINED = {75: 7, 30: 4}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def lab(meta, i, n=2):
    return "/".join(meta[i]["top_words"][:n])


def load(k):
    show, diag = show_topic_matrix(k)
    X = diag["clr_matrix"]
    n = int(min(X.shape[0] - 1, k - 1))
    sc, ld, eig = pca_scores(X, n)
    meta = load_topic_meta(k)
    y = build_targets()["extended_all"]
    common = show.index.intersection(y.index)
    pos = show.index.get_indexer(common)
    yv = y.loc[common].to_numpy(float)
    raw = show.to_numpy(float)
    reg_idx = REGISTER_SETS[k]["core"]
    reg_rate = raw[:, reg_idx].sum(axis=1)          # share of a show's content
    names = (pd.read_csv(TARGETS_PATH, dtype={"show_id": str})
             .set_index("show_id")["show"])
    return show.index, sc, ld, eig, pos, yv, meta, reg_rate, names


def ideology_pcs(sc, pos, yv, n=10):
    r = [abs(np.corrcoef(sc[pos, i], yv)[0, 1]) for i in range(n)]
    order = np.argsort(r)[::-1]
    return int(order[0]) + 1, int(order[1]) + 1, r


# ------------------------------------------------------------------ A ----
def fig_a(ld, eig, meta, k, reg_idx):
    tot = eig.sum()
    pcs = list(range(1, 6))
    fig, axes = plt.subplots(1, len(pcs), figsize=(4.4 * len(pcs), 6.4),
                             facecolor=SURFACE)
    for ax, pc in zip(np.atleast_1d(axes).ravel(), pcs):
        style(ax); ax.grid(axis="y", visible=False)
        w = ld[:, pc - 1]
        idx = np.argsort(np.abs(w))[::-1][:N_BARS]      # by MAGNITUDE
        idx = idx[np.argsort(w[idx])]                   # then draw sorted by value
        cols = [CON if w[i] > 0 else LIB for i in idx]
        # Register topics get the accent colour so contamination is impossible
        # to miss -- this is the one place colour carries a second meaning, and
        # it is labelled in the title rather than left implicit.
        cols = [ACCENT if i in reg_idx else c for i, c in zip(idx, cols)]
        ax.barh(np.arange(len(idx)), w[idx], color=cols, height=0.74, zorder=3)
        ax.set_yticks(np.arange(len(idx)))
        ax.set_yticklabels([lab(meta, i) for i in idx], fontsize=8.4)
        ax.tick_params(axis="y", length=0)
        ax.axvline(0, color=INK_2, lw=1.1, zorder=4)
        ax.set_title(f"PC{pc} — {eig[pc-1]/tot*100:.1f}% var", color=INK,
                     fontsize=12, fontweight="bold", loc="left")
        ax.set_xlabel("loading", color=INK_2, fontsize=9)
    fig.suptitle(f"What each dominant axis is made of (K={k}) — top {N_BARS} topics by |loading|;  "
                 f"purple = register topic",
                 color=INK, fontsize=14, fontweight="bold", x=0.006, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    p = OUT_DIR / f"figA_loadings_bars_k{k}.png"
    fig.savefig(p, dpi=125, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")


# ------------------------------------------------------------------ B ----
def fig_b(ld, eig, meta, k, reg_idx):
    tot = eig.sum()
    npc = min(10, ld.shape[1])
    M = ld[:, :npc]
    order = np.argsort(np.abs(M).max(axis=1))[::-1]     # loudest topics first
    M, order = M[order], order
    lim = float(np.abs(M).max())
    fig, ax = plt.subplots(figsize=(9.2, 0.30 * len(order) + 2.6), facecolor=SURFACE)
    im = ax.imshow(M, cmap=LOAD_CMAP, norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), aspect="auto")
    ax.set_xticks(range(npc))
    ax.set_xticklabels([f"PC{i+1}\n{eig[i]/tot*100:.1f}%" for i in range(npc)], fontsize=8.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [("* " if i in reg_idx else "") + lab(meta, i) for i in order], fontsize=7.2)
    for t, i in zip(ax.get_yticklabels(), order):
        if i in reg_idx:
            t.set_color(ACCENT); t.set_fontweight("bold")
    ax.tick_params(colors=INK_2, length=0)
    ax.axvline(RETAINED[k] - 0.5, color=INK, lw=2, ls="--")
    # Inside the axes, in axes-fraction coords: data coords put it above the
    # top row and straight through the title.
    ax.text((RETAINED[k] - 0.42) / npc, 1.004, "← Horn's cutoff",
            transform=ax.transAxes, color=INK, fontsize=9,
            fontweight="bold", va="bottom", ha="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("loading", color=INK, fontsize=10)
    cb.ax.tick_params(colors=INK_2, labelsize=9); cb.outline.set_edgecolor(GRID)
    ax.set_title(f"Topic × component loadings (K={k}) — topics sorted by peak |loading|;  "
                 f"* = register topic",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=16)
    p = OUT_DIR / f"figB_loadings_heatmap_k{k}.png"
    fig.savefig(p, dpi=125, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")


def _scatter(ax, x, y, c, cmap, norm, s=74):
    return ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=s,
                      edgecolors=SURFACE, linewidths=1.5, zorder=3)


# ------------------------------------------------------------------ C ----
def fig_c(sc, eig, pos, yv, reg_rate, k):
    tot = eig.sum()
    x, y = sc[pos, 0], sc[pos, 1]
    rr = reg_rate[pos]
    lim = float(np.percentile(np.abs(yv), 95))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.6, 6.0), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.42)
    style(a1); style(a2)

    s1 = _scatter(a1, x, y, yv, IDEO_CMAP, TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    r1 = abs(np.corrcoef(x, yv)[0, 1]); r1b = abs(np.corrcoef(y, yv)[0, 1])
    a1.set_title(f"coloured by IDEOLOGY — scattered\n|r| with PC1 = {r1:.3f}, with PC2 = {r1b:.3f}",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left")
    cb1 = fig.colorbar(s1, ax=a1, fraction=0.046, pad=0.13, extend="both")
    cb1.set_label("ideology", color=INK, fontsize=9.5)

    s2 = _scatter(a2, x, y, rr, REG_CMAP,
                  matplotlib.colors.Normalize(float(rr.min()),
                                              float(np.percentile(rr, 97))))
    r2 = abs(np.corrcoef(x, rr)[0, 1]); r2b = abs(np.corrcoef(y, rr)[0, 1])
    a2.set_title(f"coloured by REGISTER RATE — arranged\n|r| with PC1 = {r2:.3f}, with PC2 = {r2b:.3f}",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left")
    cb2 = fig.colorbar(s2, ax=a2, fraction=0.046, pad=0.13, extend="max")
    cb2.set_label("register share", color=INK, fontsize=9.5)

    for ax, cb in ((a1, cb1), (a2, cb2)):
        ax.set_xlabel(f"PC1 ({eig[0]/tot*100:.1f}% var)", color=INK, fontsize=10)
        ax.set_ylabel(f"PC2 ({eig[1]/tot*100:.1f}% var)", color=INK, fontsize=10)
        ax.axhline(0, color=GRID, lw=1, zorder=1); ax.axvline(0, color=GRID, lw=1, zorder=1)
        cb.ax.tick_params(colors=INK_2, labelsize=9); cb.outline.set_edgecolor(GRID)
    fig.suptitle(f"Same 204 shows, same dominant axes, two colourings (K={k}) — "
                 f"what PC1/PC2 organise is NOT ideology",
                 color=INK, fontsize=14, fontweight="bold", x=0.006, ha="left", y=1.02)
    p = OUT_DIR / f"figC_ideology_vs_register_k{k}.png"
    fig.savefig(p, dpi=125, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")


# ------------------------------------------------------------------ D ----
def fig_d(sc, eig, pos, yv, k, ipc1, ipc2, rlist):
    tot = eig.sum()
    lim = float(np.percentile(np.abs(yv), 95))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    pairs = [(1, 2), (ipc2, ipc1)]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.0), facecolor=SURFACE)
    for ax, (a, b) in zip(axes, pairs):
        style(ax)
        s = _scatter(ax, sc[pos, a - 1], sc[pos, b - 1], yv, IDEO_CMAP, norm)
        ax.axhline(0, color=GRID, lw=1, zorder=1); ax.axvline(0, color=GRID, lw=1, zorder=1)
        ax.set_xlabel(f"PC{a} ({eig[a-1]/tot*100:.1f}% var, |r|={rlist[a-1]:.3f})",
                      color=INK, fontsize=10)
        ax.set_ylabel(f"PC{b} ({eig[b-1]/tot*100:.1f}% var, |r|={rlist[b-1]:.3f})",
                      color=INK, fontsize=10)
    v = eig[[0, 1]].sum() / tot * 100
    w = eig[[ipc1 - 1, ipc2 - 1]].sum() / tot * 100
    axes[0].set_title(f"Dominant axes — {v:.0f}% of variance",
                      color=INK, fontsize=12.5, fontweight="bold", loc="left")
    axes[1].set_title(f"Most ideology-correlated axes — {w:.0f}% of variance",
                      color=INK, fontsize=12.5, fontweight="bold", loc="left")
    cb = fig.colorbar(s, ax=axes, fraction=0.028, pad=0.02, extend="both")
    cb.set_label("ideology", color=INK, fontsize=10)
    cb.ax.tick_params(colors=INK_2, labelsize=9); cb.outline.set_edgecolor(GRID)
    fig.suptitle(f"Ideology is present but minor (K={k}) — strongest single-PC "
                 f"|r| = {max(rlist):.3f} on PC{ipc1}",
                 color=INK, fontsize=14, fontweight="bold", x=0.006, ha="left", y=1.0)
    p = OUT_DIR / f"figD_paired_scatter_k{k}.png"
    fig.savefig(p, dpi=125, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")


# ------------------------------------------------------------------ E ----
def fig_e(idx, sc, eig, pos, yv, names, k, n_label=10):
    tot = eig.sum()
    x, y = sc[pos, 0], sc[pos, 1]
    lim = float(np.percentile(np.abs(yv), 95))
    fig, ax = plt.subplots(figsize=(11.6, 8.4), facecolor=SURFACE)
    style(ax)
    s = _scatter(ax, x, y, yv, IDEO_CMAP, TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), s=86)
    ax.axhline(0, color=GRID, lw=1, zorder=1); ax.axvline(0, color=GRID, lw=1, zorder=1)

    ids = idx[pos]
    # Label the extremes of BOTH axes rather than the largest radius, so each
    # pole of each component gets an anchor a reader can recognise.
    picks = set()
    for arr in (x, y):
        o = np.argsort(arr)
        picks.update(o[:n_label // 4].tolist()); picks.update(o[-(n_label // 4):].tolist())
    xm, ym = float(np.median(x)), float(np.median(y))
    for i in sorted(picks):
        nm = str(names.get(ids[i], ids[i]))
        nm = nm if len(nm) <= 30 else nm[:29] + "…"
        # Point the label back toward the centre of the cloud: an extreme show
        # sits at the panel edge, so a fixed outward offset pushes its label off
        # the axes and into the colourbar.
        right = x[i] > xm
        dx, ha = (-8, "right") if right else (8, "left")
        dy = -12 if y[i] > ym else 12
        ax.annotate(nm, (x[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.6, color=INK, fontweight="bold", zorder=6,
                    ha=ha, va="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec=GRID,
                              lw=0.7, alpha=0.92))
    ax.margins(x=0.13, y=0.09)
    ax.set_xlabel(f"PC1 ({eig[0]/tot*100:.1f}% var)", color=INK, fontsize=10.5)
    ax.set_ylabel(f"PC2 ({eig[1]/tot*100:.1f}% var)", color=INK, fontsize=10.5)
    cb = fig.colorbar(s, ax=ax, fraction=0.04, pad=0.02, extend="both")
    cb.set_label("ideology (liberal ← 0 → conservative)", color=INK, fontsize=10)
    cb.ax.tick_params(colors=INK_2, labelsize=9); cb.outline.set_edgecolor(GRID)
    ax.set_title(f"Shows in the dominant plane (K={k}), extremes labelled — "
                 f"the poles are topical, not ideological",
                 color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=12)
    p = OUT_DIR / f"figE_shows_labeled_k{k}.png"
    fig.savefig(p, dpi=125, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=75)
    a = ap.parse_args()
    k = a.k
    idx, sc, ld, eig, pos, yv, meta, reg_rate, names = load(k)
    reg_idx = set(REGISTER_SETS[k]["core"])
    ipc1, ipc2, rlist = ideology_pcs(sc, pos, yv)
    print(f"[k={k}] top ideology PCs: PC{ipc1} (|r|={rlist[ipc1-1]:.3f}), "
          f"PC{ipc2} (|r|={rlist[ipc2-1]:.3f}); register topics {sorted(reg_idx)}")
    fig_a(ld, eig, meta, k, reg_idx)
    fig_b(ld, eig, meta, k, reg_idx)
    fig_c(sc, eig, pos, yv, reg_rate, k)
    fig_d(sc, eig, pos, yv, k, ipc1, ipc2, rlist)
    fig_e(idx, sc, eig, pos, yv, names, k)


if __name__ == "__main__":
    main()
