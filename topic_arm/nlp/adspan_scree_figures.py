"""Two summary figures the earlier scree panel only half-covered.

fig6  Scree against the ACTUAL Horn's noise line. Observed eigenvalues as bars,
      the permutation null's 95th percentile drawn across as a line, the
      retained region shaded, and the ideology-bearing component marked --
      which at K=75 sits BELOW the line. "The strongest ideological axis does
      not survive the noise threshold" becomes a picture instead of a caveat.

      The earlier fig2 drew a line at the SUPERVISED direction's variance, not
      the Horn's null, so it could not make this point.

fig7  Correlation with ideology per component, with variance share encoded as
      BAR WIDTH. Both halves of "recoverable but not dominant" in one frame:
      the wide bars on the left (high variance) are flat at ~0, and the bar
      that finally rises is narrow.

      Width, not a twin y-axis. Two y-scales on one plot is the single most
      misleading chart form there is, and this figure exists precisely to be
      read quickly -- so the second variable rides on a geometric channel that
      cannot be misread as sharing the first one's scale.

    .venv/bin/python -m nlp.adspan_scree_figures --k 75
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .adspan_phase_c import OUT_DIR
from .adspan_target_robustness import build_targets
from .adspan_topic_pca import pca_scores, show_topic_matrix

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
KEPT = "#4a3aa7"       # retained components
DROPPED = "#b9b8b2"    # below the noise line
NULLC = "#e34948"      # the Horn's null line
IDEO = "#1baf7a"       # the ideology-bearing component
BAND = "#ece9f5"

N_SHOW = 20


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def load(k):
    rep = json.loads((OUT_DIR / "topic_pca_dimensionality.json").read_text())
    h = rep["per_k"][str(k)]["step2_horns"]
    show, diag = show_topic_matrix(k)
    X = diag["clr_matrix"]
    sc, ld, eig = pca_scores(X, int(min(X.shape[0] - 1, k - 1)))
    y = build_targets()["extended_all"]
    common = show.index.intersection(y.index)
    pos = show.index.get_indexer(common)
    yv = y.loc[common].to_numpy(float)
    r = np.array([abs(np.corrcoef(sc[pos, i], yv)[0, 1]) for i in range(len(eig))])
    return h, eig, r


def fig6(k, h, eig, r):
    n = min(N_SHOW, len(h["observed_eigenvalues"]))
    obs = np.array(h["observed_eigenvalues"][:n])
    null = np.array(h["null_95th_percentile"][:n])
    ret = h["retained_dimensions"]
    ipc = int(np.argmax(r[:n])) + 1
    xs = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(11.8, 6.6), facecolor=SURFACE)
    style(ax)
    ax.axvspan(0.4, ret + 0.5, color=BAND, zorder=1)
    cols = [KEPT if i < ret else DROPPED for i in range(n)]
    if ipc - 1 >= ret:
        cols[ipc - 1] = IDEO
    ax.bar(xs, obs, color=cols, width=0.7, zorder=3)
    ax.plot(xs, null, "-", color=NULLC, lw=2.4, zorder=5,
            label="Horn's null, 95th percentile of permuted eigenvalues")
    ax.scatter(xs, null, s=26, color=NULLC, zorder=6, edgecolor=SURFACE, linewidths=1)

    ax.text(ret / 2 + 0.4, ax.get_ylim()[1] * 0.93,
            f"retained: {ret} components", color=KEPT, fontsize=11,
            fontweight="bold", ha="center")
    below = obs[ipc - 1] < null[ipc - 1]
    ax.annotate(
        f"PC{ipc}: strongest ideology correlation (|r|={r[ipc-1]:.3f})\n"
        f"eigenvalue {obs[ipc-1]:.2f} vs null {null[ipc-1]:.2f} — "
        + ("BELOW the noise line" if below else "above the noise line"),
        xy=(ipc, obs[ipc - 1]), xytext=(ipc + 2.2, max(obs) * 0.55),
        color=IDEO if below else INK, fontsize=10.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=IDEO if below else INK, lw=1.8))

    ax.set_xticks(xs)
    ax.set_xlabel("principal component", color=INK, fontsize=11)
    ax.set_ylabel("eigenvalue", color=INK, fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK_2)
    ax.set_title(
        f"Scree against the Horn's noise line (K={k}) — "
        f"the strongest ideological axis is {'below' if below else 'above'} it",
        color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=12)
    p = OUT_DIR / f"fig6_scree_horns_k{k}.png"
    fig.savefig(p, dpi=130, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}  (PC{ipc} {'below' if below else 'above'} null)")


def fig7(k, h, eig, r):
    n = min(N_SHOW, len(eig))
    tot = eig.sum()
    pct = eig[:n] / tot * 100
    ret = h["retained_dimensions"]
    ipc = int(np.argmax(r[:n])) + 1
    xs = np.arange(1, n + 1)
    # Width encodes variance share. Floor it so a 2%-variance bar stays
    # clickable-thin rather than invisible; the ratio is still ~6x across the
    # range, which is what the eye needs to read "wide = big axis".
    w = 0.16 + 0.74 * (pct / pct.max())

    fig, ax = plt.subplots(figsize=(11.8, 6.4), facecolor=SURFACE)
    style(ax)
    cols = [KEPT if i < ret else DROPPED for i in range(n)]
    cols[ipc - 1] = IDEO
    ax.bar(xs, r[:n], width=w, color=cols, zorder=3)
    for i in range(n):
        ax.text(xs[i], r[i] + 0.012, f"{pct[i]:.1f}", ha="center",
                color=INK_2, fontsize=7.6)
    ax.annotate(
        f"PC{ipc}: |r|={r[ipc-1]:.3f} on only {pct[ipc-1]:.1f}% of variance"
        + ("  (not retained by Horn's)" if ipc - 1 >= ret else ""),
        xy=(ipc, r[ipc - 1]), xytext=(ipc + 1.4, r[ipc - 1] + 0.06),
        color=IDEO, fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=IDEO, lw=1.8))
    ax.annotate(
        f"PC1: {pct[0]:.1f}% of variance, |r|={r[0]:.3f}",
        xy=(1, r[0]), xytext=(1.5, max(r[:n]) * 0.62),
        color=INK_2, fontsize=10,
        arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.4))

    ax.set_xticks(xs)
    ax.set_xlabel("principal component   (bar WIDTH = share of variance; "
                  "% of variance printed above each bar)", color=INK, fontsize=10.5)
    ax.set_ylabel("|r| with ideology", color=INK, fontsize=11)
    ax.set_ylim(0, max(r[:n]) * 1.22)
    ax.set_title(
        f"Recoverable but not dominant (K={k}) — wide bars carry the variance, "
        f"the correlated bar is narrow",
        color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=12)
    p = OUT_DIR / f"fig7_ideology_by_component_k{k}.png"
    fig.savefig(p, dpi=130, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=75)
    a = ap.parse_args()
    h, eig, r = load(a.k)
    fig6(a.k, h, eig, r)
    fig7(a.k, h, eig, r)


if __name__ == "__main__":
    main()
