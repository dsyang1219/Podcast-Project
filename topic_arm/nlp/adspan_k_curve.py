"""The K selection curve: c_v across K, with the 1SE band that drives the rule.

What this is meant to show
--------------------------
That the c_v "peak" at K=75 is NOT sharp. The selection SE is 0.0132 (over the
75 per-topic scores at the peak K), and K=75 beats K=100 by only 0.0097 and
K=90 by 0.0107 -- both smaller than one SE. Five of the seven fitted K values
sit inside the 1SE band, so coherence cannot discriminate within K=30-100. The
shaded band makes that immediately visible, which a bare line chart would hide.

Three stacked panels share the K axis rather than sharing a y axis. c_v, c_npmi
and ideology R2 are on different scales, and overlaying them on twin axes would
be a dual-axis chart -- the single most misleading chart form there is. Separate
panels let each keep its own honest scale.

Panel 3 (ideology R2) is included because the metric disagreement is the
interesting part, and it is labelled as MEASURED, NEVER USED TO SELECT K.
Selecting K on the outcome and then reporting that outcome at the selected K
would be circular; this panel is here to be read after the selection, not to
justify it.

    .venv/bin/python -m nlp.adspan_k_curve
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .adspan_phase_c import OUT_DIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
ACCENT = "#4a3aa7"        # the fitted curve (single series -- no legend needed)
PEAK = "#e34948"          # peak / threshold annotation
SELECTED = "#1baf7a"      # the K the rule actually picks
BAND = "#e8e6f2"          # 1SE band fill


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, color=GRID)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def main() -> None:
    sweep = json.loads((OUT_DIR / "phase_c_ksweep.json").read_text())
    rep = json.loads((OUT_DIR / "phase_c_report.json").read_text())
    res = sorted(sweep["results"], key=lambda r: r["k"])

    ks = np.array([r["k"] for r in res], dtype=float)
    cv = np.array([r["coherence"]["c_v"] for r in res])
    npmi = np.array([r["coherence"]["c_npmi"] for r in res])
    r2 = np.array([r["ideology"]["loso_cv_r2"] for r in res])
    lo = np.array([r["ideology"]["ci_95"][0] for r in res])
    hi = np.array([r["ideology"]["ci_95"][1] for r in res])
    # Per-K standard error of the mean over that K's own per-topic c_v scores.
    # The RULE only uses the SE at the peak K, but showing every K's spread is
    # the honest version: it makes clear the curve's wiggles are inside noise.
    se = np.array([np.std(r["per_topic_cv"], ddof=1) / np.sqrt(len(r["per_topic_cv"]))
                   for r in res])

    sel = rep["selection"]
    peak_k, peak_cv = sel["peak_cv_k"], sel["peak_cv"]
    thr, chosen = sel["threshold"], sel["chosen_k"]
    inside = [int(k) for k, v in zip(ks, cv) if v >= thr]

    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(10.6, 11.2), sharex=True, facecolor=SURFACE,
        gridspec_kw={"height_ratios": [1.5, 1, 1.15], "hspace": 0.14})
    for ax in (a1, a2, a3):
        style(ax)

    # ---- panel 1: c_v, the selection criterion ----
    a1.axhspan(thr, peak_cv, color=BAND, zorder=1)
    a1.axhline(peak_cv, color=PEAK, lw=1.4, ls="-", zorder=2)
    a1.axhline(thr, color=PEAK, lw=1.4, ls="--", zorder=2)
    a1.errorbar(ks, cv, yerr=se, fmt="o-", color=ACCENT, ecolor=ACCENT,
                elinewidth=1.4, capsize=4, lw=2.2, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
    a1.scatter([peak_k], [peak_cv], s=190, facecolor="none", edgecolor=PEAK,
               linewidths=2.2, zorder=5)
    a1.scatter([chosen], [cv[list(ks).index(chosen)]], s=190, facecolor="none",
               edgecolor=SELECTED, linewidths=2.2, zorder=5)
    a1.text(peak_k, peak_cv + 0.0045, f"peak  K={peak_k}", color=PEAK,
            fontsize=10, fontweight="bold", ha="center")
    # Drop the label clear of the peak−1SE line: at K=30 the point sits high in
    # the band, so a label placed just under it lands on the dashed threshold.
    a1.annotate(f"selected  K={chosen}\n(smallest inside band)",
                xy=(chosen, cv[list(ks).index(chosen)]),
                xytext=(chosen + 6, float((cv - se).min()) + 0.001),
                color=SELECTED, fontsize=10, fontweight="bold",
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=SELECTED, lw=1.4))
    a1.text(151, thr + 0.0012, "peak − 1SE", color=PEAK, fontsize=9.5,
            va="bottom", ha="right")
    a1.text(151, peak_cv + 0.0012, "peak", color=PEAK, fontsize=9.5,
            va="bottom", ha="right")
    a1.set_ylabel("c_v  (selection criterion)", color=INK, fontsize=10.5)
    a1.set_title(
        "The c_v peak is not sharp: 5 of 7 K values sit inside the 1SE band\n"
        f"band = peak {peak_cv:.4f} − 1SE {sel['se']:.4f} = {thr:.4f};  "
        f"inside: K = {', '.join(map(str, inside))}",
        color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    # Limits must contain every error bar: a clipped whisker reads as a shorter
    # interval than it is, which would understate exactly the overlap this
    # panel exists to show.
    a1.set_ylim(float((cv - se).min()) - 0.004, float((cv + se).max()) + 0.004)

    # ---- panel 2: c_npmi, which disagrees ----
    a2.plot(ks, npmi, "o-", color=ACCENT, lw=2.2, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
    a2.scatter([ks[int(np.argmax(npmi))]], [npmi.max()], s=190, facecolor="none",
               edgecolor=PEAK, linewidths=2.2, zorder=4)
    a2.set_ylabel("c_npmi", color=INK, fontsize=10.5)
    a2.set_title("c_npmi rises monotonically and peaks at K=150 — the two "
                 "coherence metrics disagree",
                 color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)

    # ---- panel 3: ideology R2, measured but never used to select ----
    a3.fill_between(ks, lo, hi, color=ACCENT, alpha=0.15, zorder=2)
    a3.plot(ks, r2, "o-", color=ACCENT, lw=2.2, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
    a3.axvspan(75, 90, color=SELECTED, alpha=0.10, zorder=1)
    a3.text(82.5, min(lo) + 0.012, "R² plateau\nK=75–90", color=INK_2,
            fontsize=9.5, ha="center", va="bottom")
    a3.set_ylabel("ideology LOSO-CV R²  (95% CI)", color=INK, fontsize=10.5)
    a3.set_xlabel("K (number of topics)", color=INK, fontsize=11)
    a3.set_title("Ideology R² — MEASURED at every K, NEVER used to select K "
                 "(using it would be circular)",
                 color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)

    a3.set_xticks(ks)
    a3.set_xticklabels([str(int(k)) for k in ks])
    a1.set_xlim(22, 168)

    fig.suptitle("K selection curve — ad-span-cleaned corpus, moderate regime, 204 shows",
                 color=INK, fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.997)
    p = OUT_DIR / "fig5_k_curve.png"
    fig.savefig(p, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")
    print(f"[band] threshold={thr:.4f}  inside={inside}  peak K={peak_k}  chosen K={chosen}")
    for k, v, s in zip(ks, cv, se):
        print(f"  K={int(k):<4} c_v={v:.4f}  SE={s:.4f}  "
              f"{'inside band' if v >= thr else 'below band'}")


if __name__ == "__main__":
    main()
