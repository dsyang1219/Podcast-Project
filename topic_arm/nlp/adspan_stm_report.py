"""Read the STM arm's outputs and produce the Q1 / Q2 descriptive tables.

CIRCULARITY BOUNDARY -- every claim this file produces is DESCRIPTIVE.
Ideology is an input to the STM, so nothing here is evidence that ideology is
recoverable from discourse. Phrase results as "conservative-leaning shows
discuss X more", never as "topics predict ideology". The recoverability claim
belongs to the unsupervised LDA/PCA arm alone.

Q1 reports TWO sets of intervals for the same effects:
  - passage-level (estimateEffect, n=37,942, uncertainty="Global")
  - show-clustered (theta collapsed to show means, OLS on n=204 shows)
The passage-level CIs are ANTI-CONSERVATIVE because passages inherit their
show's ideology and are not independent. The clustered version is the honest
effective sample size and is what the writeup should quote. They are printed
side by side so the inflation is visible rather than hidden.

    .venv/bin/python -m nlp.adspan_stm_report [--k 75] [--top 12]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .adspan_phase_c import OUT_DIR

pd.set_option("display.width", 200)


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def short(words: str, n: int = 8) -> str:
    return " ".join(str(words).split()[:n])


def q1(k: int, top: int) -> pd.DataFrame:
    p = OUT_DIR / f"stm_prevalence_effects_k{k}.csv"
    if not p.exists():
        print(f"[q1] {p.name} not present yet")
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["ci_low_show"] = df["est_show"] - 1.96 * df["se_show"]
    df["ci_high_show"] = df["est_show"] + 1.96 * df["se_show"]
    # Bonferroni over K topics, on the CLUSTERED p-values -- the passage-level
    # ones are not trustworthy enough to correct.
    thresh = 0.05 / len(df)
    df["survives_bonferroni_clustered"] = df["p_show"] < thresh
    df["se_inflation"] = df["se_show"] / df["se"]

    print(f"\n{'=' * 100}\nQ1 -- PREVALENCE: which topics does ideology predict more/less of? (K={k})\n{'=' * 100}")
    print(f"Clustered Bonferroni threshold p < {thresh:.2e} ({len(df)} topics)")
    print(f"SE inflation passage->show: median {df['se_inflation'].median():.1f}x "
          f"(range {df['se_inflation'].min():.1f}-{df['se_inflation'].max():.1f}) "
          f"-- passage-level CIs are this much too narrow")

    for direction, label in ((1, "CONSERVATIVE-leaning shows discuss MORE"),
                             (-1, "LIBERAL-leaning shows discuss MORE")):
        sub = df[np.sign(df["est_show"]) == direction].copy()
        sub = sub.sort_values("est_show", key=abs, ascending=False)
        for reg, rlab in ((False, "SUBSTANTIVE CONTENT"), (True, "REGISTER / FILLER")):
            s = sub[sub["register"] == reg].head(top)
            if s.empty:
                continue
            print(f"\n--- {label}  [{rlab}] ---")
            print(f"{'T':>4} {'est(show)':>10} {'95% CI (show)':>22} "
                  f"{'p':>9} {'bonf':>5} {'prop':>6}  words")
            for _, r in s.iterrows():
                ci = f"[{r['ci_low_show']:+.4f},{r['ci_high_show']:+.4f}]"
                print(f"{int(r['topic']):>4} {r['est_show']:>+10.4f} {ci:>22} "
                      f"{r['p_show']:>9.2e} {'YES' if r['survives_bonferroni_clustered'] else '  -':>5} "
                      f"{r['prop']:>6.3f}  {short(r['top_words'])}")
    n_reg = int(df["register"].sum())
    n_sig = int(df["survives_bonferroni_clustered"].sum())
    print(f"\n[q1] {n_reg}/{len(df)} topics flagged register; "
          f"{n_sig}/{len(df)} clustered effects survive Bonferroni")
    df.to_csv(OUT_DIR / f"stm_q1_summary_k{k}.csv", index=False)
    return df


def q2(k: int, topics: list[int] | None, top: int) -> None:
    p = OUT_DIR / f"stm_content_words_k{k}.csv"
    if not p.exists():
        print(f"\n[q2] {p.name} not present yet")
        return
    w = pd.read_csv(p)
    qp = OUT_DIR / f"stm_content_quality_k{k}.csv"
    qual = pd.read_csv(qp) if qp.exists() else None

    print(f"\n{'=' * 100}\nQ2 -- CONTENT: do liberal and conservative shows discuss the SAME topic "
          f"differently? (K={k})\n{'=' * 100}")
    print("Words are SAGE deviations for that ideology tercile within the topic --\n"
          "i.e. how each group words this topic relative to the shared baseline.\n")

    piv = w.pivot(index="topic", columns="level", values="words")
    reg = w[w["level"] == "MARGINAL"].set_index("topic")["register"]
    chosen = topics if topics else (
        qual.sort_values("prop", ascending=False)["topic"].tolist()
        if qual is not None else piv.index.tolist())
    shown = 0
    for t in chosen:
        if t not in piv.index:
            continue
        if bool(reg.get(t, False)):
            continue                      # register topics reported separately
        print(f"--- Topic {t}"
              + (f"  (prop {qual.loc[qual.topic == t, 'prop'].iloc[0]:.3f})" if qual is not None else "")
              + " ---")
        print(f"  baseline    : {short(piv.loc[t, 'MARGINAL'], top)}")
        for lvl in ("Liberal", "Moderate", "Conservative"):
            if lvl in piv.columns:
                print(f"  {lvl:<12}: {short(piv.loc[t, lvl], top)}")
        print()
        shown += 1
        if shown >= 10:
            break

    rt = [t for t in piv.index if bool(reg.get(t, False))]
    print(f"[q2] register-flagged topics excluded from the list above: {rt}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=75)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--topics", type=int, nargs="*", default=None)
    a = ap.parse_args()
    q1(a.k, a.top)
    q2(a.k, a.topics, 10)


if __name__ == "__main__":
    main()
