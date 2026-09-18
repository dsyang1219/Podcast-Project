"""Within-show event study: does listener-directed address shift around eight political shocks 2020-2025? Placebo dates included. Paper finding 1 (register is a stable trait)."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2024)
# Episode air dates: keep only rows with a full YYYY-MM-DD string that parses to a real date.
E = pd.read_csv(INPUTS / "ep_dates.csv")
E = E[E.date.astype(str).str.len() == 10].copy()
E["dt"] = pd.to_datetime(E.date, errors="coerce")
E = E[E.dt.notna()]
E["show_id"] = E.show_id.astype(str)
# Show-level ideology labels (LLM lean/side, host DIME score) so shifts can be split by side.
S = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "lean", "side", "avg_host_cfscore"]]
S["show_id"] = S.show_id.astype(str)
# Passage-level register measures from the remeasured corpus: word count (nw) plus the three address
# components (syntactic imperatives, dictionary imperatives, second-person pronouns), all per 10,000 words.
d = pd.read_csv(
    INPUTS / "remeasured.csv", usecols=["episode_id", "show_id", "nw", "imper_syn", "imper_rx", "you_rx"]
)
d["show_id"] = d.show_id.astype(str)
# Collapse passages to episodes: word-count-weighted mean of each component; nw becomes the episode's total words.
ep = (
    d.groupby(["episode_id", "show_id"])
    .apply(
        lambda g: pd.Series(
            {c: np.average(g[c], weights=g.nw) for c in ("imper_syn", "imper_rx", "you_rx")}
            | {"nw": g.nw.sum()}
        ),
        include_groups=False,
    )
    .reset_index()
)
# Episode-level listener-directed address (lda): z-score each component across all dated episodes and average,
# so one unit = one SD of episode-level address in this corpus (the paper's dir_z is the show-level analogue).
z = lambda s: (s - s.mean()) / s.std(ddof=0)
ep["lda"] = (
    z(ep.imper_syn) + z(ep.imper_rx) + z(ep.you_rx)
) / 3  # episode-level listener-directed address, corpus-standardised
# Attach air dates and show labels; t = days since 2018-01-01 (a numeric clock for trend checks).
ep = ep.merge(E[["episode_id", "dt"]], on="episode_id").merge(S, on="show_id")
ep["t"] = (ep.dt - pd.Timestamp("2018-01-01")).dt.days
print(
    f"{len(ep):,} dated episodes with register, {ep.show_id.nunique()} shows, {ep.dt.min().date()} -> {ep.dt.max().date()}\n"
)


# Episodes within +/- `days` of an event date; post = on/after the event, rel = days relative to it.
def window(ev, days=60):
    e0 = pd.Timestamp(ev)
    w = ep[(ep.dt >= e0 - pd.Timedelta(days=days)) & (ep.dt <= e0 + pd.Timedelta(days=days))].copy()
    w["post"] = (w.dt >= e0).astype(int)
    w["rel"] = (w.dt - e0).dt.days
    return w


# Post-minus-pre shift in lda with show fixed effects (C(show_id)), so each show is compared only with itself;
# SEs clustered by show. by_side=True adds post x lean: returns the L shift, the extra R shift, and its p-value.
def post_coef(w, by_side=False):
    if by_side:
        m = smf.ols("lda ~ post*C(lean) + C(show_id)", data=w).fit(
            cov_type="cluster", cov_kwds={"groups": w.show_id}
        )
        k = [x for x in m.params.index if x.startswith("post:")][0]
        return m.params["post"], m.params[k], m.pvalues[k], m
    m = smf.ols("lda ~ post + C(show_id)", data=w).fit(
        cov_type="cluster", cov_kwds={"groups": w.show_id}
    )
    return m.params["post"], m.pvalues["post"], m


# The eight political shocks (date, label); windows are +/-60 days unless stated otherwise.
EVENTS = [
    ("2024-11-05", "2024 general election"),
    ("2025-01-20", "2025 inauguration"),
    ("2024-07-13", "Trump assassination attempt"),
    ("2024-07-21", "Biden withdraws"),
    ("2022-11-08", "2022 midterm"),
    ("2020-11-03", "2020 general election"),
    ("2021-01-06", "January 6"),
    ("2022-06-24", "Dobbs"),
]
print(
    "=== A. Within-show shift in listener-directed address, ±60 days (post - pre), clustered by show ==="
)
print(
    f"{'event':<30}{'episodes':>9}{'shows':>7}{'post b (SD)':>12}{'p':>8}   {'L shift':>8}{'R shift':>8}{'post×R p':>10}"
)
# --- A. For each shock: pooled within-show post shift, then the shift by side (R = L shift + interaction).
res = {}
for ev, lab in EVENTS:
    w = window(ev)
    if len(w) < 150:  # a show-FE model needs a reasonably populated window
        print(f"{lab:<30}{len(w):>9}  too few")
        continue
    b, p, _ = post_coef(w)
    bL, bint, pint, m = post_coef(w, by_side=True)
    res[ev] = (w, b)  # keep the window and pooled shift (reused by the placebo test in B)
    print(
        f"{lab:<30}{len(w):>9}{w.show_id.nunique():>7}{b:>+12.3f}{p:>8.4f}   {bL:>+8.3f}{bL+bint:>+8.3f}{pint:>10.4f}"
    )

print(
    "\n=== B. Placebo-date inference for the 2024 election: same ±60d design at 60 random non-election dates (2019-2026) ==="
)
# --- B. Placebo test: rerun the identical design at 60 random dates that are not near any real shock,
# and see where the real 2024-election shift falls in that distribution.
w, b_real = res["2024-11-05"]
real = [pd.Timestamp(x) for x, _ in EVENTS]
plac = []
cands = pd.date_range("2019-03-01", "2026-05-01", freq="D")  # candidate placebo dates spanning the dated corpus
while len(plac) < 60:
    c = cands[rng.integers(len(cands))]
    if any(abs((c - r).days) < 90 for r in real):  # skip dates within 90 days of any real shock
        continue
    pw = window(c.strftime("%Y-%m-%d"))
    if len(pw) < 150 or pw.post.nunique() < 2:  # same size floor as A, and both pre and post episodes must exist
        continue
    try:
        plac.append(post_coef(pw)[0])
    except Exception:
        pass
plac = np.array(plac)
# Two-sided placebo p = share of placebo shifts at least as large in absolute value as the real one;
# rank = share of placebo shifts below the real one.
print(
    f"  real post-shift = {b_real:+.3f} SD | placebo distribution mean {plac.mean():+.3f}, sd {plac.std():.3f} | two-sided placebo p = {(np.abs(plac)>=abs(b_real)).mean():.3f} | rank {(plac<b_real).mean():.2f}"
)

# --- C. Week-by-week profile around the 2024 election, by side, after subtracting each show's own mean
# (demeaning within show = the show fixed effect done by hand).
print(
    "\n=== C. Event-time profile, 2024 election: weekly within-show mean of lda (show FE), by side ==="
)
w = window("2024-11-05", 70).copy()
w["wk"] = (w.rel // 7).clip(-10, 9)  # week index relative to election day, capped to -10..9
w["lda_dm"] = w.lda - w.groupby("show_id").lda.transform("mean")  # demean lda within show
prof = w.groupby(["wk", "lean"]).lda_dm.mean().unstack()
prof["n"] = w.groupby("wk").size()
print(prof.round(3).to_string())
# Pre-trend check: straight-line slope of the weekly L and R means over the 10 pre-election weeks;
# a flat slope means address was not already drifting before the event.
print(
    "\n  pre-trend check (weeks -10..-1): slope of weekly L / R means:",
    f"L {np.polyfit(prof.index[prof.index<0],prof.loc[prof.index<0,'L'],1)[0]:+.4f}/wk, R {np.polyfit(prof.index[prof.index<0],prof.loc[prof.index<0,'R'],1)[0]:+.4f}/wk",
)
# --- D. Mobilisation check for three elections: the final 4 weeks before election day vs the 4 weeks before that.
# If hosts rally listeners as the vote approaches, final4 should be positive.
print(
    "\n=== D. Run-up: is address elevated in the final 4 weeks BEFORE the election vs the 4 weeks before that? (mobilisation) ==="
)
for ev, lab in (("2024-11-05", "2024"), ("2022-11-08", "2022"), ("2020-11-03", "2020")):
    e0 = pd.Timestamp(ev)
    a = ep[(ep.dt >= e0 - pd.Timedelta(days=56)) & (ep.dt < e0)].copy()
    a["final4"] = (a.dt >= e0 - pd.Timedelta(days=28)).astype(int)  # 1 = final 28 days before election day, 0 = the 4 weeks before that
    m = smf.ols("lda ~ final4 + C(show_id)", data=a).fit(
        cov_type="cluster", cov_kwds={"groups": a.show_id}
    )
    m2 = smf.ols("lda ~ final4*C(lean) + C(show_id)", data=a).fit(
        cov_type="cluster", cov_kwds={"groups": a.show_id}
    )
    k = [x for x in m2.params.index if x.startswith("final4:")][0]  # the final4 x lean interaction (extra shift for R shows)
    print(
        f"  {lab}: final-4-weeks shift {m.params['final4']:+.3f} SD (p={m.pvalues['final4']:.4f}, n={len(a)}) | L {m2.params['final4']:+.3f}, R {m2.params['final4']+m2.params[k]:+.3f}, diff p={m2.pvalues[k]:.3f}"
    )
ep.to_pickle(INPUTS / "ep_lda.pkl")  # episode-level lda saved for downstream scripts
