"""The validation triangle at the listener level: label vs DIME vs audience party, all 59 shows, listener-weighted; plus a Pew cross-check. Paper finding 1."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, scipy.stats as st, statsmodels.api as sm, warnings

warnings.filterwarnings("ignore")
rng = np.random.default_rng(71)  # fixed seed for the bootstrap
# Expanded respondent-show matches joined to the respondent's party id and to the show's LLM side label and DIME host score.
K = pd.read_pickle(INPUTS / "kett.pkl")
M = pd.read_csv(INPUTS / "match_expanded.csv").drop(columns=["show"], errors="ignore")
S = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "show", "side", "avg_host_cfscore"]]
J = M.merge(K[["resp", "pid"]], on="resp").merge(S, on="show_id")
J["lean"] = J.pid.isin(["R", "leanR"]).astype(float) - J.pid.isin(["D", "leanD"]).astype(float)  # listener lean: +1 Republican/lean R, -1 Democrat/lean D, 0 otherwise
# Show-level table: listeners, mean audience lean, LLM side, DIME score.
A = (
    J.groupby("show_id")
    .agg(
        n=("resp", "nunique"),
        aud=("lean", "mean"),
        side=("side", "first"),
        cf=("avg_host_cfscore", "first"),
    )
    .reset_index()
)


# Pearson r with a 95% percentile bootstrap CI, resampling SHOWS (not listeners) with replacement.
def bootr(a, x, y, B=5000, w=None):
    b = a.dropna(subset=[x, y]).reset_index(drop=True)
    obs = st.pearsonr(b[x], b[y])[0]
    rs = []
    for _ in range(B):
        idx = rng.integers(0, len(b), len(b))  # draw shows with replacement
        s = b.iloc[idx]
        if s[x].std() == 0 or s[y].std() == 0:  # skip degenerate draws where a variable is constant
            continue
        rs.append(np.corrcoef(s[x], s[y])[0, 1])
    return obs, np.percentile(rs, [2.5, 97.5]), len(b)


# The triangle at two audience-size thresholds (>=5 and >=10 listeners), for each pair of the three ideology measures.
print("=== TRIANGLE with 95% bootstrap CIs (resampling SHOWS) ===")
for k in (5, 10):
    a = A[A.n >= k]
    for x, y, lab in (
        ("aud", "side", "audience lean vs LLM label"),
        ("aud", "cf", "audience lean vs DIME"),
        ("side", "cf", "LLM label vs DIME"),
    ):
        r, ci, n = bootr(a, x, y)
        print(
            f"  >={k:2d} listeners  {lab:<28} r={r:+.3f}  95% CI [{ci[0]:+.3f},{ci[1]:+.3f}]  n={n} shows"
        )


# listener-weighted, all 59 shows (no threshold): weight each show by sqrt(n)
def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)  # weighted means
    c = np.average((x - mx) * (y - my), weights=w)
    return c / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))  # weighted covariance / product of weighted SDs


# No threshold: use every show but down-weight tiny audiences by sqrt(n).
a = A.copy()
print(
    f"\n  ALL {len(a)} shows, sqrt(n)-weighted: audience lean vs LLM label r={wcorr(a.aud,a.side,np.sqrt(a.n)):+.3f}"
)
b = a.dropna(subset=["cf"])
print(
    f"  ALL {len(b)} shows with DIME, sqrt(n)-weighted: audience lean vs DIME r={wcorr(b.aud,b.cf,np.sqrt(b.n)):+.3f}"
)
# listener-level: does the label predict an individual's party? (the strongest form: n=listeners, clustered)
X = sm.add_constant(J[["side"]])
m = pd.concat([X, J[["lean", "show_id"]]], axis=1).dropna()
r = sm.OLS(m.lean, m[["const", "side"]]).fit(cov_type="cluster", cov_kwds={"groups": m.show_id})  # clustered by show: listeners of one show share its label
print(
    f"\n  LISTENER-LEVEL: listener partisan lean ~ show LLM label, {len(m)} listeners / {m.show_id.nunique()} shows, clustered: b={r.params['side']:+.3f} p={r.pvalues['side']:.2e}; R2={r.rsquared:.3f}"
)
# classification accuracy: predict majority party of audience from label
a = A[A.n >= 5]
acc = ((a.side > 0) == (a.aud > 0)).mean()  # side > 0 = right label; aud > 0 = majority-Republican audience
print(
    f"  label predicts majority-party of audience correctly in {acc:.0%} of {len(a)} shows (>=5 listeners)"
)

print(
    "\n=== PEW MECHANISM ITEMS — within party / within age (podcast-news freq -> parasocial connection) ==="
)
# Pew American Trends Panel Wave 150 (July 2024): the news-influencer module.
w = pd.read_csv(DATA / "external/pew_W150_Jul24/W150_Jul24/ATP W150.csv", low_memory=False)


# Pew items as numbers; codes >= 90 are refused/skipped.
def cl(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s > 0) & (s < 90))


# Respondents who get news from influencers (NEWSINF = 1). pod = how often they get news via podcasts; y = feel connected to the influencer; party, age band, education and internet-use frequency as controls.
inf = w[cl(w.NEWSINF_W150) == 1].copy()
inf["pod"] = cl(inf.NEWSPLAT_DIG_d_W150)  # podcast news frequency (the predictor)
inf["y"] = (cl(inf.NEWSINFCONNECT_W150) == 1).astype(float).where(cl(inf.NEWSINFCONNECT_W150).notna())  # 1 = feels a personal connection; missing stays missing
inf["party"] = cl(inf.F_PARTYSUMIDEO_FINAL)
inf["age"] = cl(inf.F_AGECAT)
inf["edu"] = cl(inf.F_EDUCCAT)
inf["attn"] = cl(inf.F_INTFREQ)


# Survey-weighted linear probability model of connection on podcast-news frequency, within a subgroup.
def fit(g, lab):
    X = pd.DataFrame({"age": g.age, "edu": g.edu, "attn": g.attn, "pod": g.pod})
    X = sm.add_constant(X, has_constant="add")
    m = pd.concat([X, g.y, g.WEIGHT_W150], axis=1).dropna()
    r = sm.WLS(m.y, m[X.columns], weights=m.WEIGHT_W150).fit(cov_type="HC1")
    print(f"  {lab:<28} n={len(m):4d}  b={r.params['pod']:+.4f}/level  p={r.pvalues['pod']:.4f}")


# Run within party and within age band so the association is not a party or age artefact.
for lab, mask in (
    ("Rep/lean Rep (1-2)", inf.party.isin([1, 2])),
    ("Dem/lean Dem (3-4)", inf.party.isin([3, 4])),
    ("age 18-29", inf.age == 1),
    ("age 30-49", inf.age == 2),
    ("age 50+", inf.age >= 3),
):
    fit(inf[mask], lab)
