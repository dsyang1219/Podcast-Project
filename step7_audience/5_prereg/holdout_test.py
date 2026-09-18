"""CONFIRMATORY out-of-frame generalization test, executing handoff/prereg/prereg_holdout.md and its two amendments exactly as frozen: tests (i)-(viii), secondary outcome, H2, robustness and leave-one-show-out. Paper finding 7 and the disclosure section."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS

"""CONFIRMATORY out-of-frame generalization test — executes handoff/prereg/prereg_holdout.md (+Amendments 1-2) as frozen.
Primary (i), required robustness (ii), sensitivities (iii)-(viii), secondary Q33E, H2 exclusivity, reported-regardless tables."""
import pandas as pd, numpy as np, statsmodels.api as sm, re, warnings, hashlib, datetime, os, sys
from scipy import stats, linalg

warnings.filterwarnings("ignore")
# Fixed seed: the wild-bootstrap and permutation draws below are reproducible run to run.
rng = np.random.default_rng(5)
# Work inside inputs/, and fingerprint the frozen prereg text, alias table and show scores so the run log records exactly which versions were used.
SP = str(INPUTS)
ROOT = str(ROOT)
os.chdir(SP)
H = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]  # short SHA-256 fingerprint of a file
print(
    f"RUN {datetime.datetime.utcnow().isoformat()}Z | prereg {H(ROOT+'/handoff/prereg/prereg_holdout.md')} | alias {H(ROOT+'/handoff/prereg/holdout_alias_table_FROZEN.csv')} | scores {H(SP+'/holdout_show_dirz_FINAL.csv')}"
)
# ---------------- exposure (show level) ----------------
# Show-level exposure for the 10 out-of-frame shows: dir_z (and the declarative-you variant dir_z_declar, Amendment 2 test viii), scored against the 194-show reference scale.
S = pd.read_csv(INPUTS / "holdout_show_dirz_FINAL.csv")
# Human-readable show names; the frozen alias table is keyed on these.
NAME = {
    "HOLDOUT_rogan": "The Joe Rogan Experience",
    "HOLDOUT_shapiro": "The Ben Shapiro Show",
    "HOLDOUT_thedaily": "The Daily (NYT)",
    "HOLDOUT_tucker": "The Tucker Carlson Show",
    "HOLDOUT_beck": "The Glenn Beck Program",
    "HOLDOUT_kirk": "The Charlie Kirk Show",
    "HOLDOUT_parnas": "The Parnas Perspective",
    "HOLDOUT_kelly": "The Megyn Kelly Show",
    "HOLDOUT_ryan": "The Shawn Ryan Show",
    "HOLDOUT_bongino": "The Dan Bongino Show",
}
# Show ideology control (prereg 'Model and inference': LLM side label). Rogan is coded 0.5 = neither side; a robustness run below recodes him right.
RIGHT = {
    "HOLDOUT_shapiro": 1,
    "HOLDOUT_tucker": 1,
    "HOLDOUT_beck": 1,
    "HOLDOUT_kirk": 1,
    "HOLDOUT_kelly": 1,
    "HOLDOUT_ryan": 1,
    "HOLDOUT_bongino": 1,
    "HOLDOUT_thedaily": 0,
    "HOLDOUT_parnas": 0,
    "HOLDOUT_rogan": 0.5,
}
# Amendment 1 (test v): INTERVIEW_DOMINANT = long-form host+guest interview shows, where 'you' is often aimed at the guest rather than the listener.
INTERVIEW = {
    "HOLDOUT_rogan": 1,
    "HOLDOUT_tucker": 1,
    "HOLDOUT_ryan": 1,
}  # Amendment 1, coded from public knowledge
S["show"] = S.show_id.map(NAME)
S["share_right"] = S.show_id.map(RIGHT)
S["interview"] = S.show_id.map(INTERVIEW).fillna(0)  # every show not listed is monologue/co-host = 0
S = S[S.episodes >= 25]  # prereg 'Episode sample': a show needs >= 25 usable episodes to be retained
print(f"shows retained (>=25 eps): {len(S)}")
# ---------------- respondents ----------------
# Respondent-show pairs from the FROZEN alias table (prereg 'Respondent match'), joined to the show scores; the inner merge drops any show that failed the episode cutoff.
A = pd.read_csv(f"{ROOT}/handoff/prereg/holdout_alias_table_FROZEN.csv").merge(
    S[["show", "show_id", "dir_z", "dir_z_declar", "share_right", "interview", "fmt_pos"]],
    on="show",
    how="inner",
)
# Flags for the two least certain mappings, needed for the pre-registered mapping robustness checks:
# wrote 'Blaze' but not 'Beck'/'Glenn': mapped to Beck only via his network
A["blaze_only"] = A.verbatim.str.contains("blaze", case=False) & ~A.verbatim.str.contains(
    "beck|glenn", case=False
)
# wrote 'Daily Wire' but not 'Shapiro'/'Ben': mapped to Shapiro only via his network
A["dw_only"] = A.verbatim.str.contains(r"daily ?wire", case=False) & ~A.verbatim.str.contains(
    "shapiro|ben ", case=False
)
# Cluster assignment: a respondent naming two holdout shows is clustered on the alphabetically first one.
prim = A.sort_values("show").groupby("resp").first().reset_index()[["resp", "show_id", "show"]]
# Collapse to one row per respondent. Exposure = mean over the shows named (prereg: 'gets the mean dir_z'). The mapping flags use 'all' so a respondent is dropped in the robustness runs only if EVERY string they gave was the ambiguous one.
R = (
    A.groupby("resp")
    .agg(
        dir_z=("dir_z", "mean"),
        dir_z_declar=("dir_z_declar", "mean"),
        share_right=("share_right", "mean"),
        interview=("interview", "mean"),
        fmt_pos=("fmt_pos", "mean"),
        n_shows=("show", "nunique"),
        blaze_only=("blaze_only", "all"),
        dw_only=("dw_only", "all"),
    )
    .reset_index()
    .merge(prim, on="resp")
)
# Kettering respondents (built by model.py) and the pre-registered outcomes.
K = pd.read_pickle(INPUTS / "kett.pkl")
num = lambda g, c: pd.to_numeric(g[c], errors="coerce").where(lambda s: s > 0)  # survey item as numbers; negative codes = missing
z = lambda s: (s - s.mean()) / s.std(ddof=0)  # z-score with population SD
# PRIMARY outcome cyn8 (prereg 'Outcome'): 8 institutional-trust items, each reversed (higher = more cynical) and z-scored on the FULL Kettering sample, averaged, re-standardised.
ITEMS = ["Q35E", "Q35C", "Q34B", "Q33H", "Q31", "Q33E", "Q35A", "Q35D"]
Z = pd.DataFrame({c: -z(num(K, c)) for c in ITEMS})
K["cyn8"] = z(Z.mean(axis=1).where(Z.notna().sum(axis=1) >= 6))  # require at least 6 of the 8 items answered
K["q33e_rev"] = z(-num(K, "Q33E"))  # SECONDARY outcome: elections administered well, reversed
# Right-wing platform index: count of Truth Social / Rumble / Parler / Gab used regularly (<= 2 on the frequency scale).
for c in ["Q20C", "Q20G", "Q20E", "Q20H"]:
    K[c] = num(K, c)
K["rw_plat"] = (K[["Q20C", "Q20G", "Q20E", "Q20H"]] <= 2).sum(axis=1).astype(float)
# H2 outcome: exclusivity = named NO mainstream outlet in the three Q17 verbatims (same regex as discovery round 8).
MAIN = r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|washington post|wall street|wsj|reuters|associated press|\bap\b|usa today|local news|newspaper|the hill|politico|axios|bloomberg|cnbc|news ?nation|newsmax"
L = pd.read_csv(INPUTS / "verbatims_long.csv")
L["main"] = L.v.str.contains(MAIN, regex=True, na=False)
E = L.groupby("resp").main.max().rename("any_main").reset_index()
E["excl"] = 1 - E.any_main.astype(float)  # 1 = no mainstream outlet named
# Analysis frame: one row per matched respondent with exposure, controls and outcomes.
g = R.merge(K, on="resp", how="left").merge(E[["resp", "excl"]], on="resp", how="left")
print(
    f"respondents: {len(g)} | pairs: {len(A)} | two-show: {(g.n_shows>1).sum()} | cyn8 non-missing: {g.cyn8.notna().sum()}"
)
# ---------------- reported regardless: per-show table ----------------
# Prereg 'What will be reported regardless': per-show n, dir_z, audience partisan lean and mean outcomes, as a check on the show ideology label.
g["isR"] = g.pid.isin(["R", "leanR"]).astype(float)
g["isD"] = g.pid.isin(["D", "leanD"]).astype(float)
T = (
    g.groupby("show")
    .agg(
        n=("resp", "size"),
        dir_z=("dir_z", "first"),
        shareR=("isR", "mean"),
        shareD=("isD", "mean"),
        cyn8=("cyn8", "mean"),
        excl=("excl", "mean"),
    )
    .sort_values("dir_z", ascending=False)
)
T["lean"] = T.shareR - T.shareD  # audience lean = Republican share minus Democratic share
print(
    "\n=== PER-SHOW (reported regardless): dir_z, audience partisan lean (R−D share), mean cynicism, exclusivity ==="
)
print(T[["n", "dir_z", "shareR", "shareD", "lean", "cyn8", "excl"]].round(2).to_string())
print(f"show-level corr(dir_z, audience lean) = {np.corrcoef(T.dir_z,T.lean)[0,1]:+.2f}  (n=10)")


# Inference engine: same maths as common.cr_all, plus one-sided p-values and a CR2 confidence interval.
# ---------------- inference ----------------
# OLS of y on X; returns coefficient j with CR2 (Bell-McCaffrey, Satterthwaite df) and wild-bootstrap p-values, clustered by g_.
def cr_all(y, X, g_, j, B=2000):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    g_ = np.asarray(g_)
    N, K_ = X.shape
    # Plain OLS by hand.
    M = np.linalg.inv(X.T @ X)
    b = M @ X.T @ y
    e = y - X @ b
    # One index array per show (cluster).
    groups = [np.where(g_ == u)[0] for u in np.unique(g_)]
    G = len(groups)
    # CR1 sandwich (conventional cluster-robust SE); its t is the reference statistic for the bootstrap below.
    Sm = sum((X[i].T @ e[i])[:, None] @ (X[i].T @ e[i])[None, :] for i in groups)
    V1 = (G / (G - 1)) * ((N - 1) / (N - K_)) * M @ Sm @ M
    se1 = np.sqrt(V1[j, j])
    t1 = b[j] / se1
    # CR2: rescale each cluster's residuals by (I - H_gg)^(-1/2) so the sandwich is unbiased under homoskedasticity; D collects the pieces needed for the Satterthwaite df.
    Hm = X @ M @ X.T
    IH = np.eye(N) - Hm
    c = np.zeros(K_)
    c[j] = 1
    S2 = np.zeros((K_, K_))
    D = np.zeros((N, G))
    for k, i in enumerate(groups):
        Ad = linalg.fractional_matrix_power(np.eye(len(i)) - Hm[np.ix_(i, i)], -0.5).real  # per-cluster Bell-McCaffrey adjustment matrix
        Xa = Ad @ X[i]
        S2 += Xa.T @ np.outer(e[i], e[i]) @ Xa
        D[i, k] = Ad @ X[i] @ M @ c
    V2 = M @ S2 @ M
    se2 = np.sqrt(V2[j, j])
    t2 = b[j] / se2
    # Satterthwaite df = trace(W)^2 / trace(W^2): the effective number of clusters (prereg expects ~3-4 here because Rogan dominates).
    Wm = D.T @ IH @ D
    df = np.trace(Wm) ** 2 / np.trace(Wm @ Wm)
    p2_two = 2 * stats.t.sf(abs(t2), df)
    p2_one = stats.t.sf(t2, df)  # one-sided in the predicted (positive) direction
    # Wild cluster bootstrap under the null: refit WITHOUT column j, then flip each show's residuals with one Rademacher coin per show, B=2000 times (prereg: 2000 reps).
    X0 = np.delete(X, j, axis=1)
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    f0 = X0 @ b0
    r0 = y - f0
    tb = np.empty(B)
    for s in range(B):
        yb = f0.copy()
        for i in groups:
            yb[i] += r0[i] * rng.choice([-1.0, 1.0])  # one +1/-1 coin per show
        bb = M @ X.T @ yb
        eb = yb - X @ bb
        Sb = sum((X[i].T @ eb[i])[:, None] @ (X[i].T @ eb[i])[None, :] for i in groups)
        Vb = (G / (G - 1)) * ((N - 1) / (N - K_)) * M @ Sb @ M
        tb[s] = bb[j] / np.sqrt(Vb[j, j])
    # Bootstrap p = share of null t-statistics at least as extreme as the observed CR1 t (two-sided and one-sided).
    pw_two = (np.abs(tb) >= abs(t1)).mean()
    pw_one = (tb >= t1).mean()
    tc = stats.t.ppf(0.975, df)  # critical t for a 95% CI on the Satterthwaite df
    return dict(
        b=b[j],
        se2=se2,
        lo=b[j] - tc * se2,
        hi=b[j] + tc * se2,
        df=df,
        p2_one=p2_one,
        p2_two=p2_two,
        pw_one=pw_one,
        pw_two=pw_two,
        G=G,
        N=N,
    )


# Control matrix as pre-registered: party id dummies (5 categories, one dropped), attention, age, education, show ideology, right-wing platform index, plus any sensitivity covariate.
def design(d, extra=()):
    X = pd.get_dummies(d["pid"], prefix="pid", drop_first=True).astype(float)
    X["attn"] = d.attn
    X["age"] = d.age
    X["edu"] = d.edu
    X["share_right"] = d.share_right
    X["rw_plat"] = d.rw_plat
    for e in extra:
        X[e] = d[e]
    return sm.add_constant(X, has_constant="add")


# Every fitted model is appended here and written to outputs/holdout_test_results.csv at the end.
ROWS = []


# Fit one pre-registered model and print the standard line: coefficient, CR2 95% CI, Satterthwaite df, one- and two-sided CR2 and wild p, PASS/fail.
def run(label, d, y="cyn8", expo="dir_z", extra=(), w=None, B=2000):
    X = design(d, extra)
    X["dir_z"] = d[expo].values  # column keeps the name dir_z even when the exposure is the declarative-you variant
    m = pd.concat([X, d[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    Xm = m[X.columns].values
    ym = m[y].values
    # Optional per-show weights, applied as sqrt(w) row scaling (= weighted least squares).
    if w is not None:
        ww = np.sqrt(m.show_id.map(w).values)
        Xm = Xm * ww[:, None]
        ym = ym * ww
    r = cr_all(ym, Xm, m.show_id.values, list(X.columns).index("dir_z"), B=B)
    r["label"] = label
    ROWS.append(r)
    flag = "PASS" if (r["b"] > 0 and r["p2_one"] < 0.05 and r["pw_one"] < 0.05) else "fail"  # prereg success rule: positive AND one-sided CR2 p < .05 AND wild p < .05
    print(
        f"{label:<58} b={r['b']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] CR2 df={r['df']:.1f} p1={r['p2_one']:.3f} (2s {r['p2_two']:.3f}) | wild p1={r['pw_one']:.3f} (2s {r['pw_two']:.3f}) | G={r['G']} N={r['N']}  {flag}",
        flush=True,
    )
    return r


# Test (iv): show-level test. Residualise the outcome on the controls, average residuals by show, regress the 10 show means on dir_z weighted by listener count, and compare against shuffles of dir_z across shows.
def showperm(d, y="cyn8", extra=(), B=4000):
    X = design(d, extra)
    m = pd.concat([X, d[[y, "dir_z", "show_id"]]], axis=1).dropna().reset_index(drop=True)
    m["r"] = sm.OLS(m[y], m[X.columns]).fit().resid
    a = (
        m.groupby("show_id")
        .agg(dir_z=("dir_z", "first"), r=("r", "mean"), n=("r", "size"))
        .reset_index()
    )
    fitb = lambda x: sm.WLS(a.r.values, sm.add_constant(x), weights=a.n.values).fit().params[1]
    b = fitb(a.dir_z.values)
    null = np.array([fitb(rng.permutation(a.dir_z.values)) for _ in range(B)])  # null: shuffle dir_z across shows
    return b, (null >= b).mean(), (np.abs(null) >= abs(b)).mean(), len(a)  # (b, one-sided p, two-sided p, number of shows)


# Frame for test (ii): the required robustness drops Rogan, the single cluster holding about half the sample.
noR = g[g.show_id != "HOLDOUT_rogan"]
# ---- H1, primary outcome: tests (i)-(iv) as frozen, (v)-(viii) from the amendments. ----
print("\n=== H1 PRIMARY OUTCOME: 8-item institutional cynicism (z on full Kettering sample) ===")
i_ = run("(i)   PRIMARY: respondent-level, CR2 + wild", g)
ii_ = run("(ii)  REQUIRED: Rogan excluded", noR)
# Test (iii): cap each show's weight so no show exceeds 20% of the sample (weight < 1 only for shows above the cap).
ns = g.show_id.value_counts()
cap = {s: min(1.0, 0.2 * len(g) / n) for s, n in ns.items()}
run("(iii) SENS: per-show weights capped at 20% of sample", g, w=cap)
b, p1, p2, n = showperm(g)
print(
    f"(iv)  show-level precision-weighted permutation: b={b:+.3f} one-sided p={p1:.3f} two-sided p={p2:.3f} (shows={n})"
)
# Test (v), Amendment 1: hold interview format constant with the hand-coded indicator, with and without Rogan.
run("(v)   SENS [Amend.1]: + INTERVIEW_DOMINANT", g, extra=("interview",))
run("(v)   SENS [Amend.1]: + INTERVIEW_DOMINANT, Rogan excluded", noR, extra=("interview",))
# Test (vii), Amendment 2: the data-derived, position-based format score in place of the hand-coded indicator.
run("(vii) SENS [Amend.2]: + position-based format score", g, extra=("fmt_pos",))
# Test (viii), Amendment 2: exposure rebuilt from declarative, non-'you know', non-PERSON second person.
run("(viii)SENS [Amend.2]: declarative-you exposure", g, expo="dir_z_declar")
run("(viii)SENS [Amend.2]: declarative-you exposure, Rogan excl.", noR, expo="dir_z_declar")
# Verdict as pre-registered: H1 needs BOTH (i) and (ii) to pass.
print(
    f"\nVERDICT H1 (requires (i) AND (ii), one-sided CR2<.05 AND wild<.05): (i) {'pass' if i_['p2_one']<.05 and i_['pw_one']<.05 and i_['b']>0 else 'FAIL'}; (ii) {'pass' if ii_['p2_one']<.05 and ii_['pw_one']<.05 and ii_['b']>0 else 'FAIL'}"
)
# Secondary outcome (Q33E reversed) and H2 (exclusivity, linear probability model), each with and without Rogan.
print("\n=== SECONDARY: Q33E alone (elections administered well, reversed) ===")
run("(i)   secondary Q33E", g, y="q33e_rev")
run("(ii)  secondary Q33E, Rogan excluded", noR, y="q33e_rev")
print("\n=== H2: exclusivity (no mainstream outlet named), LPM ===")
run("(i)   H2 exclusivity", g, y="excl")
run("(ii)  H2 exclusivity, Rogan excluded", noR, y="excl")
# Reported robustness runs. First the survey-weighted version of the primary: WEIGHT applied as sqrt scaling, as in run().
print("\n=== REPORTED ROBUSTNESS ===")
gw = g.copy()  # (unused leftover)
wt = {s: 1.0 for s in g.show_id.unique()}  # (unused leftover)
Xw = design(g)
Xw["dir_z"] = g.dir_z
mw = pd.concat([Xw, g[["cyn8", "show_id", "WEIGHT"]]], axis=1).dropna().reset_index(drop=True)
rw = cr_all(
    mw.cyn8.values * np.sqrt(mw.WEIGHT.values),
    mw[Xw.columns].values * np.sqrt(mw.WEIGHT.values)[:, None],
    mw.show_id.values,
    list(Xw.columns).index("dir_z"),
)
print(
    f"{'survey-weighted (WEIGHT) primary':<58} b={rw['b']:+.3f} CR2 p1={rw['p2_one']:.3f} wild p1={rw['pw_one']:.3f} df={rw['df']:.1f}"
)
# Mapping checks pre-registered in 'Respondent match': drop respondents matched only through the Blaze / Daily Wire network strings.
run("drop Blaze-only strings (Beck mapping check)", g[~g.blaze_only], B=1000)
run("drop Daily-Wire-only strings (Shapiro mapping check)", g[~g.dw_only], B=1000)
# Ideology-label check: recode Rogan from 0.5 to right-leaning.
g2 = g.copy()
g2.loc[g2.show_id == "HOLDOUT_rogan", "share_right"] = 1.0
run("Rogan coded right (share_right=1)", g2, B=1000)
# Leave-one-show-out on the primary (B=500 bootstrap reps to keep runtime down).
print("\n--- leave-one-show-out (primary) ---")
for s in sorted(g.show_id.unique()):
    run(f"  drop {s.replace('HOLDOUT_',''):<10}", g[g.show_id != s], B=500)
# Save every fitted row.
pd.DataFrame(ROWS).to_csv(OUTPUTS / "holdout_test_results.csv", index=False)
# ---------------- (vi) discovery sample with the format indicator ----------------
# Test (vi), Amendment 1: apply a format indicator to the DISCOVERY sample. The corpus has no interview-dominant shows, so the proxy is the multi-host (non-solo) flag; outcome is the 4-item inst_cyn used in discovery.
print(
    "\n=== (vi) DISCOVERY sample: does a multi-host/format indicator change the discovery coefficient? (strict, 4-item inst_cyn as in discovery) ==="
)
# common.py is imported only here because it loads the discovery frames, which the confirmatory test above must not touch.
from common import st, clus, wild

DF = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "solo", "n_hosts"]]
st = st.merge(DF, on="show_id", how="left")
st["multi"] = (st.solo == 0).astype(float)  # multi-host = not solo
# Discovery coefficient with and without the format proxy: CR1 p and wild-bootstrap p.
for extra, lab in (
    (("share_right", "rw_plat"), "discovery as reported"),
    (("share_right", "rw_plat", "multi"), "+ multi-host indicator (corpus proxy for interview format)"),
):
    b_, p_, n_ = clus(st, "inst_cyn", extra)
    pw_ = wild(st, "inst_cyn", extra, B=1000)
    print(f"  {lab:<60} b={b_:+.3f} CR1 p={p_:.3f} wild p={pw_:.3f} n={n_}")
print("\nDONE")
