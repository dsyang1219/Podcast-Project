"""CONFIRMATORY within-frame test on 150 fresh corpus-show listeners, executing handoff/prereg/prereg_withinframe.md as frozen. Disclosure section."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS

"""CONFIRMATORY out-of-frame generalization test — executes handoff/prereg/prereg_holdout.md (+Amendments 1-2) as frozen.
Primary (i), required robustness (ii), sensitivities (iii)-(viii), secondary Q33E, H2 exclusivity, reported-regardless tables."""
import pandas as pd, numpy as np, statsmodels.api as sm, re, warnings, hashlib, datetime, os, sys
from scipy import stats, linalg

warnings.filterwarnings("ignore")
rng = np.random.default_rng(5)  # same seed as common.py
# Work inside the inputs folder; ROOT becomes a plain string for path building.
SP = str(INPUTS)
ROOT = str(ROOT)
os.chdir(SP)
# Log the run time and short SHA-256 fingerprints of the frozen pre-registration and alias table, tying the output to the exact frozen files.
H = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
print(
    f"WITHIN-FRAME RUN {datetime.datetime.utcnow().isoformat()}Z | prereg {H(ROOT+'/handoff/prereg/prereg_withinframe.md')} | alias {H(ROOT+'/handoff/prereg/withinframe_alias_table_FROZEN.csv')}"
)
# ---------------- exposure (show level): corpus dir_z (directive_final) + Majority Report scored from existing transcripts ----------------
S = pd.read_csv(
    INPUTS / "WITHINFRAME_SCORES.csv"
)  # show, show_id, dir_z, share_right, interview(0), fmt_pos(nan), dir_z_declar(nan)
print(f"shows in frame: {len(S)}")
# ---------------- respondents ----------------
# Frozen alias table (respondent, verbatim string, matched show, primary-frame flag) joined to the show-level exposure columns.
A = pd.read_csv(f"{ROOT}/handoff/prereg/withinframe_alias_table_FROZEN.csv")[
    ["resp", "verbatim", "show", "primary_frame"]
].merge(
    S[["show", "show_id", "dir_z", "dir_z_declar", "share_right", "interview", "fmt_pos"]],
    on="show",
    how="inner",
)
A["blaze_only"] = A.verbatim.str.contains("blaze", case=False) & ~A.verbatim.str.contains(  # Blaze / Daily Wire string flags inherited from the holdout test; always False in this frame
    "beck|glenn", case=False
)
A["dw_only"] = A.verbatim.str.contains(r"daily ?wire", case=False) & ~A.verbatim.str.contains(
    "shapiro|ben ", case=False
)
A_all = A.copy()  # keep everyone (incl. Clay & Buck) for the robustness run
A = A[A.primary_frame]  # primary frame only: Clay & Buck excluded
prim = A.sort_values("show").groupby("resp").first().reset_index()[["resp", "show_id", "show"]]  # cluster = alphabetically first show named, as in discovery
# Collapse to one row per respondent: mean exposure over the shows they named (4 respondents name two).
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
# Pre-registered primary outcome: 8-item institutional cynicism. Each item reversed and z-scored on the FULL Kettering sample, averaged (>= 6 of 8 answered), then re-standardised. Secondary: Q33E alone, reversed.
K = pd.read_pickle(INPUTS / "kett.pkl")
num = lambda g, c: pd.to_numeric(g[c], errors="coerce").where(lambda s: s > 0)  # negative codes -> NaN
z = lambda s: (s - s.mean()) / s.std(ddof=0)
ITEMS = ["Q35E", "Q35C", "Q34B", "Q33H", "Q31", "Q33E", "Q35A", "Q35D"]
Z = pd.DataFrame({c: -z(num(K, c)) for c in ITEMS})  # minus sign: items ask whether things work well; higher = more cynical
K["cyn8"] = z(Z.mean(axis=1).where(Z.notna().sum(axis=1) >= 6))  # require at least 6 of the 8 items
K["q33e_rev"] = z(-num(K, "Q33E"))
# Right-wing platform index: how many of Truth Social, Rumble, Parler, Gab are used regularly (code <= 2).
for c in ["Q20C", "Q20G", "Q20E", "Q20H"]:
    K[c] = num(K, c)
K["rw_plat"] = (K[["Q20C", "Q20G", "Q20E", "Q20H"]] <= 2).sum(axis=1).astype(float)
# H2 exclusivity: 1 if none of the respondent's Q17 verbatims names a mainstream outlet (same regex as discovery).
MAIN = r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|washington post|wall street|wsj|reuters|associated press|\bap\b|usa today|local news|newspaper|the hill|politico|axios|bloomberg|cnbc|news ?nation|newsmax"
L = pd.read_csv(INPUTS / "verbatims_long.csv")
L["main"] = L.v.str.contains(MAIN, regex=True, na=False)
E = L.groupby("resp").main.max().rename("any_main").reset_index()
E["excl"] = 1 - E.any_main.astype(float)
g = R.merge(K, on="resp", how="left").merge(E[["resp", "excl"]], on="resp", how="left")  # analysis frame: exposure + survey items + exclusivity
print(
    f"respondents: {len(g)} | pairs: {len(A)} | two-show: {(g.n_shows>1).sum()} | cyn8 non-missing: {g.cyn8.notna().sum()}"
)
# ---------------- reported regardless: per-show table ----------------
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
    .sort_values("dir_z", ascending=False)  # most directive first
)
T["lean"] = T.shareR - T.shareD  # audience lean = Republican share minus Democratic share
print(
    "\n=== PER-SHOW (reported regardless): dir_z, audience partisan lean (R−D share), mean cynicism, exclusivity ==="
)
print(T[["n", "dir_z", "shareR", "shareD", "lean", "cyn8", "excl"]].round(2).to_string())
print(f"show-level corr(dir_z, audience lean) = {np.corrcoef(T.dir_z,T.lean)[0,1]:+.2f}  (n=10)")  # label check: do more directive shows draw a more Republican audience?


# ---------------- inference ----------------
# Same estimator as common.cr_all (CR1; CR2 Bell-McCaffrey with Satterthwaite df; wild cluster bootstrap) but also returning one-sided p-values and a 95% CI, as the pre-registration requires.
def cr_all(y, X, g_, j, B=2000):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    g_ = np.asarray(g_)
    N, K_ = X.shape
    M = np.linalg.inv(X.T @ X)
    b = M @ X.T @ y
    e = y - X @ b
    groups = [np.where(g_ == u)[0] for u in np.unique(g_)]
    G = len(groups)
    Sm = sum((X[i].T @ e[i])[:, None] @ (X[i].T @ e[i])[None, :] for i in groups)  # CR1 meat: sum of per-show score outer products
    V1 = (G / (G - 1)) * ((N - 1) / (N - K_)) * M @ Sm @ M  # small-sample factors G/(G-1) and (N-1)/(N-K)
    se1 = np.sqrt(V1[j, j])
    t1 = b[j] / se1
    # CR2: rescale each show's residuals by (I - H_gg)^(-1/2) so the variance estimate is unbiased under homoskedasticity.
    Hm = X @ M @ X.T
    IH = np.eye(N) - Hm
    c = np.zeros(K_)
    c[j] = 1
    S2 = np.zeros((K_, K_))
    D = np.zeros((N, G))
    for k, i in enumerate(groups):
        Ad = linalg.fractional_matrix_power(np.eye(len(i)) - Hm[np.ix_(i, i)], -0.5).real
        Xa = Ad @ X[i]
        S2 += Xa.T @ np.outer(e[i], e[i]) @ Xa
        D[i, k] = Ad @ X[i] @ M @ c
    V2 = M @ S2 @ M
    se2 = np.sqrt(V2[j, j])
    t2 = b[j] / se2
    # Satterthwaite degrees of freedom: trace(W)^2 / trace(W^2); far below G-1 when one show dominates.
    Wm = D.T @ IH @ D
    df = np.trace(Wm) ** 2 / np.trace(Wm @ Wm)
    p2_two = 2 * stats.t.sf(abs(t2), df)  # two-sided
    p2_one = stats.t.sf(t2, df)  # one-sided, predicted direction positive
    # Wild cluster bootstrap under the null: refit without dir_z, then flip each show's residual block by a coin toss and recompute the CR1 t-statistic.
    X0 = np.delete(X, j, axis=1)
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    f0 = X0 @ b0
    r0 = y - f0
    tb = np.empty(B)
    for s in range(B):
        yb = f0.copy()
        for i in groups:
            yb[i] += r0[i] * rng.choice([-1.0, 1.0])  # one Rademacher draw per show
        bb = M @ X.T @ yb
        eb = yb - X @ bb
        Sb = sum((X[i].T @ eb[i])[:, None] @ (X[i].T @ eb[i])[None, :] for i in groups)
        Vb = (G / (G - 1)) * ((N - 1) / (N - K_)) * M @ Sb @ M
        tb[s] = bb[j] / np.sqrt(Vb[j, j])
    pw_two = (np.abs(tb) >= abs(t1)).mean()  # two-sided wild p
    pw_one = (tb >= t1).mean()  # one-sided wild p: bootstrap t at least as large as observed
    tc = stats.t.ppf(0.975, df)  # CR2 t critical value for the 95% CI
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


# Pre-registered control set: party (5 categories), attention, age, education, show ideology (share_right), right-wing platform index, plus any sensitivity covariates.
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


ROWS = []  # every fitted specification, for the results csv


# Fit one specification, print it with the PASS/fail rule, and record it. expo lets a sensitivity swap in an alternative exposure; per-show weights w are applied by scaling rows by sqrt(weight).
def run(label, d, y="cyn8", expo="dir_z", extra=(), w=None, B=2000):
    X = design(d, extra)
    X["dir_z"] = d[expo].values  # exposure column always named dir_z so the index lookup below works
    m = pd.concat([X, d[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    Xm = m[X.columns].values
    ym = m[y].values
    if w is not None:
        ww = np.sqrt(m.show_id.map(w).values)  # WLS via sqrt-weight scaling of both sides
        Xm = Xm * ww[:, None]
        ym = ym * ww
    r = cr_all(ym, Xm, m.show_id.values, list(X.columns).index("dir_z"), B=B)
    r["label"] = label
    ROWS.append(r)
    flag = "PASS" if (r["b"] > 0 and r["p2_one"] < 0.05 and r["pw_one"] < 0.05) else "fail"  # pre-registered success rule: positive, one-sided CR2 p < .05 AND wild p < .05
    print(
        f"{label:<58} b={r['b']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] CR2 df={r['df']:.1f} p1={r['p2_one']:.3f} (2s {r['p2_two']:.3f}) | wild p1={r['pw_one']:.3f} (2s {r['pw_two']:.3f}) | G={r['G']} N={r['N']}  {flag}",
        flush=True,
    )
    return r


# Show-level permutation (iv): residualise the outcome on controls, average residuals by show, regress on dir_z weighted by show n, and compare with 4000 random reassignments of dir_z across shows.
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
    null = np.array([fitb(rng.permutation(a.dir_z.values)) for _ in range(B)])  # permute exposure across shows
    return b, (null >= b).mean(), (np.abs(null) >= abs(b)).mean(), len(a)  # slope, one-sided p, two-sided p, number of shows


# H1 primary (i), the 20%-cap sensitivity (iii), the show-level permutation (iv), and the verdict.
print("\n=== H1 PRIMARY OUTCOME: 8-item institutional cynicism (z on full Kettering sample) ===")
i_ = run("(i)   PRIMARY within-frame: respondent-level, CR2 + wild", g)
ns = g.show_id.value_counts()
cap = {s: min(1.0, 0.2 * len(g) / n) for s, n in ns.items()}  # weight = min(1, 0.2*N/n_show): shows above 20% of the sample are down-weighted
run("(iii) SENS: per-show weights capped at 20% of sample", g, w=cap)
b, p1, p2, n = showperm(g)
print(
    f"(iv)  show-level precision-weighted permutation: b={b:+.3f} one-sided p={p1:.3f} two-sided p={p2:.3f} (shows={n})"
)
print(
    f"\nVERDICT H1 within-frame (one-sided CR2<.05 AND wild<.05): {'PASS' if i_['p2_one']<.05 and i_['pw_one']<.05 and i_['b']>0 else 'FAIL'}"
)
# Secondary outcome and H2, same model.
print("\n=== SECONDARY: Q33E alone (reversed) ===")
run("(i)   secondary Q33E", g, y="q33e_rev")
print("\n=== H2: exclusivity (no mainstream outlet named), LPM ===")
run("(i)   H2 exclusivity", g, y="excl")
print("\n=== REPORTED ROBUSTNESS ===")
# Survey-weighted robustness: multiply outcome and design rows by sqrt(WEIGHT) before the same clustered estimator.
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
# Clay & Buck inclusion
# Rebuild the respondent frame from ALL alias rows (adds the 12 Clay & Buck respondents as an 11th show).
A2 = A_all.copy()
prim2 = A2.sort_values("show").groupby("resp").first().reset_index()[["resp", "show_id", "show"]]
R2 = (
    A2.groupby("resp")
    .agg(
        dir_z=("dir_z", "mean"),
        dir_z_declar=("dir_z_declar", "mean"),
        share_right=("share_right", "mean"),
        interview=("interview", "mean"),
        fmt_pos=("fmt_pos", "mean"),
        n_shows=("show", "nunique"),
    )
    .reset_index()
    .merge(prim2, on="resp")
)
g2 = R2.merge(K, on="resp", how="left").merge(E[["resp", "excl"]], on="resp", how="left")
run("include Clay & Buck respondents (11 shows)", g2, B=1000)  # fewer bootstrap draws for the robustness run
print("\n--- leave-one-show-out (primary) ---")
# Leave-one-show-out: refit the primary dropping each show in turn (500 bootstrap draws each).
for s in sorted(g.show_id.unique()):
    run(f"  drop {S.set_index('show_id').show.get(s,s)[:24]:<24}", g[g.show_id != s], B=500)
pd.DataFrame(ROWS).to_csv(OUTPUTS / "withinframe_test_results.csv", index=False)  # all specifications, one row each
print("\nDONE")
