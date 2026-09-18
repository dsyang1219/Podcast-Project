"""Shared data frame and inference helpers for the audience arm.

Importing this module (``from common import *``) gives every script the same objects the
original scripts obtained by exec-ing the top of verify.py and cr2.py:

    st, ex          respondent-level frames for the strict (exact-title) and expanded (host-name)
                    Kettering matches, each with the show's listener-directed-address score
                    (dir_z), the share of the respondent's named shows that lean right, the
                    right-wing-platform index, and the two 4-item composites from the discovery
                    phase (elec_distrust, inst_cyn)
    design()        the control matrix: party dummies, attention, age, education, plus extras
    clus()          OLS with conventional cluster-robust (CR1) standard errors, clustered by show
    wild()          wild cluster bootstrap p-value (Rademacher weights, restricted residuals)
    showperm()      show-level precision-weighted permutation test
    cr_all()        CR1, CR2 (Bell-McCaffrey) with Satterthwaite df, and wild bootstrap, in one call
    design_np()     numpy design for cr_all()
    R               per-respondent flag: named at least one mainstream outlet (for exclusivity)

Inputs (step7_audience/inputs/): kett.pkl, directive_final.csv, match_strict.csv,
match_expanded.csv, verbatims_long.csv.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import linalg, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import INPUTS  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng(5)

# ---------------------------------------------------------------------------
# Respondent frames
# ---------------------------------------------------------------------------
K = pd.read_pickle(INPUTS / "kett.pkl")
S = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "dir_z", "lean"]]
IDEO = ("share_right", "rw_plat")


def frame(match_file):
    """Respondent-level frame from a respondent-show match table.

    Exposure is the mean dir_z across the corpus shows a respondent named; the cluster is the
    alphabetically first show they named.
    """
    M = pd.read_csv(match_file).merge(S, on="show_id", how="left")
    prim = M.sort_values("show_id").groupby("resp").first().reset_index()[["resp", "show_id"]]
    R_ = (
        M.groupby("resp")
        .agg(dir_z=("dir_z", "mean"), share_right=("lean", lambda x: (x == "R").mean()))
        .reset_index()
    )
    g = R_.merge(prim, on="resp").merge(K, on="resp", how="left")
    for c in ["Q20C", "Q20G", "Q20E", "Q20H"]:  # Truth Social, Rumble, Parler, Gab
        s = pd.to_numeric(g[c], errors="coerce")
        g[c] = s.where(s > 0)
    g["rw_plat"] = (g[["Q20C", "Q20G", "Q20E", "Q20H"]] <= 2).sum(axis=1).astype(float)
    return g


def num(g, c):
    """A survey item as numbers, with Kettering's negative missing codes set to NaN."""
    s = pd.to_numeric(g[c], errors="coerce")
    return s.where(s > 0)


def addcomp(g):
    """Add the two discovery-phase composites (standardised within the frame)."""
    g = g.copy()
    # Election distrust: reversed Q33E (elections administered well) + Q32C (assume fraud)
    e = 6 - num(g, "Q33E")
    f = num(g, "Q32C")
    g["elec_distrust"] = ((e - e.mean()) / e.std() + (f - f.mean()) / f.std()) / 2
    # Institutional cynicism, 4 items reversed: Q35E, Q35C, Q34B, Q33H
    z = lambda s: (s - s.mean()) / s.std()
    g["inst_cyn"] = -(z(num(g, "Q35E")) + z(num(g, "Q35C")) + z(num(g, "Q34B")) + z(num(g, "Q33H"))) / 4
    return g


st = addcomp(frame(INPUTS / "match_strict.csv"))
ex = addcomp(frame(INPUTS / "match_expanded.csv"))

# Show side label (from the LLM ideology classifier) attached to both frames
SIDE = pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "side"]]
st = st.drop(columns=["side"], errors="ignore").merge(SIDE, on="show_id", how="left")
ex = ex.drop(columns=["side"], errors="ignore").merge(SIDE, on="show_id", how="left")

# Did the respondent name any mainstream outlet? (1 - this = "exclusive" diet)
L = pd.read_csv(INPUTS / "verbatims_long.csv")
MAIN = (
    r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|washington post|wall street|wsj|"
    r"reuters|associated press|\bap\b|usa today|local news|newspaper|the hill|politico|axios|"
    r"bloomberg|cnbc|news ?nation|newsmax"
)
L["main"] = L.v.str.contains(MAIN, regex=True, na=False)
R = L.groupby("resp").agg(any_main=("main", "max")).reset_index()


# ---------------------------------------------------------------------------
# Models and inference
# ---------------------------------------------------------------------------
def design(g, extra):
    """Control matrix: party dummies, attention, age, education, plus any extra columns."""
    X = pd.get_dummies(g["pid"], prefix="pid", drop_first=True).astype(float)
    X["attn"] = g.attn
    X["age"] = g.age
    X["edu"] = g.edu
    for e in extra:
        X[e] = g[e]
    return sm.add_constant(X, has_constant="add")


def clus(g, y, extra=()):
    """OLS of y on dir_z + controls with conventional show-clustered SEs. Returns (b, p, n)."""
    X = design(g, extra)
    X["dir_z"] = g.dir_z
    m = pd.concat([X, g[[y, "show_id"]]], axis=1).dropna()
    r = sm.OLS(m[y], m[X.columns]).fit(cov_type="cluster", cov_kwds={"groups": m.show_id})
    return r.params["dir_z"], r.pvalues["dir_z"], len(m)


def wild(g, y, extra=(), B=1500):
    """Wild cluster bootstrap p-value for dir_z (Rademacher weights, residuals from the null model)."""
    X = design(g, extra)
    Xf = X.copy()
    Xf["dir_z"] = g.dir_z
    m = pd.concat([Xf, g[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    c0 = [c for c in Xf.columns if c != "dir_z"]
    ids = m.show_id.values
    u = np.unique(ids)
    full = sm.OLS(m[y], m[Xf.columns]).fit(cov_type="cluster", cov_kwds={"groups": ids})
    t0 = full.tvalues["dir_z"]
    r0 = sm.OLS(m[y], m[c0]).fit()
    res = r0.resid.values
    f0 = r0.fittedvalues.values
    sg = pd.Series(index=u, dtype=float)
    tb = []
    for _ in range(B):
        sg[:] = rng.choice([-1.0, 1.0], len(u))  # one coin per show
        yb = f0 + res * sg.reindex(ids).values
        tb.append(
            sm.OLS(yb, m[Xf.columns]).fit(cov_type="cluster", cov_kwds={"groups": ids}).tvalues["dir_z"]
        )
    return (np.abs(np.array(tb)) >= abs(t0)).mean()


def showperm(g, y, extra=(), B=4000):
    """Show-level test: regress show-mean residuals on dir_z, weighted by n; permute dir_z across shows."""
    X = design(g, extra)
    m = pd.concat([X, g[[y, "dir_z", "show_id"]]], axis=1).dropna().reset_index(drop=True)
    m["r"] = sm.OLS(m[y], m[X.columns]).fit().resid
    a = (
        m.groupby("show_id")
        .agg(dir_z=("dir_z", "first"), r=("r", "mean"), n=("r", "size"))
        .reset_index()
    )
    a = a[a.n >= 2]
    fitb = lambda x: sm.WLS(a.r.values, sm.add_constant(x), weights=a.n.values).fit().params[1]
    b = fitb(a.dir_z.values)
    null = np.array([fitb(rng.permutation(a.dir_z.values)) for _ in range(B)])
    return b, (np.abs(null) >= abs(b)).mean(), len(a)


def cr_all(y, X, g, j, B=2000):
    """Coefficient j of OLS(y ~ X) with three inferences, clustered by g.

    Returns a dict: b, se1/p1 (CR1, t on G-1 df), se2/df/p2 (CR2 Bell-McCaffrey with
    Satterthwaite df, Pustejovsky & Tipton 2018), pw (wild cluster bootstrap), G, N.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    g = np.asarray(g)
    N, K_ = X.shape
    M = np.linalg.inv(X.T @ X)
    b = M @ X.T @ y
    e = y - X @ b
    groups = [np.where(g == u)[0] for u in np.unique(g)]
    G = len(groups)
    # CR1
    S_ = sum((X[i].T @ e[i])[:, None] @ (X[i].T @ e[i])[None, :] for i in groups)
    V1 = (G / (G - 1)) * ((N - 1) / (N - K_)) * M @ S_ @ M
    se1 = np.sqrt(V1[j, j])
    t1 = b[j] / se1
    p1 = 2 * stats.t.sf(abs(t1), G - 1)
    # CR2 + Satterthwaite df
    H = X @ M @ X.T
    IH = np.eye(N) - H
    c = np.zeros(K_)
    c[j] = 1
    S2 = np.zeros((K_, K_))
    D = np.zeros((N, G))
    for k, i in enumerate(groups):
        A = linalg.fractional_matrix_power(np.eye(len(i)) - H[np.ix_(i, i)], -0.5).real
        Xa = A @ X[i]
        S2 += Xa.T @ np.outer(e[i], e[i]) @ Xa
        D[i, k] = A @ X[i] @ M @ c
    V2 = M @ S2 @ M
    se2 = np.sqrt(V2[j, j])
    t2 = b[j] / se2
    W = D.T @ IH @ D
    df = np.trace(W) ** 2 / np.trace(W @ W)
    p2 = 2 * stats.t.sf(abs(t2), df)
    # Wild cluster bootstrap under the null (coefficient j removed), Rademacher weights
    X0 = np.delete(X, j, axis=1)
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    f0 = X0 @ b0
    r0 = y - f0
    tb = np.empty(B)
    for s in range(B):
        yb = f0.copy()
        for i in groups:
            yb[i] += r0[i] * rng.choice([-1.0, 1.0])
        bb = M @ X.T @ yb
        eb = yb - X @ bb
        Sb = sum((X[i].T @ eb[i])[:, None] @ (X[i].T @ eb[i])[None, :] for i in groups)
        Vb = (G / (G - 1)) * ((N - 1) / (N - K_)) * M @ Sb @ M
        tb[s] = bb[j] / np.sqrt(Vb[j, j])
    pw = (np.abs(tb) >= abs(t1)).mean()
    return dict(b=b[j], se1=se1, p1=p1, se2=se2, df=df, p2=p2, pw=pw, G=G, N=N)


def design_np(g, y, extra):
    """Numpy arrays for cr_all(): y, X (with dir_z last), cluster ids, and the index of dir_z."""
    X = design(g, extra)
    X["dir_z"] = g.dir_z
    m = pd.concat([X, g[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    return m[y].values, m[X.columns].values, m.show_id.values, list(X.columns).index("dir_z")
