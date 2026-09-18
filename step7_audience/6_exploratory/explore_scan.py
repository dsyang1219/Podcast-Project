"""Pooled 64-show, 1,284-listener feature scan: every show content feature against cynicism, election distrust and exclusivity (scans A, B, D). The block above the SCANS marker assembles the pooled frame and is reused by distill_inten.py. Exploratory record F.3.4; education composition (finding 3) and the political frame (finding 7) build on its outputs."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, warnings, os, re
from statsmodels.stats.multitest import multipletests
from scipy import stats, linalg

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)
pd.set_option("display.width", 250)
# Show table for the scored corpus shows: address (dir_z and its three components), lean, format (solo,
# n_hosts, episode length), host DIME score, LLM side and ideological intensity.
D = pd.read_csv(INPUTS / "directive_final.csv")
D["show_id"] = D.show_id.astype(str)
F = D[
    [
        "show_id",
        "show",
        "dir_z",
        "imper_syn",
        "imper_rx",
        "you_rx",
        "lean",
        "solo",
        "n_hosts",
        "ep_len",
        "avg_host_cfscore",
        "side",
        "inten",
    ]
].copy()
# Merge in the per-show content-feature files: populism rates, anti-media / political-content rates, game-frame
# (horse-race and strategy) rates, Biber dimension 4 with its imperative (VIMP) and 2nd-person (PP2) loadings, and
# the imperative / 'you' subtypes. All rates are per 10,000 words, show means weighted by passage word count.
for f, cols in (
    (INPUTS / "show_populist.csv", ["elite_rate", "people_rate", "media_rate", "populist_rate"]),
    (INPUTS / "show_rp_antimedia.csv", ["rp_rate", "pc_rate", "antimedia_rate"]),
    (INPUTS / "show_gameframe.csv", ["horse_rate", "strat_rate", "game_rate"]),
    (INPUTS / "show_biber_d4.csv", ["D4_biber", "VIMP", "PP2"]),
    (
        INPUTS / "show_imp_decomp.csv",
        [
            "you_deontic_rate",
            "you_know_rate",
            "imp_attention_rate",
            "imp_action_rate",
            "imp_epistemic_rate",
            "you_addressive_rate",
        ],
    ),
):
    x = pd.read_csv(f)
    x["show_id"] = x.show_id.astype(str)
    F = F.merge(
        x[["show_id"] + cols].rename(
            columns={"media_rate": "mediacrit_rate"} if "populist" in str(f) else {}  # the populism file's media_rate means 'media criticism'; rename to avoid a clash
        ),
        on="show_id",
        how="left",
    )
# Share of a show's passages that are political, if the corpus-level file exists (else left missing).
if os.path.exists(INPUTS / "corpus_poldensity.csv"):
    x = pd.read_csv(INPUTS / "corpus_poldensity.csv")
    x["show_id"] = x.show_id.astype(str)
    F = F.merge(x[["show_id", "share_political"]], on="show_id", how="left")
else:
    F["share_political"] = np.nan
# Out-of-frame shows: the 10 holdout shows plus Majority Report (id 402306412, found by the within-frame re-scan),
# with their dir_z, dictionary features and political density brought to the same column names as F.
H = pd.read_csv(INPUTS / "holdout_show_dirz_FINAL.csv")
M = pd.read_csv(INPUTS / "mr_show_dirz.csv")
M["show_id"] = M.show_id.astype(str)
H = pd.concat([H, M])
hd = pd.read_csv(INPUTS / "holdout_dict_features.csv")
hd["show_id"] = hd.show_id.astype(str)
pdn = pd.read_csv(INPUTS / "holdout_political_density.csv")
pdn["show_id"] = "HOLDOUT_" + pdn.show  # holdout ids are 'HOLDOUT_<slug>'
pdn = pd.concat([pdn, pd.DataFrame([{"show_id": "402306412", "share_political_passages": np.nan}])])  # Majority Report has no density file: add an explicit missing row
H = H.merge(
    hd[
        [
            "show_id",
            "elite_rate",
            "people_rate",
            "mediacrit_rate",
            "populist_rate",
            "rp_rate",
            "pc_rate",
            "antimedia_rate",
            "horse_rate",
            "strat_rate",
            "game_rate",
        ]
    ],
    on="show_id",
    how="left",
).merge(
    pdn[["show_id", "share_political_passages"]].rename(
        columns={"share_political_passages": "share_political"}
    ),
    on="show_id",
    how="left",
)
# Display names and lean for the out-of-frame shows (all lean R unless listed in LEAN); the three interview
# shows are coded multi-host.
NAME = {
    "HOLDOUT_rogan": "Joe Rogan",
    "HOLDOUT_shapiro": "Ben Shapiro",
    "HOLDOUT_thedaily": "The Daily",
    "HOLDOUT_tucker": "Tucker Carlson",
    "HOLDOUT_beck": "Glenn Beck",
    "HOLDOUT_kirk": "Charlie Kirk",
    "HOLDOUT_parnas": "Parnas",
    "HOLDOUT_kelly": "Megyn Kelly",
    "HOLDOUT_ryan": "Shawn Ryan",
    "HOLDOUT_bongino": "Dan Bongino",
    "402306412": "Majority Report",
}
LEAN = {"HOLDOUT_thedaily": "L", "HOLDOUT_parnas": "L", "402306412": "L"}
H["show"] = H.show_id.map(NAME)
H["lean"] = H.show_id.map(LEAN).fillna("R")
H["solo"] = np.where(
    H.show_id.isin(["HOLDOUT_rogan", "HOLDOUT_tucker", "HOLDOUT_ryan"]), "multi", "solo"
)
# Pooled 64-show feature table: corpus shows plus the out-of-frame shows (replacing any duplicates), with dummies
# for interview format, multi-host format and right lean. Saved for other scripts.
F = pd.concat([F[~F.show_id.isin(H.show_id)], H], ignore_index=True)
F["interview"] = F.show_id.isin(["HOLDOUT_rogan", "HOLDOUT_tucker", "HOLDOUT_ryan"]).astype(float)
F["multi"] = (F.solo == "multi").astype(float)
F["right"] = (F.lean == "R").astype(float)
F.to_csv(INPUTS / "show_features_ALL.csv", index=False)
# Respondent-level pooled frame (discovery + out-of-frame + within-frame-pocket listeners) joined to show features;
# keep only respondents whose show has an address score. share_right here is simply the show's right dummy.
G = pd.read_pickle(INPUTS / "pooled_resp.pkl").merge(F, on="show_id", how="left")
G = G[G.dir_z.notna()].copy()
G["share_right"] = G.right
# Coverage report: how many of the pooled shows carry each feature.
print(
    f"pooled: {len(G)} respondents, {G.show_id.nunique()} shows | feature coverage (shows): "
    + ", ".join(
        f"{c}:{G.groupby('show_id')[c].first().notna().sum()}"
        for c in [
            "dir_z",
            "populist_rate",
            "antimedia_rate",
            "game_rate",
            "share_political",
            "D4_biber",
            "you_deontic_rate",
            "avg_host_cfscore",
        ]
    )
)
# The show features to scan (those present in the frame): address and its subtypes, populism, anti-media,
# game frame, Biber D4, political share, episode length, intensity, host ideology, and the format/lean dummies.
FEATS = [
    f
    for f in [
        "dir_z",
        "imper_syn",
        "imper_rx",
        "you_rx",
        "you_deontic_rate",
        "you_know_rate",
        "imp_attention_rate",
        "imp_action_rate",
        "imp_epistemic_rate",
        "populist_rate",
        "elite_rate",
        "people_rate",
        "mediacrit_rate",
        "antimedia_rate",
        "rp_rate",
        "game_rate",
        "horse_rate",
        "strat_rate",
        "D4_biber",
        "share_political",
        "ep_len",
        "inten",
        "avg_host_cfscore",
        "right",
        "multi",
        "interview",
    ]
    if f in G.columns
]


# Local copy of common.cr_all with fewer bootstrap draws: OLS coefficient j with CR2 (Bell-McCaffrey) SE and
# Satterthwaite df, plus a wild cluster bootstrap p referenced to the CR1 t. Returns b, se2, df, p2, pw, G, N.
def cr_all(y, X, g_, j, B=1000):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    g_ = np.asarray(g_)
    N, K_ = X.shape
    Mi = np.linalg.inv(X.T @ X)
    b = Mi @ X.T @ y
    e = y - X @ b
    groups = [np.where(g_ == u)[0] for u in np.unique(g_)]
    Gn = len(groups)
    Sm = sum(np.outer(X[i].T @ e[i], X[i].T @ e[i]) for i in groups)  # CR1 meat: sum of per-show score outer products
    V1 = (Gn / (Gn - 1)) * ((N - 1) / (N - K_)) * Mi @ Sm @ Mi  # CR1 sandwich with the usual small-sample factors
    se1 = np.sqrt(V1[j, j])
    t1 = b[j] / se1
    Hm = X @ Mi @ X.T
    IH = np.eye(N) - Hm
    c = np.zeros(K_)
    c[j] = 1
    S2 = np.zeros((K_, K_))
    Dm = np.zeros((N, Gn))
    # CR2: rescale each show's residual block by (I - H_gg)^(-1/2) so the SE is unbiased under homoskedasticity;
    # Dm collects the pieces needed for the Satterthwaite df.
    for k, i in enumerate(groups):
        Ad = linalg.fractional_matrix_power(np.eye(len(i)) - Hm[np.ix_(i, i)], -0.5).real
        Xa = Ad @ X[i]
        S2 += Xa.T @ np.outer(e[i], e[i]) @ Xa
        Dm[i, k] = Ad @ X[i] @ Mi @ c
    V2 = Mi @ S2 @ Mi
    se2 = np.sqrt(V2[j, j])
    t2 = b[j] / se2
    Wm = Dm.T @ IH @ Dm
    df = np.trace(Wm) ** 2 / np.trace(Wm @ Wm)  # Satterthwaite effective df: the honest number of clusters
    p2 = 2 * stats.t.sf(abs(t2), df)
    # Wild cluster bootstrap under the null: refit without column j, flip each show's residuals by a coin toss,
    # re-estimate, and collect CR1 t-stats; pw = share at least as extreme as the observed t.
    X0 = np.delete(X, j, axis=1)
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    f0 = X0 @ b0
    r0 = y - f0
    tb = np.empty(B)
    for s in range(B):
        yb = f0.copy()
        for i in groups:
            yb[i] += r0[i] * rng.choice([-1.0, 1.0])
        bb = Mi @ X.T @ yb
        eb = yb - X @ bb
        Sb = sum(np.outer(X[i].T @ eb[i], X[i].T @ eb[i]) for i in groups)
        Vb = (Gn / (Gn - 1)) * ((N - 1) / (N - K_)) * Mi @ Sb @ Mi
        tb[s] = bb[j] / np.sqrt(Vb[j, j])
    return dict(b=b[j], se2=se2, df=df, p2=p2, pw=(np.abs(tb) >= abs(t1)).mean(), G=Gn, N=N)


# Control matrix: party dummies, attention, age, education, right-wing-platform count, plus any extras.
def design(d, extra=()):
    X = pd.get_dummies(d["pid"], prefix="pid", drop_first=True).astype(float)
    X["attn"] = d.attn
    X["age"] = d.age
    X["edu"] = d.edu
    X["rw_plat"] = d.rw_plat
    for e in extra:
        X[e] = d[e]
    return sm.add_constant(X, has_constant="add")


# One scan-B regression: outcome y on a show feature z-scored ACROSS SHOWS (one unit = one show SD, not one
# respondent SD), controls plus show lean ('right') unless the feature is lean itself; clustered by show.
def fit(d, y, feat, extra=("right",), B=1000):
    dd = d.dropna(subset=[feat, y])
    sf = dd.groupby("show_id")[feat].first()  # one value per show
    zf = (dd[feat] - sf.mean()) / sf.std()  # z using the show-level mean and SD
    X = design(dd, [e for e in extra if e != feat])
    X["f"] = zf.values
    m = pd.concat([X, dd[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    return cr_all(m[y].values, m[X.columns].values, m.show_id.values, list(X.columns).index("f"), B=B)


# ==== SCANS START (the block above assembles the pooled frame and is reused by distill_inten.py) ====
# --- SCAN B: respondent-level; every feature x every outcome (cynicism, reversed Q33E, exclusivity).
print(
    "\n=== SCAN B: each show feature (z across shows) -> outcome; respondent-level; controls party/attn/age/edu/rw_plat/show-lean; CR2 + wild; all shows as clusters; BH across features ==="
)
out = []
for y in ("cyn8", "q33e_rev", "excl"):
    for f in FEATS:
        try:
            r = fit(G, y, f)
            out.append(
                dict(y=y, feat=f, b=r["b"], p2=r["p2"], pw=r["pw"], df=r["df"], G=r["G"], N=r["N"])
            )
        except Exception as e:
            print("skip", f, y, e)
# BH false-discovery correction within each outcome, applied to the wild-bootstrap p-values (floored at 1e-4).
O = pd.DataFrame(out)
for y in ("cyn8", "q33e_rev", "excl"):
    s = O[O.y == y].copy()
    s["q_wild"] = multipletests(s.pw.clip(1e-4), method="fdr_bh")[1]  # q-values: BH across the features for this outcome
    s = s.sort_values("pw")
    print(f"\n--- outcome {y} ---")
    print(s[["feat", "b", "p2", "pw", "q_wild", "df", "G", "N"]].round(3).to_string(index=False))
O.to_csv(OUTPUTS / "explore_scanB.csv", index=False)
# --- SCAN A: show-level. Residualise cyn8 and excl on the controls (no show terms), average the residuals by
# show, keep shows with >= 5 listeners, and correlate the show mean with each feature, weighting by listeners.
print(
    "\n=== SCAN A: show-level, party-adjusted audience means (shows >=5 resp), n-weighted correlation, permutation p ==="
)
X = design(G)
m = pd.concat([X, G[["cyn8", "q33e_rev", "excl", "show_id"]]], axis=1).dropna()
for y in ("cyn8", "excl"):
    m["r_" + y] = sm.OLS(m[y], m[X.columns]).fit().resid
A = (
    m.groupby("show_id")
    .agg(n=("cyn8", "size"), r_cyn=("r_cyn8", "mean"), r_excl=("r_excl", "mean"))
    .reset_index()
    .merge(F, on="show_id")
)
A = A[A.n >= 5]  # at least 5 respondents per show


# Weighted Pearson correlation (weights = respondents per show).
def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return np.sum(w * (x - mx) * (y - my)) / np.sqrt(
        np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)
    )


# Permutation p: shuffle the feature across shows 3,000 times and compare the weighted correlation.
rows = []
for f in FEATS:
    a = A.dropna(subset=[f])
    if len(a) < 8 or a[f].std() == 0:  # need at least 8 shows with the feature, and some variation in it
        continue
    for y in ("r_cyn", "r_excl"):
        r = wcorr(a[f].values, a[y].values, a.n.values)
        null = np.array(
            [wcorr(rng.permutation(a[f].values), a[y].values, a.n.values) for _ in range(3000)]
        )
        rows.append((f, y, len(a), r, (np.abs(null) >= abs(r)).mean()))
SA = pd.DataFrame(rows, columns=["feat", "y", "shows", "r_w", "p_perm"])
# BH across features within each audience outcome; print sorted by permutation p.
for y in ("r_cyn", "r_excl"):
    s = SA[SA.y == y].copy()
    s["q"] = multipletests(s.p_perm.clip(1e-4), method="fdr_bh")[1]
    print(f"\n--- audience {y} ---")
    print(
        s.sort_values("p_perm")[["feat", "shows", "r_w", "p_perm", "q"]].round(3).to_string(index=False)
    )
A.to_csv(OUTPUTS / "explore_showlevel.csv", index=False)
# --- SCAN D: does the dir_z -> cynicism slope differ by respondent or show characteristics?
# dz = dir_z z-scored across shows; each moderator is z-scored and interacted with dz.
print("\n=== SCAN D: heterogeneity of dir_z -> cyn8 (interactions; z-scored moderators), all shows ===")
sf = G.groupby("show_id").dir_z.first()
G["dz"] = (G.dir_z - sf.mean()) / sf.std()
G["isR"] = G.pid.isin(["R", "leanR"]).astype(float)  # Republican identifiers and leaners
G["isD"] = G.pid.isin(["D", "leanD"]).astype(float)
G["fresh"] = (G.src != "discovery").astype(float)  # 1 = out-of-frame or within-frame-pocket listener, 0 = discovery sample
for mod in (
    "isR",
    "isD",
    "attn",
    "age",
    "edu",
    "excl",
    "rw_plat",
    "fresh",
    "right",
    "share_political",
    "multi",
):
    d = G.dropna(subset=[mod, "cyn8"]).copy()
    if d[mod].std() == 0:
        continue
    mz = (d[mod] - d[mod].mean()) / d[mod].std()  # z-score the moderator across respondents
    d["inter"] = d.dz * mz
    d["modz"] = mz
    X = design(d, ("right", "modz", "inter"))  # controls + right + moderator main effect + interaction
    X["dz"] = d.dz.values
    if mod == "right":  # when the moderator IS right, its main effect is already in the design
        X = X.drop(columns=["modz"])
    m = pd.concat([X, d[["cyn8", "show_id"]]], axis=1).dropna().reset_index(drop=True)
    r = cr_all(  # inference on the interaction term (600 bootstrap draws)
        m.cyn8.values, m[X.columns].values, m.show_id.values, list(X.columns).index("inter"), B=600
    )
    rm = cr_all(m.cyn8.values, m[X.columns].values, m.show_id.values, list(X.columns).index("dz"), B=200)  # dir_z main effect at the mean moderator (fewer draws; descriptive)
    print(
        f"  dir_z x {mod:<16} interaction b={r['b']:+.3f}  CR2 p={r['p2']:.3f} wild p={r['pw']:.3f} | dir_z main at mean moderator b={rm['b']:+.3f} p={rm['p2']:.3f}  (G={r['G']} N={r['N']})"
    )
G.to_pickle(INPUTS / "pooled_resp_features.pkl")  # pooled frame with features and dz, for downstream scripts
