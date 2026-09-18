"""Distil the passage-level LLM ideological-intensity labels into a TF-IDF classifier (grouped CV) to score the 11 shows without labels; then the intensity-vs-populism horse race on the pooled frame. Exploratory record F.3.4."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, re, glob, os, sys, importlib.util, gzip, csv, warnings, builtins

warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# --- Step 1. Assemble the LLM-labelled passages with their text.
# ideology_cap40.csv holds the passage-level LLM labels (a capped per-show sample, per its name):
# politics (Yes/No) and ternary (-1 left / 0 neutral / +1 right).
SP = str(INPUTS)
os.chdir(ROOT)  # the rest of the script uses repository-relative paths
sys.path.insert(0, ".")
ide = pd.read_csv("data/output/ideology_cap40.csv")
ids = set(ide.chunk_id.astype(str))
txt = {}
sid = {}
# Stream the 1.38M-passage corpus once and keep text + show id only for the labelled passages.
with gzip.open(DATA / "output/scoring_chunks.csv.gz", "rt", newline="") as fh:
    for r in csv.DictReader(fh):
        if r["chunk_id"] in ids:
            txt[r["chunk_id"]] = r["text"]
            sid[r["chunk_id"]] = r["collection_id"]
ide["text"] = ide.chunk_id.astype(str).map(txt)
ide["show_id"] = ide.chunk_id.astype(str).map(sid)
ide = ide.dropna(subset=["text", "show_id"]).reset_index(drop=True)
ide["pol"] = (ide.politics == "Yes").astype(int)  # stage-1 target: is the passage about politics?
ide["intense"] = (ide.ternary.abs() > 0).astype(int)  # stage-2 target: clearly left or right (vs neutral)
D = pd.read_csv(INPUTS / "directive_final.csv")
D["show_id"] = D.show_id.astype(str)
# Text features: word uni- and bigrams, TF-IDF weighted, capped at 200k features.
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200000, sublinear_tf=True)
X = vec.fit_transform(ide.text)
g = ide.show_id.values  # group id for cross-validation = show


# Fit a logistic classifier on the rows selected by `mask`, with 5-fold cross-validation grouped
# by show (a show's passages are never in both train and test folds, so the score is honest for
# unseen shows). Returns out-of-fold probabilities for every row plus a final model fit on all rows.
def oof_fit(mask, y):
    idx = np.where(mask)[0]
    Xm = X[idx]
    ym = y[idx]
    gm = g[idx]
    oof = np.full(len(y), np.nan)
    o = np.zeros(len(idx))
    for tr, te in GroupKFold(5).split(Xm, ym, gm):
        o[te] = LogisticRegression(C=2.0, max_iter=3000).fit(Xm[tr], ym[tr]).predict_proba(Xm[te])[:, 1]  # C=2: mild regularisation
    oof[idx] = o
    return oof, LogisticRegression(C=2.0, max_iter=3000).fit(Xm, ym)


# --- Step 2. Two-stage distillation: (1) political or not, on all passages;
# (2) ideologically loaded or not, trained only on the political passages.
oof_pol, clf_pol = oof_fit(np.ones(len(ide), bool), ide.pol.values)
print(f"stage 1 (political?): grouped AUC={roc_auc_score(ide.pol,oof_pol):.3f}")
m_pol = (ide.pol == 1).values
oof_int, clf_int = oof_fit(m_pol, ide.intense.values)
print(
    f"stage 2 (ideologically loaded | political): grouped AUC={roc_auc_score(ide.intense.values[m_pol],oof_int[m_pol]):.3f}"
)
ide["p_pol"] = oof_pol
ide["p_int"] = oof_int


# Show-level intensity = mean predicted "loaded" probability over passages the model calls political (p > 0.5),
# mirroring the definition of `inten` (share of political passages labelled clearly left or right).
def show_inten(df):
    return df.p_int[df.p_pol > 0.5].mean()


# Validate at show level: compare the out-of-fold distilled intensity with the true LLM-based inten.
T = (
    ide.groupby("show_id")
    .apply(show_inten, include_groups=False)
    .rename("inten_hat")
    .reset_index()
    .merge(D[["show_id", "inten"]], on="show_id")
)
print(
    f"show-level: corr(true inten, OUT-OF-FOLD distilled) = {T.inten.corr(T.inten_hat):.3f} over {len(T)} shows | slope check: mean true {T.inten.mean():.3f} vs hat {T.inten_hat.mean():.3f}"
)
# --- Step 3. Score the unlabelled shows: the 10 holdout shows plus show 402306412.
# Reuse the corpus's own chunker (split_750 / transcript_text) by loading build_scoring_chunks.py as a module;
# it may call sys.exit() on import, which we swallow.
spec = importlib.util.spec_from_file_location("bsc", "step4_ideology/build_scoring_chunks.py")
bsc = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(bsc)
except SystemExit:
    pass
rows = []
for p in sorted(glob.glob("data/transcripts/HOLDOUT_*/*.json")) + sorted(
    glob.glob("data/transcripts/402306412/*.json")
):
    s = p.split("/")[2]  # the folder name is the show id (e.g. HOLDOUT_rogan)
    rows += [(s, t) for t in bsc.split_750(bsc.transcript_text(p)) if len(t) >= 120]  # drop stub passages under 120 characters
H = pd.DataFrame(rows, columns=["show_id", "text"])
Xh = vec.transform(H.text)  # same vocabulary as the training set
H["p_pol"] = clf_pol.predict_proba(Xh)[:, 1]
H["p_int"] = clf_int.predict_proba(Xh)[:, 1]
HS = H.groupby("show_id").apply(show_inten, include_groups=False).rename("inten_hat").reset_index()
# Combined table: true inten where available, distilled estimate otherwise -> inten_use.
out = pd.concat([T[["show_id", "inten", "inten_hat"]], HS.assign(inten=np.nan)])
out["inten_use"] = out.inten.fillna(out.inten_hat)
out.to_csv(INPUTS / "inten_distilled.csv", index=False)
print("\nunlabeled shows, distilled intensity:")
print(HS.sort_values("inten_hat", ascending=False).round(3).to_string(index=False))
print(
    "labeled corpus shows true inten: mean %.3f range [%.2f,%.2f]"
    % (T.inten.mean(), T.inten.min(), T.inten.max())
)
# --- Step 4. Horse race on the pooled respondent frame (discovery + fresh survey).
# Borrow the set-up code from explore_scan.py (everything before its SCANS START marker): it builds
# the pooled frame and defines design()/cr_all() for that frame. Its prints are silenced while it runs.
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "explore_scan.py")).read()
pre = src[: src.index("# ==== SCANS START")]
_p = builtins.print
builtins.print = lambda *a, **k: None
exec(pre)
builtins.print = _p
# Pooled respondent features, with the intensity columns replaced by the freshly distilled inten_use.
G = (
    pd.read_pickle(INPUTS / "pooled_resp_features.pkl")
    .drop(columns=["inten", "inten_use", "inten_use_z"], errors="ignore")
    .merge(out[["show_id", "inten_use"]], on="show_id", how="left")
)
# z-score the three show-level exposures across SHOWS (not respondents), so 1 unit = 1 SD between shows.
# rp_rate = Rooduijn-Pauwels populism-dictionary rate; dir_z = listener-directed address.
sf = G.groupby("show_id")[["rp_rate", "dir_z", "inten_use"]].first()
for c in sf.columns:
    G[c + "_z"] = (G[c] - sf[c].mean()) / sf[c].std()


# Print one result line: coefficient, CR2 p with Satterthwaite df, wild p, clusters and N.
def show(label, r):
    print(
        f"  {label:<58} b={r['b']:+.3f}  CR2 p={r['p2']:.3f} (df {r['df']:.1f})  wild p={r['pw']:.3f}  G={r['G']} N={r['N']}"
    )


# Fit outcome y on exposure `feat` + base controls + `extra`, clustered by show, via cr_all (B=800 bootstrap draws).
def run(d, y, feat, extra, label, B=800):
    dd = d.dropna(subset=[feat, y] + list(extra))
    X = design(dd, extra)
    X["f"] = dd[feat].values
    m = pd.concat([X, dd[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    show(
        label,
        cr_all(m[y].values, m[X.columns].values, m.show_id.values, list(X.columns).index("f"), B=B),
    )


# The race: does ideological intensity or populist language (rp_rate) better explain listener
# cynicism (cyn8), election distrust (q33e_rev) and exclusivity (excl)? Each alone, then each
# controlling for the other; always controlling for show side (`right`). Run on the pooled frame
# and on the fresh (non-discovery) survey only.
print(
    f"\n=== HORSE RACE, all shows (true intensity for corpus shows, distilled for 11): show-level corr(rp_rate, inten)={sf.corr().loc['rp_rate','inten_use']:+.2f}, corr(dir_z, inten)={sf.corr().loc['dir_z','inten_use']:+.2f} ==="
)
for nm, d in (("POOLED", G), ("FRESH only", G[G.src != "discovery"])):
    run(d, "cyn8", "inten_use_z", ("right",), f"{nm}: intensity alone -> cyn8")
    run(d, "cyn8", "rp_rate_z", ("right",), f"{nm}: rp_rate alone -> cyn8")
    run(d, "cyn8", "rp_rate_z", ("right", "inten_use_z"), f"{nm}: rp_rate | intensity")
    run(d, "cyn8", "inten_use_z", ("right", "rp_rate_z"), f"{nm}: intensity | rp_rate")
    run(
        d, "cyn8", "inten_use_z", ("right", "rp_rate_z", "dir_z_z"), f"{nm}: intensity | rp_rate + dir_z"
    )  # intensity after also netting out address
    run(d, "q33e_rev", "inten_use_z", ("right", "rp_rate_z"), f"{nm}: intensity | rp_rate -> Q33E")
    run(d, "q33e_rev", "rp_rate_z", ("right", "inten_use_z"), f"{nm}: rp_rate | intensity -> Q33E")
    run(d, "excl", "inten_use_z", ("right", "rp_rate_z"), f"{nm}: intensity | rp_rate -> exclusivity")
    run(d, "excl", "rp_rate_z", ("right", "inten_use_z"), f"{nm}: rp_rate | intensity -> exclusivity")
G.to_pickle(INPUTS / "pooled_resp_features.pkl")  # save the frame back with the updated inten_use columns
