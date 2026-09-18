"""Topic invariance of address: within-show topic means across the 75-topic model; the most second-person topics. Paper finding 1."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.formula.api as smf, warnings

warnings.filterwarnings("ignore")
# LDA outputs from the topic arm: per-passage topic proportions for the K=75 model (columns T0..T74) and the top words of each topic.
B = str(ROOT / "topic_arm/output") + "/"
T = pd.read_csv(B + "lda_doctopic_k75_500.csv")
W = pd.read_csv(B + "lda_k75_topic_words.csv")
print("doc-topic:", T.shape, "| keys:", [c for c in T.columns if not c.startswith("T")][:6])
print("topic words cols:", list(W.columns)[:5])
tcols = [c for c in T.columns if c.startswith("T")]  # the 75 topic-proportion columns
import gzip, csv

# LDA rows are keyed by chunk_id; map each back to (episode_id, chunk_ix) through the scoring-chunk table so it can be joined to the register features.
mp = {}
with gzip.open(B + "scoring_chunks.csv.gz", "rt", newline="") as fh:
    for r in csv.DictReader(fh):
        mp[r["chunk_id"]] = (r["episode_id"], int(r["chunk_ix"]))
T["episode_id"] = T.chunk_id.astype(str).map(lambda k: mp.get(k, (None, None))[0])
T["chunk_ix"] = T.chunk_id.astype(str).map(lambda k: mp.get(k, (None, None))[1])
print("chunk->passage mapped:", T.episode_id.notna().mean())  # share of LDA rows we could locate
# Each passage's single dominant topic and how dominant it is.
T["topic"] = T[tcols].idxmax(axis=1)
T["topic_p"] = T[tcols].max(axis=1)
# Per-passage register features from the remeasure run (rates per 10k words, word count nw).
R = pd.read_csv(
    INPUTS / "remeasured.csv",
    usecols=["episode_id", "show_id", "chunk_ix", "nw", "imper_syn", "imper_rx", "you_rx"],
)
R["show_id"] = R.show_id.astype(str)
# try passage-level join on (episode_id, chunk_ix); fall back to episode-level
if T.episode_id.notna().mean() > 0.5:  # passage-level join if most chunks were located
    M = R.merge(
        T[["episode_id", "chunk_ix", "topic", "topic_p"]], on=["episode_id", "chunk_ix"], how="inner"
    )
    level = "passage"
elif "episode_id" in T.columns:
    ep = (
        R.groupby(["episode_id", "show_id"])
        .apply(
            lambda g: pd.Series(
                {c: np.average(g[c], weights=g.nw) for c in ("imper_syn", "imper_rx", "you_rx")}
                | {"nw": g.nw.sum()}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    te = T.groupby("episode_id")[tcols].mean()  # episode's mean topic profile -> dominant topic
    te["topic"] = te.idxmax(axis=1)
    M = ep.merge(te[["topic"]], on="episode_id", how="inner")
    level = "episode"
else:
    raise SystemExit("cannot join topics to register")
# Passage-level address index: mean of the three z-scored rates (z on this joined sample, a passage-SD scale rather than the show-level dir_z scale).
z = lambda s: (s - s.mean()) / s.std(ddof=0)
M["lda"] = (z(M.imper_syn) + z(M.imper_rx) + z(M.you_rx)) / 3
# Attach the show's L/R lean.
M = M.merge(
    pd.read_csv(INPUTS / "directive_final.csv")[["show_id", "lean"]].assign(
        show_id=lambda d: d.show_id.astype(str)
    ),
    on="show_id",
    how="left",
)
print(
    f"joined at {level} level: {len(M):,} units, {M.show_id.nunique()} shows, {M.topic.nunique()} topics\n"
)
# within-show topic effect: demean lda by show, then average by topic
M["lda_dm"] = M.lda - M.groupby("show_id").lda.transform("mean")  # subtract the show's own mean: removes between-show differences
words = W.set_index("topic").top_words.astype(str).to_dict()  # topic id -> top words
# Per-topic summary: passages, shows covered, raw and within-show mean address, and the within-show means for left and right shows separately.
g = (
    M.groupby("topic")
    .agg(
        n=("lda", "size"),
        shows=("show_id", "nunique"),
        lda_raw=("lda", "mean"),
        lda_within=("lda_dm", "mean"),
        L_within=("lda_dm", lambda s: s[M.loc[s.index, "lean"] == "L"].mean()),
        R_within=("lda_dm", lambda s: s[M.loc[s.index, "lean"] == "R"].mean()),
    )
    .reset_index()
)
g = g[g.n >= 300]  # only topics with enough passages
g["words"] = g.topic.map(lambda t: str(words.get(t, ""))[:70])
# If address were topic-driven, the within-show spread across topics would rival the spread across shows.
print(
    "=== listener-directed address by TOPIC, within show (SD units of the corpus; + = more address than the show's own average) ==="
)
print(
    f"  spread across topics: sd of within-show topic means = {g.lda_within.std():.3f} SD  (vs sd across shows of show means = {M.groupby('show_id').lda.mean().std():.3f})"
)
print("\n  TOP 12 topics for address:")
print(
    g.sort_values("lda_within", ascending=False)
    .head(12)[["topic", "n", "lda_within", "L_within", "R_within", "words"]]
    .to_string(index=False, float_format=lambda x: f"{x:+.3f}")
)
print("\n  BOTTOM 12:")
print(
    g.sort_values("lda_within")
    .head(12)[["topic", "n", "lda_within", "L_within", "R_within", "words"]]
    .to_string(index=False, float_format=lambda x: f"{x:+.3f}")
)
# variance decomposition: how much of lda variance is topic vs show?
import statsmodels.api as sm

m0 = smf.ols("lda ~ C(show_id)", data=M).fit()  # show fixed effects only
m1 = smf.ols("lda ~ C(show_id)+C(topic)", data=M).fit()  # add topic fixed effects
print(
    f"\n=== variance decomposition (R²): show alone {m0.rsquared:.3f} | show + topic {m1.rsquared:.3f} | topic adds {m1.rsquared-m0.rsquared:.3f}"
)
# does the LEFT-RIGHT gap depend on topic? topic x side interaction, within topic
sub = M[M.lean.isin(["L", "R"])].copy()
sub["R"] = (sub.lean == "R").astype(int)  # 1 = right-leaning show
mi = smf.ols("lda ~ C(topic)*R", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.show_id})  # topic-specific left-right gaps
ma = smf.ols("lda ~ C(topic)+R", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.show_id})  # one common gap
print(
    f"    left-right gap in address, pooled across topics: {ma.params['R']:+.3f} SD (p={ma.pvalues['R']:.2e}); topic×side interaction adds R² {mi.rsquared-ma.rsquared:.4f}"
)
# Raw right-minus-left address gap by topic: which topics carry the most and least contrast.
gap = sub.groupby(["topic", "R"]).lda.mean().unstack()
gap["gap"] = gap[1] - gap[0]
gap = gap.join(g.set_index("topic")[["n", "words"]]).dropna()
print("\n  topics where the RIGHT-LEFT address gap is LARGEST:")
print(
    gap.sort_values("gap", ascending=False)
    .head(6)[["gap", "n", "words"]]
    .to_string(float_format=lambda x: f"{x:+.3f}")
)
print("  ...and SMALLEST / reversed:")
print(
    gap.sort_values("gap").head(6)[["gap", "n", "words"]].to_string(float_format=lambda x: f"{x:+.3f}")
)
g.to_csv(INPUTS / "topic_lda.csv", index=False)  # per-topic table for the paper
