"""Pre-registered test on the free-text democracy item (handoff/prereg/prereg_Q28.md). It failed on its own criterion; reported in the disclosure section."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import pandas as pd, numpy as np, statsmodels.api as sm, re, warnings
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
rng = np.random.default_rng(28)
from common import *

# IDEO and K duplicate what common already provides (same values). Note that `from common import *`
# also imports common's rng (seed 5), which replaces the seed-28 generator created just above.
IDEO = ("share_right", "rw_plat")
K = pd.read_pickle(INPUTS / "kett.pkl")
# VADER sentiment for M3: try the standalone package, then NLTK's copy; if neither exists M3 is skipped.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    V = SentimentIntensityAnalyzer()
except Exception:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        V = SentimentIntensityAnalyzer()
    except Exception:
        V = None
# Pre-registered dictionaries (handoff/prereg/prereg_Q28.md). M1 = out-group pronouns (they/them/their) as a
# share of all personal pronouns; M2 = negation words; M4 = cynicism lexicon. Apostrophe-free variants
# (dont, doesnt, ...) are added because respondents often type without apostrophes.
PRON = re.compile(
    r"\b(i|me|my|mine|we|us|our|ours|you|your|yours|he|him|his|she|her|hers|they|them|their|theirs|it|its)\b"
)
OUTG = re.compile(r"\b(they|them|their|theirs)\b")
NEG = re.compile(
    r"\b(not|no|never|don't|doesn't|isn't|aren't|can't|won't|nothing|none|dont|doesnt|isnt|arent|cant|wont)\b"
)
CYN = re.compile(
    r"\b(corrupt|corrupted|corruption|broken|rigged|lie|lies|lying|joke|sham|fake|illusion|supposed|elite|elites|control|controlled|power|fail|failed|failing)\b|used to|no longer|rich|money|doesn't work|doesnt work|not really|in theory"
)


# Text features for one Q28 answer. Answers under 5 words are dropped (rates undefined), per the prereg.
def feats(t):
    t = str(t).lower()
    w = re.findall(r"[a-z']+", t)
    n = len(w)
    if n < 5:
        return pd.Series({"words": np.nan})  # NaN word count -> every measure NaN downstream
    pr = len(PRON.findall(t))
    og = len(OUTG.findall(t))
    return pd.Series(
        {
            "words": n,
            "M1_outgroup": og / pr if pr else np.nan,  # share of pronouns that are out-group; NaN if no pronouns
            "M2_negation": len(NEG.findall(t)) / n,  # negation words per word
            "M3_valence": V.polarity_scores(t)["compound"] if V else np.nan,  # VADER compound in [-1, 1]; predicted negative
            "M4_cynicism": len(CYN.findall(t)) / n,  # cynicism hits per word
        }
    )


# Keep non-empty Q28 answers; "-98"/"-99" are Kettering refused / don't-know codes stored as text.
q = K[["resp", "Q28"]].copy()
q["Q28"] = q.Q28.astype(str).str.strip()
q = q[~q.Q28.isin(["nan", "", "-98", "-99"])]
F = pd.concat([q[["resp"]], q.Q28.apply(feats)], axis=1)
F["logw"] = np.log(F.words)  # log word count enters every model as a control
# The four measures and their pre-registered directions (used for the one-sided p-value).
MEAS = ["M1_outgroup", "M2_negation", "M3_valence", "M4_cynicism"]
PRED = {"M1_outgroup": +1, "M2_negation": +1, "M3_valence": -1, "M4_cynicism": +1}
print("VADER available:", V is not None)


# Wild cluster bootstrap for dir_z, as in common.wild(), but with the prereg's extra controls
# (share_right, rw_plat, log words) and returning both one-sided and two-sided p-values.
def wild_dir(g, y, B=2000):
    X = design(g, IDEO)
    X["logw"] = g.logw
    Xf = X.copy()
    Xf["dir_z"] = g.dir_z
    m = pd.concat([Xf, g[[y, "show_id"]]], axis=1).dropna().reset_index(drop=True)
    if len(m) < 60:  # too few usable answers to test
        return np.nan, np.nan, np.nan, len(m)
    c0 = [c for c in Xf.columns if c != "dir_z"]  # null model: everything except dir_z
    ids = m.show_id.values
    u = np.unique(ids)
    # t-statistic for dir_z from the full model with show-clustered SEs.
    full = sm.OLS(m[y], m[Xf.columns]).fit(cov_type="cluster", cov_kwds={"groups": ids})
    b, t0 = full.params["dir_z"], full.tvalues["dir_z"]
    # Residuals and fitted values from the null model (dir_z removed); the bootstrap re-signs these.
    r0 = sm.OLS(m[y], m[c0]).fit()
    res = r0.resid.values
    f0 = r0.fittedvalues.values
    sg = pd.Series(index=u, dtype=float)
    tb = np.empty(B)
    # B bootstrap draws: flip each show's residuals by one coin toss, refit, keep the t for dir_z.
    for i in range(B):
        sg[:] = rng.choice([-1.0, 1.0], len(u))  # one Rademacher coin per show
        tb[i] = (
            sm.OLS(f0 + res * sg.reindex(ids).values, m[Xf.columns])  # rebuild y under the null with flipped show residuals
            .fit(cov_type="cluster", cov_kwds={"groups": ids})
            .tvalues["dir_z"]
        )
    p2 = (np.abs(tb) >= abs(t0)).mean()  # two-sided: |t*| at least as extreme as observed
    p1 = (tb * PRED[y] >= t0 * PRED[y]).mean()  # one-sided in the predicted direction (sign flipped for M3)
    return b, p1, p2, len(m)


print(
    "\n=== SAMPLE A: show-level, register -> Q28 framing (ideology controls + log words; wild cluster bootstrap) ==="
)
# Sample A: strict title matches are confirmatory (n=385); expanded host-name matches are robustness (n=544).
rows = []
for lab, g in (("strict (confirmatory)", st), ("expanded (robustness)", ex)):
    g = g.merge(F, on="resp", how="left")
    for y in MEAS:
        b, p1, p2, n = wild_dir(g, y)
        sd = g[y].std()
        rows.append((lab, y, b, b / sd if sd else np.nan, p1, p2, n))  # b in SD units of the measure
R = pd.DataFrame(rows, columns=["sample", "measure", "b", "b_SD", "p_1sided_predicted", "p_2sided", "n"])
# M1 is judged alone at alpha=.05; M2-M4 form one BH family per sample (two-sided p, as pre-registered).
for lab in R["sample"].unique():
    sub = R[R["sample"] == lab].copy()
    sec = sub[sub.measure != "M1_outgroup"].copy()
    sec["q_bh_secondary"] = multipletests(sec.p_2sided, method="fdr_bh")[1]
    sub = sub.merge(sec[["measure", "q_bh_secondary"]], on="measure", how="left")
    print(f"\n  {lab}")
    print(
        sub.drop(columns="sample").to_string(
            index=False, float_format=lambda x: f"{x:+.4f}" if abs(x) < 10 else f"{x:.0f}"
        )
    )
# Pre-registered robustness subsample: Gallup Panel (probability) respondents only, primary measure only.
gp = st[st.panel].merge(F, on="resp", how="left")
b, p1, p2, n = wild_dir(gp, "M1_outgroup")
print(f"\n  Gallup Panel only, M1: b={b:+.4f}  p1={p1:.4f}  p2={p2:.4f}  n={n}")

print(
    "\n=== SAMPLE B: population, corpus-show / any-podcast namers vs TV-print-web only (weighted, HC1) ==="
)
# Sample B: population test. Anyone typing >= 1 source; 'corpus' = named a corpus show (strict match),
# 'otherpod' = named some other political podcast/host (BIG regex) but no corpus show;
# reference = named only TV/print/web sources.
L = pd.read_csv(INPUTS / "verbatims_long.csv")
strict = set(pd.read_csv(INPUTS / "match_strict.csv").resp)
BIG = r"joe rogan|rogan|daily wire|shapiro|tucker|megyn kelly|charlie kirk|tim pool|hasan|piker|glenn beck|bongino|theo von|lex fridman|pod save|meidas|midas|bulwark|breaking points|young turks|tyt|pakman|crowder|candace|benny johnson|heather cox|matt walsh|knowles|klavan|jordan peterson|sam harris|ezra klein|the daily|up first|npr politics|\bpodcast|\bpod\b"
anypod = set(L[L.v.str.contains(BIG, regex=True, na=False)].resp)
B = K[K.resp.isin(set(L.resp))].merge(F, on="resp", how="inner")  # inner merge keeps only respondents with a usable Q28 answer
B["corpus"] = B.resp.isin(strict).astype(float)
B["otherpod"] = ((~B.resp.isin(strict)) & B.resp.isin(anypod)).astype(float)
rows = []
# One weighted regression per measure: z-scored measure on the two podcast dummies + controls + log
# words, survey weight w, HC1 robust SEs.
for y in MEAS:
    Z = pd.get_dummies(B.pid, prefix="pid", drop_first=True).astype(float)
    Z["attn"] = B.attn
    Z["age"] = B.age
    Z["edu"] = B.edu
    Z["logw"] = B.logw
    Z["corpus"] = B.corpus
    Z["otherpod"] = B.otherpod
    Z = sm.add_constant(Z, has_constant="add")
    yy = (B[y] - B[y].mean()) / B[y].std()  # SD units
    m = pd.concat([Z, yy.rename("y"), B.w], axis=1).dropna()
    r = sm.WLS(m.y, m[Z.columns], weights=m.w).fit(cov_type="HC1")
    rows.append(
        (y, len(m), r.params["corpus"], r.pvalues["corpus"], r.params["otherpod"], r.pvalues["otherpod"])
    )
# BH-correct across the four measures, separately for each podcast dummy.
P = pd.DataFrame(
    rows, columns=["measure", "n", "b_corpus_SD", "p_corpus", "b_otherpod_SD", "p_otherpod"]
)
P["q_corpus"] = multipletests(P.p_corpus, method="fdr_bh")[1]
P["q_otherpod"] = multipletests(P.p_otherpod, method="fdr_bh")[1]
print(P.to_string(index=False, float_format=lambda x: f"{x:+.4f}" if abs(x) < 10 else f"{x:.0f}"))
