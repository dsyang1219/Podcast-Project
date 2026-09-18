"""Strategy / game-frame lexicon (Aalberg, Stromback & de Vreese 2012) scored over the corpus, and its relation to address, ideology and the audience outcomes. Paper section 2.1 and finding 6 qualification."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import gzip, csv, re, pandas as pd, numpy as np, scipy.stats as st, statsmodels.api as sm, warnings

warnings.filterwarnings("ignore")
csv.field_size_limit(10_000_000)  # passages can be long; lift the csv module's default field cap
# Strategy / game-frame lexicon after Aalberg, Strömbäck & de Vreese (2012): two subframes.
# HORSE = horse-race vocabulary (polls, ahead/behind, frontrunner, battleground, win/lose...).
HORSE = r"\b(polls?|polling|pollster|approval rating|favorab(?:le|ility)|ahead|behind|lead(?:s|ing)?|trail(?:s|ing)|frontrunner|front-runner|momentum|surg(?:e|ing)|slump|swing state|battleground|electoral (?:map|college|votes?)|margin(?:s)?|turnout|primary|primaries|ticket|race|horse race|who(?:'s| is) (?:winning|going to win|ahead)|win(?:s|ning)?|los(?:e|es|ing)|victory|defeat|landslide|upset|neck and neck|dead heat|toss-?up)\b"
# STRAT = strategy vocabulary (tactics, messaging, spin, optics, donors, playing to the base...).
STRAT = r"\b(strateg(?:y|ic|ist|ists)|tactic(?:s|al)?|messaging|optics|playbook|spin|narrative|framing|gaffe|attack ad|campaign(?:s|ing)?|fundrais(?:ing|er)|war chest|donors?|endorse(?:d|ment|ments)?|consultants?|operatives?|base|the establishment|positioning|pivot(?:ed|ing)?|calculat(?:ed|ion|ing)|political(?:ly)? (?:motivated|calculated|expedient)|maneuver(?:ing)?|play(?:s|ed|ing)? (?:to|for) (?:the|his|her|their) base)\b"
H = re.compile(HORSE, re.I)  # case-insensitive
S = re.compile(STRAT, re.I)
TOK = re.compile(r"[a-z']+")  # word tokeniser for word counts
# Pass 1: count lexicon hits in every corpus passage (1.38M x 750 chars), keeping the show id and word count.
rows = []
with gzip.open(DATA / "output/scoring_chunks.csv.gz", "rt", newline="") as fh:
    for r in csv.DictReader(fh):
        t = r["text"]
        n = len(TOK.findall(t.lower()))
        if n < 40:  # skip passages under 40 words (same rule as the register scoring)
            continue
        rows.append((r["collection_id"], n, len(H.findall(t)), len(S.findall(t))))  # (show, words, horse-race hits, strategy hits)
P = pd.DataFrame(rows, columns=["show_id", "nw", "horse", "strat"])
print(f"{len(P):,} passages, {P.show_id.nunique()} shows")
# Aggregate to shows: rates per 10,000 words, so a show's rate is weighted by how much text it contributed.
Sh = (
    P.groupby("show_id")
    .agg(nw=("nw", "sum"), horse=("horse", "sum"), strat=("strat", "sum"))
    .reset_index()
)
Sh["horse_rate"] = Sh.horse / Sh.nw * 1e4
Sh["strat_rate"] = Sh.strat / Sh.nw * 1e4
Sh["game_rate"] = Sh.horse_rate + Sh.strat_rate  # combined game frame = horse race + strategy
Sh["show_id"] = Sh.show_id.astype(str)
# Attach show-level register (dir_z) and ideology (side, avg_host_cfscore); save the show table for hr_infer.py.
D = pd.read_csv(INPUTS / "directive_final.csv")
D["show_id"] = D.show_id.astype(str)
M = D.merge(Sh, on="show_id")
print(
    f"merged {len(M)} shows | game-frame rate per 10k words: median {M.game_rate.median():.1f}, IQR [{M.game_rate.quantile(.25):.1f},{M.game_rate.quantile(.75):.1f}]"
)
Sh.to_csv(INPUTS / "show_gameframe.csv", index=False)
# A. Does address (dir_z) simply track game framing across the 194 shows? Pearson and Spearman correlations.
print("\n=== A. Is directive register the same thing as horse-race / strategy framing? (show level) ===")
for c, lab in (
    ("horse_rate", "horse-race subframe"),
    ("strat_rate", "strategy subframe"),
    ("game_rate", "combined game frame"),
):
    r, p = st.pearsonr(M.dir_z, M[c])
    rho, _ = st.spearmanr(M.dir_z, M[c])
    print(f"  dir_z vs {lab:<22} r={r:+.3f} (p={p:.4f})  rho={rho:+.3f}")
# B. Is game framing itself partisan? Correlate with the DIME donor score (hosts who have one) and the LLM side label.
print("\n=== B. Is game framing partisan? ===")
c = M.dropna(subset=["avg_host_cfscore"])
for x, lab in (("game_rate", "game frame"), ("horse_rate", "horse race"), ("strat_rate", "strategy")):
    print(
        f"  {lab:<12} vs DIME r={st.pearsonr(c[x],c.avg_host_cfscore)[0]:+.3f}  vs LLM side r={st.pearsonr(M[x],M.side)[0]:+.3f}"
    )
# C. Respondent level: put address and game framing side by side in the audience models on the strict Kettering match.
print(
    "\n=== C. Does game framing explain the directive -> audience results? (Kettering strict, ideology controls) ==="
)
from common import *  # loads st (strict frame), design(), etc.

# Exclusivity outcome: 1 if none of the respondent's named sources is a mainstream outlet.
IDEO = ("share_right", "rw_plat")
L = pd.read_csv(INPUTS / "verbatims_long.csv")
MAIN = r"cnn|fox|nbc|abc|cbs|msnbc|npr|pbs|bbc|new york times|nyt|washington post|wall street|wsj|reuters|associated press|\bap\b|usa today|local news|newspaper|the hill|politico|axios|bloomberg|cnbc|news ?nation|newsmax"
L["main"] = L.v.str.contains(MAIN, regex=True, na=False)
R = L.groupby("resp").agg(any_main=("main", "max")).reset_index()
# Attach the show's game-frame rate, standardised on the 194-show corpus distribution so game_z is on the same footing as dir_z.
g = st.copy()
g = g.merge(R, on="resp")
g["excl"] = 1 - g.any_main
g["show_id"] = g.show_id.astype(str)
g = g.merge(Sh[["show_id", "game_rate"]], on="show_id", how="left")
g["game_z"] = (g.game_rate - M.game_rate.mean()) / M.game_rate.std()
# Drop NBC-branded shows (title-match ambiguity with the network) from the respondent frame.
nb = D[D.show.str.contains("NBC")].show_id.astype(str)
g = g[~g.show_id.isin(nb)]


# OLS with the standard controls + ideology, plus the chosen exposure column(s); CR1 show-clustered SEs.
def fit(y, extra):
    X = design(g, IDEO)
    for e in extra:
        X[e] = g[e]
    m = pd.concat([X, g[[y, "show_id"]]], axis=1).dropna()
    r = sm.OLS(m[y], m[X.columns]).fit(cov_type="cluster", cov_kwds={"groups": m.show_id})
    return r, len(m)


# Horse race: address alone, game frame alone, both together. Does game framing absorb the address effect?
for y, lab in (("inst_cyn", "institutional cynicism"), ("excl", "audience exclusivity")):
    r1, n = fit(y, ["dir_z"])
    r2, _ = fit(y, ["game_z"])
    r3, _ = fit(y, ["dir_z", "game_z"])
    print(f"  {lab} (n={n}):")
    print(f"    directive alone      b={r1.params['dir_z']:+.3f} p={r1.pvalues['dir_z']:.4f}")
    print(f"    game frame alone     b={r2.params['game_z']:+.3f} p={r2.pvalues['game_z']:.4f}")
    print(
        f"    both: directive      b={r3.params['dir_z']:+.3f} p={r3.pvalues['dir_z']:.4f}   game frame b={r3.params['game_z']:+.3f} p={r3.pvalues['game_z']:.4f}"
    )
print("\n(analytic clustered p shown for screening; treat as ~1.5x optimistic per the CR2 results)")
