"""Two populism lexicons (anti-elite, people-centric) and a media-criticism lexicon scored over the corpus, against address and ideology. Paper section 2.3. Writes inputs/show_populist.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import gzip, csv, re, pandas as pd, numpy as np, scipy.stats as st, warnings

warnings.filterwarnings("ignore")
csv.field_size_limit(10_000_000)
# Two content lexicons that the institutional-distrust literature ties directly to cynicism:
# (a) POPULIST anti-elite / people-centric style (Jagers & Walgrave 2007; Engesser et al. 2017; Ernst et al. 2019)
# (b) MEDIA CRITICISM / hostile-media cues (Ladd 2012; Vallone, Ross & Lepper 1985)
# Each lexicon is one big alternation regex with \b anchors (whole-word matches). ELITE = anti-elite /
# anti-establishment phrases; PEOPLE = appeals to "the people"; MEDIA = hostile-media cues (mainstream /
# legacy / corporate media, fake news, censorship, "the narrative").
ELITE = r"\b(elites?|the establishment|establishment (?:republicans|democrats|politicians|media)|the swamp|deep state|globalists?|the ruling class|career politicians?|political class|the system is rigged|rigged system|the system|insiders?|the powers that be|corrupt(?:ion|ed)? (?:politicians|system|elites|establishment)|bureaucrats?|technocrats?|unelected|out of touch|they don'?t care about you|they think you'?re stupid|the uniparty|uniparty|the machine)\b"
PEOPLE = r"\b(the (?:american )?people|ordinary (?:americans|people|folks)|real americans?|everyday americans?|working (?:people|families|americans)|regular (?:people|folks|americans)|the common man|the little guy|hardworking americans|we the people|the silent majority|the forgotten (?:man|men|americans))\b"
MEDIA = r"\b(mainstream media|the msm|msm|legacy media|corporate media|fake news|the media (?:won'?t|will never|doesn'?t want|is lying|lied)|they'?re lying to you|what they'?re not telling you|they don'?t want you to know|propaganda|the narrative|state media|regime media|media blackout|censor(?:ed|ship|ing)|the press (?:is|are) (?:lying|corrupt)|journalists? (?:are|is) (?:lying|corrupt|activists?))\b"
# Compile case-insensitively; TOK is the word tokenizer used for the per-10k-word denominator.
E = re.compile(ELITE, re.I)
Pp = re.compile(PEOPLE, re.I)
Md = re.compile(MEDIA, re.I)
TOK = re.compile(r"[a-z']+")
rows = []
# Stream all 1.38M passages once (DictReader keeps memory flat). For each passage keep the show id,
# word count, and the number of hits from each of the three lexicons.
with gzip.open(DATA / "output/scoring_chunks.csv.gz", "rt", newline="") as fh:
    for r in csv.DictReader(fh):
        t = r["text"]
        n = len(TOK.findall(t.lower()))
        if n < 40:  # skip very short passages (rates are unstable)
            continue
        rows.append((r["collection_id"], n, len(E.findall(t)), len(Pp.findall(t)), len(Md.findall(t))))
P = pd.DataFrame(rows, columns=["show_id", "nw", "elite", "people", "media"])
# Sum words and hits within show, then convert to rates per 10,000 words. Summing before dividing means
# show rates are automatically weighted by passage length.
Sh = (
    P.groupby("show_id")
    .agg(nw=("nw", "sum"), elite=("elite", "sum"), people=("people", "sum"), media=("media", "sum"))
    .reset_index()
)
for c in ("elite", "people", "media"):
    Sh[c + "_rate"] = Sh[c] / Sh.nw * 1e4
Sh["populist_rate"] = Sh.elite_rate + Sh.people_rate  # combined populism index = anti-elite + people-centric
Sh["show_id"] = Sh.show_id.astype(str)  # string ids so the merge with directive_final.csv lines up
Sh.to_csv(INPUTS / "show_populist.csv", index=False)
# Attach the show-level register score (dir_z), lean, LLM side and DIME host score for the scored shows.
D = pd.read_csv(INPUTS / "directive_final.csv")
D["show_id"] = D.show_id.astype(str)
M = D.merge(Sh, on="show_id")
print(
    f"{len(P):,} passages; {len(M)} shows merged. per-10k rates — anti-elite median {M.elite_rate.median():.1f}, people-centric {M.people_rate.median():.1f}, media-criticism {M.media_rate.median():.1f}"
)
print("\n=== A. show-level correlations with directive register and ideology ===")
# Part A: correlate each lexicon rate with address (dir_z), host DIME score (only shows with a donor
# record), the LLM side label, and the Sobieraj-Berry insult rate.
c = M.dropna(subset=["avg_host_cfscore"])  # DIME correlation only for shows with a host cfscore
for x, lab in (
    ("elite_rate", "anti-elite"),
    ("people_rate", "people-centric"),
    ("populist_rate", "populist (elite+people)"),
    ("media_rate", "media criticism"),
):
    # NOTE: the insult column comes from inputs/show_measures.csv (Sobieraj-Berry outrage rates);
    # rp_test.py reads the same table.
    print(
        f"  {lab:<24} vs dir_z r={st.pearsonr(M.dir_z,M[x])[0]:+.3f} (p={st.pearsonr(M.dir_z,M[x])[1]:.4f}) | vs DIME r={st.pearsonr(c[x],c.avg_host_cfscore)[0]:+.3f} | vs LLM side r={st.pearsonr(M[x],M.side)[0]:+.3f} | vs insult r={st.pearsonr(M[x],pd.read_csv(INPUTS / 'show_measures.csv').set_index('show_id').reindex(M.show_id.astype(int)).insult.values)[0]:+.3f}"
    )
# Sanity check: which shows top each index, with their lean label (names truncated to 20 chars).
t = M.sort_values("populist_rate")
print(
    "  most populist-style:", ", ".join(f"{s[:20]}({l})" for s, l in zip(t.show.tail(5), t.lean.tail(5)))
)
t = M.sort_values("media_rate")
print(
    "  most media-critical :",
    ", ".join(f"{s[:20]}({l})" for s, l in zip(t.show.tail(5), t.lean.tail(5))),
)
