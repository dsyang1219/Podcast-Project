"""Decompose imperatives and second person into subtypes (epistemic / attention / action; deontic / addressive / filler) on a stratified corpus sample. Paper section 2.2. Writes inputs/show_imp_decomp.csv."""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # step7_audience/: paths.py, common.py
from paths import ROOT, DATA, HANDOFF, HERE, INPUTS, OUTPUTS
import gzip, csv, re, random, collections, pandas as pd, numpy as np, spacy, scipy.stats as st, warnings

warnings.filterwarnings("ignore")
csv.field_size_limit(10_000_000)  # passages can be long
random.seed(3)  # reproducible sample
# stratified sample: up to 600 passages per show -> ~110k passages
by = collections.defaultdict(list)
with gzip.open(DATA / "output/scoring_chunks.csv.gz", "rt", newline="") as fh:
    for r in csv.DictReader(fh):
        if len(r["text"]) >= 120:  # skip very short passages
            by[r["collection_id"]].append(r["text"])
sample = [(s, t) for s, ts in by.items() for t in (random.sample(ts, 600) if len(ts) > 600 else ts)]  # cap at 600 passages per show so big shows do not dominate
print(f"{len(sample):,} passages from {len(by)} shows", flush=True)
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])  # parser gives POS/dependency tags; NER not needed here
TOK = re.compile(r"[a-z']+")
# imperative verb buckets (what is the listener being told to do?)
EPIST = {
    "understand",
    "realize",
    "know",
    "think",
    "remember",
    "consider",
    "imagine",
    "notice",
    "recognize",
    "believe",
    "ask",
    "wonder",
    "forget",
    "bear",
    "keep",
    "trust",
    "picture",
    "assume",
    "note",
    "realise",
}  # tell the listener how to think / what is true
ATTN = {
    "look",
    "listen",
    "see",
    "watch",
    "hear",
    "check",
    "wait",
    "hold",
    "stop",
    "hang",
    "guess",
    "mark",
    "behold",
}  # attention-getters / discourse markers
ACTION = {
    "go",
    "get",
    "call",
    "vote",
    "donate",
    "subscribe",
    "share",
    "sign",
    "join",
    "buy",
    "support",
    "fight",
    "stand",
    "take",
    "make",
    "do",
    "give",
    "send",
    "read",
    "visit",
    "use",
    "try",
    "help",
    "tell",
    "write",
    "download",
    "follow",
    "click",
    "order",
    "pay",
    "come",
    "let",
    "put",
    "start",
    "turn",
    "show",
    "find",
    "bring",
    "leave",
    "pick",
    "stay",
    "run",
    "push",
    "demand",
    "protest",
    "register",
    "contact",
}  # do something
# Counters: imp = verb counts overall; imp_side = by show lean; you = second-person subtypes; per = per-show counts (for rates).
imp = collections.Counter()
imp_side = collections.defaultdict(collections.Counter)
you = collections.Counter()
per = collections.defaultdict(lambda: collections.Counter())
# show_id -> LLM lean label (L/R).
side = dict(
    zip(
        pd.read_csv(INPUTS / "directive_final.csv").show_id.astype(str),
        pd.read_csv(INPUTS / "directive_final.csv").lean,
    )
)
# Parse all sampled passages in parallel (10 processes).
texts = [t for _, t in sample]
shows = [s for s, _ in sample]
for i, doc in enumerate(nlp.pipe(texts, batch_size=800, n_process=10)):
    s = shows[i]
    n = len(TOK.findall(texts[i].lower()))
    per[s]["nw"] += n  # words in the passage, for per-10k rates
    for sent in doc.sents:
        toks = list(sent)
        # Imperative detection: a base-form verb (VB) heading a clause, with no subject and no auxiliary, not preceded by 'to'/'not'/'never'.
        for k, tok in enumerate(toks):
            if (
                tok.pos_ == "VERB"
                and tok.tag_ == "VB"
                and tok.dep_ in ("ROOT", "conj", "ccomp", "advcl")
            ):
                if any(c.dep_ in ("nsubj", "nsubjpass", "expl", "aux", "auxpass") for c in tok.children):  # has a subject or auxiliary -> not an imperative
                    continue
                if k > 0 and toks[k - 1].text.lower() in ("to", "not", "never"):  # infinitives ('to go') and bare negations ('never go') excluded
                    continue
                v = tok.text.lower()
                imp[v] += 1
                imp_side[side.get(s, "?")][v] += 1
                # Sort the verb into a bucket: epistemic / attention / action / other.
                b = (
                    "epistemic"
                    if v in EPIST
                    else "attention" if v in ATTN else "action" if v in ACTION else "other"
                )
                per[s]["imp_" + b] += 1
                per[s]["imp_all"] += 1
            # Second-person pronoun (you/your...): classify by the neighbouring words.
            if tok.pos_ == "PRON" and "Person=2" in str(tok.morph):  # spaCy morphology marks second person
                nxt = toks[k + 1].text.lower() if k + 1 < len(toks) else ""
                prv = toks[k - 1].text.lower() if k > 0 else ""
                if nxt == "know":
                    kind = "you_know"  # filler
                elif nxt in (
                    "need",
                    "have",
                    "should",
                    "must",
                    "gotta",
                    "got",
                    "ought",
                    "better",
                    "want",
                ):
                    kind = "you_deontic"  # you need to / should
                elif prv in (
                    "tell",
                    "telling",
                    "told",
                    "promise",
                    "assure",
                    "warn",
                    "guarantee",
                    "remind",
                ) or (
                    prv == "to"
                    and k > 1
                    and toks[k - 2].text.lower() in ("say", "tell", "talking", "speaking")
                ):
                    kind = "you_addressive"  # I tell you / talking to you
                else:
                    kind = "you_other"
                per[s][kind] += 1
                per[s]["you_all"] += 1
                you[kind] += 1
    if i and i % 40000 == 0:  # progress
        print(f"  {i:,}", flush=True)
# Descriptives: the most common imperative verbs, the share in each bucket, and the top verbs by side.
print("\n=== what the imperatives ARE (top 30 verbs, whole sample) ===")
tot = sum(imp.values())
print("  " + ", ".join(f"{v} {c/tot:.1%}" for v, c in imp.most_common(30)))
for b, S in (("epistemic", EPIST), ("attention", ATTN), ("action", ACTION)):
    print(f"  bucket {b:<10} = {sum(imp[v] for v in S)/tot:.1%} of imperatives")
print("\n=== top imperatives by side ===")
for sd in ("L", "R"):
    t = sum(imp_side[sd].values())
    print(f"  {sd}: " + ", ".join(f"{v} {c/t:.1%}" for v, c in imp_side[sd].most_common(15)))
print("\n=== what the second-person IS ===")
t = sum(you.values())
print("  " + ", ".join(f"{k} {c/t:.1%}" for k, c in you.most_common()))
# Per-show table: counts -> rates per 10,000 words, saved for the paper's section 2.2 tables.
S = pd.DataFrame.from_dict(per, orient="index").fillna(0)
S.index.name = "show_id"
S = S.reset_index()
for c in [c for c in S.columns if c.startswith(("imp_", "you_"))]:
    S[c + "_rate"] = S[c] / S.nw * 1e4
S.to_csv(INPUTS / "show_imp_decomp.csv", index=False)
print("\nsaved show_imp_decomp.csv")
# Which subtype carries the ideological signal? Correlate each show-level rate with DIME (hosts with a donor score), LLM side, and dir_z.
D = pd.read_csv(INPUTS / "directive_final.csv")
D["show_id"] = D.show_id.astype(str)
M = D.merge(S, on="show_id")
c = M.dropna(subset=["avg_host_cfscore"])
print("\n=== which SUBTYPE tracks ideology? (show level) ===")
for x in (
    "imp_epistemic_rate",
    "imp_attention_rate",
    "imp_action_rate",
    "imp_other_rate",
    "you_know_rate",
    "you_deontic_rate",
    "you_addressive_rate",
    "you_other_rate",
):
    print(
        f"  {x:<22} vs DIME r={st.pearsonr(c[x],c.avg_host_cfscore)[0]:+.3f} (p={st.pearsonr(c[x],c.avg_host_cfscore)[1]:.3f})   vs LLM side r={st.pearsonr(M[x],M.side)[0]:+.3f}   vs dir_z r={st.pearsonr(M[x],M.dir_z)[0]:+.3f}"
    )
