"""Re-measure with validated / syntactic instruments. CSV output, checkpointed."""
import gzip,csv,re,time
import numpy as np, pandas as pd, spacy
csv.field_size_limit(10_000_000)

emfd=pd.read_csv("/tmp/emfd.csv")
EMFD={r.word:(r.care_p,r.fairness_p,r.loyalty_p,r.authority_p,r.sanctity_p) for r in emfd.itertuples()}
mfd2=set()
with open("/tmp/mfd2.dic") as fh:
    for line in fh.read().split("%")[2].strip().splitlines():
        p=line.split("\t")
        if len(p)>=2: mfd2.add(p[0].strip().lower())
LD=[w.strip().lower() for w in open("/tmp/ldnoobw.txt") if w.strip() and " " not in w.strip()]
LD_RX=re.compile(r"\b(?:"+"|".join(re.escape(w) for w in LD)+r")\b",re.I)

W=lambda a: re.compile(r"\b(?:"+"|".join(a)+r")\b",re.I)
OLD={"vulg":W(["damn","hell","shit","fuck","fucking","crap","ass","bastard","bullshit"]),
 "relig":W(["god","lord","jesus","christ","faith","pray","church","evil","moral","sin"]),
 "you":W(["you","your","yours","yourself"]),
 "imper":W(["look","listen","think about it","consider","remember","imagine","ask yourself","understand","realize","let me tell you"]),
 "hedge":W(["maybe","perhaps","might","possibly","probably","seems","apparently","sort of","kind of","i think","i guess","arguably","somewhat"]),
 "cert":W(["definitely","certainly","obviously","clearly","undeniably","absolutely","unquestionably","of course","literally","no question"])}
TOK=re.compile(r"[a-z']+")
POSS={"might","may","could","would","should","can"}
COLS=["episode_id","show_id","chunk_ix","nw","imper_syn","you_syn","modal_poss",
      "emfd_total","emfd_care","emfd_authority","emfd_sanctity","mfd2","vulg_ldn"]+[k+"_rx" for k in OLD]

nlp=spacy.load("en_core_web_sm",disable=["ner","lemmatizer"])

def syn_feats(doc):
    imp=you=modal=0
    for sent in doc.sents:
        for tok in sent:
            if tok.pos_=="VERB" and tok.tag_=="VB" and tok.dep_ in ("ROOT","conj","ccomp","advcl"):
                if any(c.dep_ in ("nsubj","nsubjpass","expl","aux","auxpass") for c in tok.children): continue
                if tok.i>sent.start and sent[tok.i-sent.start-1].text.lower() in ("to","not","never"): continue
                imp+=1
            if tok.pos_=="PRON" and "Person=2" in str(tok.morph): you+=1
            if tok.tag_=="MD" and tok.text.lower() in POSS: modal+=1
    return imp,you,modal

rows=[];meta=[]
with gzip.open("data/output/scoring_chunks.csv.gz","rt",newline="") as fh:
    for r in csv.DictReader(fh):
        t=r["text"]
        if len(t)<120: continue
        rows.append(t); meta.append((r["episode_id"],r["collection_id"],int(r["chunk_ix"])))
print(f"{len(rows):,} passages to process",flush=True)

out=[];t0=time.time()
for i,(doc,txt) in enumerate(zip(nlp.pipe(rows,batch_size=1000,n_process=14),rows)):
    lo=txt.lower(); toks=TOK.findall(lo); n=len(toks)
    if n<40: continue
    imp,you2,modal=syn_feats(doc)
    e=np.zeros(5)
    for w in toks:
        v=EMFD.get(w)
        if v is not None: e+=v
    per=lambda c: 10000.0*c/n
    ep,sh,ix=meta[i]
    out.append((ep,sh,ix,n, per(imp),per(you2),per(modal),
        10000.0*e.sum()/n, 10000.0*e[0]/n, 10000.0*e[3]/n, 10000.0*e[4]/n,
        per(sum(1 for w in toks if w in mfd2)), per(len(LD_RX.findall(txt))),
        *[per(len(OLD[k].findall(txt))) for k in OLD]))
    if i and i%300000==0:
        print(f"  {i:,}  ({time.time()-t0:.0f}s)",flush=True)
        pd.DataFrame(out,columns=COLS).to_csv("/tmp/rm_ckpt.csv",index=False)
d=pd.DataFrame(out,columns=COLS)
d.to_csv("/tmp/remeasured.csv",index=False)
print(f"\nwrote {len(d):,} rows in {time.time()-t0:.0f}s -> /tmp/remeasured.csv",flush=True)
