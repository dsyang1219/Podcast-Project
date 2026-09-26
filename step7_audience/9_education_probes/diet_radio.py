import sys; sys.argv=['x']; exec(open('step7_audience/9_education_probes/edu_consequences.py').read().split('a_edu=fit(')[0])
FL={'radio':r"radio|\bam\b|\bfm\b|iheart|sirius|npr|wnyc|kqed|wbur|wamu|hannity|rush limbaugh|glenn beck|mark levin|dan bongino|clay travis|buck sexton|dave ramsey|michael savage|bbc|talk show",
 'cable_tv':r"fox|cnn|msnbc|newsmax|oan|one america|cnbc|c-span|cspan",'network_tv':r"\babc\b|\bnbc\b|\bcbs\b|nightly news|60 minutes|today show|good morning america|local news|channel \d|\bwgn\b|\bktla\b|\bpbs\b",
 'newspaper':r"times|post|journal|tribune|gazette|herald|newspaper|wall street|wsj|nyt|wapo|guardian|economist|atlantic|politico|the hill|axios|reuters|associated press|\bap\b|bloomberg",
 'youtube':r"youtube|you tube",'social':r"facebook|twitter|\bx\b|tiktok|tik tok|instagram|reddit|threads|truth social|snapchat|telegram|rumble",'podcast_word':r"podcast"}
def flags(K):
    src=K[['Q17_1','Q17_2','Q17_3']].fillna('').astype(str).replace({'nan':'','-99':'','-98':'','-97':''}); t=(src.iloc[:,0]+' | '+src.iloc[:,1]+' | '+src.iloc[:,2]).str.lower()
    for k,p in FL.items(): K[k]=t.str.contains(p,regex=True).astype(float)
    return K
K1=flags(K1); K2=flags(K2); R=pd.concat([R[R.year==1].drop(columns=list(FL),errors='ignore').merge(K1[['resp']+list(FL)],on='resp'),R[R.year==2].drop(columns=list(FL),errors='ignore').assign(resp0=lambda d:d.resp-10**6).merge(K2[['resp']+list(FL)].rename(columns={'resp':'resp0'}),on='resp0').drop(columns='resp0')],ignore_index=True)
print(f"pooled {len(R)} listeners. Share naming each source type: corpus listeners vs full survey (Y1/Y2):")
for k in FL: print(f"  {k:<12} listeners {R[k].mean():.0%} | survey {K1[k].mean():.0%} / {K2[k].mean():.0%}")
print("\nP(also names source type) ~ address, net of one-sidedness (+ controls), pooled | Y1 | Y2  (points per SD, CR2 p / wild p)")
for k in FL:
    out=[]
    for d in (R,R[R.year==1],R[R.year==2]):
        r=fit(d,k,'x',('inten_z',),B=600); out.append(f"{100*r['b']:+5.1f} ({r['p2']:.2f}/{r['pw']:.2f})")
    print(f"  {k:<12} "+' | '.join(out))
print("\nsame for ONE-SIDEDNESS net of address, pooled:")
for k in FL:
    r=fit(R,k,'inten_z',('x',),B=600); print(f"  {k:<12} {100*r['b']:+5.1f} ({r['p2']:.2f}/{r['pw']:.2f})")
print("\nage (years per SD of address, net one-sidedness): pooled/Y1/Y2")
for d in (R,R[R.year==1],R[R.year==2]):
    r=fit(d,'age','x',('inten_z',),B=600); print(f"  {r['b']:+.2f} ({r['p2']:.2f}/{r['pw']:.2f})",end='')
print()
print("\nis the education link explained by ALSO listening to radio? education ~ address | one-sided + radio flag + age:")
for lab,ex in [('baseline',('inten_z',)),('+ names radio',('inten_z','radio')),('+ names radio + cable + youtube + social',('inten_z','radio','cable_tv','youtube','social'))]:
    r=fit(R,'edu_z','x',ex); print(f"  {lab:<42} {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
r=fit(R[R.radio==0],'edu_z','x',('inten_z',)); print(f"  within listeners who name NO radio source ({int((R.radio==0).sum())}): {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
print("\nby show-address tercile (pooled): share naming a newspaper / network TV / YouTube; and share HS-or-less")
R['terc']=pd.cut(R.x,[-9,R.x.quantile(1/3),R.x.quantile(2/3),9],labels=['low','mid','high'])
print(R.groupby('terc').agg(n=('resp','size'),newspaper=('newspaper','mean'),network_tv=('network_tv','mean'),youtube=('youtube','mean'),radio=('radio','mean'),hs=('edu',lambda s:(s<=2).mean())).round(3).to_string())
print("full survey newspaper share: Y1",f"{K1.newspaper.mean():.1%}","Y2",f"{K2.newspaper.mean():.1%}")
r=fit(R,'newspaper','x',('inten_z','edu_z')); print(f"newspaper ~ address | one-sided + EDUCATION: {100*r['b']:+.1f} pts ({r['p2']:.2f}/{r['pw']:.2f})  [is it just education?]")
