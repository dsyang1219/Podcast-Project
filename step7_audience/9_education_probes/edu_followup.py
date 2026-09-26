import sys; sys.argv=['x']; exec(open('step7_audience/9_education_probes/edu_consequences.py').read().split('a_edu=fit(')[0])
print("\n=== LIFE LADDER x EDUCATION: robustness ===")
for lab,d in [('pooled',R),('Y1',R[R.year==1]),('Y2',R[R.year==2])]:
    lo=fit(d[d.edu<=2],'Q1_z','x',('inten_z',)); hi=fit(d[d.edu>=6],'Q1_z','x',('inten_z',)); dd=d.dropna(subset=['Q1_z','edu_z']).copy(); dd['xe']=dd.x*dd.edu_z; it=fit(dd,'Q1_z','xe',('inten_z','edu_z','x'))
    print(f"  {lab:<7} HS-or-less {lo['b']:+.3f} ({lo['p2']:.2f}/{lo['pw']:.2f}) N{lo['N']} G{lo['G']} | BA+ {hi['b']:+.3f} ({hi['p2']:.2f}/{hi['pw']:.2f}) N{hi['N']} | interaction {it['b']:+.3f} ({it['p2']:.2f}/{it['pw']:.2f})")
lo=R[R.edu<=2].copy(); lo['inc']=lo.Q49_z; lo['incdiff']=lo.Q3_z
for ex,lab in [(('inten_z','Q49_z'),'+ household income'),(('inten_z','Q3_z'),'+ income difficulty'),(('inten_z','Q49_z','Q3_z','Q21_z'),'+ income, difficulty, social media'),(('inten_z','Q47_z'),'+ registered')]:
    r=fit(lo,'Q1_z','x',ex); print(f"  HS-or-less, life ladder ~ address | one-sided {lab:<36} {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f}) N{r['N']}")
bs=[]; 
for s in lo.show_id.unique():
    r=fit(lo[lo.show_id!=s],'Q1_z','x',('inten_z',),B=2); bs.append((r['b'],r['p2'],s))
bs.sort(); print(f"  LOSO HS-or-less: b in [{bs[0][0]:+.3f}, {bs[-1][0]:+.3f}], CR2 p<.05 in {sum(p<.05 for b,p,s in bs)}/{len(bs)} drops; weakest after dropping {S.set_index('show_id').show.get(bs[0][2],bs[0][2])}")
r=fit(lo,'age','x',('inten_z',)); print(f"  HS-or-less: address -> age (years per SD) {r['b']:+.2f} ({r['p2']:.2f}/{r['pw']:.2f})")
r=fit(lo,'Q2_z','x',('inten_z',)); print(f"  HS-or-less: address -> life ladder in 5 yrs {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
# does the life-ladder bump hold for one-sidedness too, or is it address-specific?
r=fit(lo,'Q1_z','inten_z',('x',)); print(f"  HS-or-less: ONE-SIDEDNESS -> life ladder (net address) {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
# education of the same people: is the life-ladder gap simply that low-edu high-address listeners are older/retired?
lo['emp']=num(lo,'Q48'); print("  HS-or-less by address tercile: age / % not employed (codes 3-4) / life ladder mean:")
lo['terc']=pd.qcut(lo.x,3,labels=['low','mid','high'])
print(lo.groupby('terc').agg(n=('resp','size'),age=('age','mean'),not_emp=('emp',lambda s:(s>=3).mean()),ladder=('Q1','mean'),edu=('edu','mean'),right=('share_right','mean')).round(2).to_string())
r=fit(lo,'Q1_z','x',('inten_z','Q48')) if False else None
lo['notemp']=(lo.emp>=3).astype(float); r=fit(lo,'Q1_z','x',('inten_z','notemp')); print(f"  HS-or-less, life ladder ~ address | one-sided + not employed  {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")

print("\n=== REACH: education profile of audiences by show address tercile (respondent level, all 63 shows) ===")
R['terc']=pd.cut(R.x,[-9,R.x.quantile(1/3),R.x.quantile(2/3),9],labels=['low-address','mid','high-address'])
for lab,d,K in [('Y1',R[R.year==1],K1),('Y2',R[R.year==2],K2),('pooled',R,None)]:
    g=d.groupby('terc').agg(n=('resp','size'),shows=('show_id','nunique'),hs=('edu',lambda s:(s<=2).mean()),ba=('edu',lambda s:(s>=6).mean()),age=('age','mean'),attn=('attn','mean'),right=('share_right','mean'))
    print(f"  {lab}: "+' | '.join(f"{i}: HS-or-less {r.hs:.0%}, BA+ {r.ba:.0%}, age {r.age:.0f}, attention {r.attn:.2f}, n {r.n} ({r.shows} shows)" for i,r in g.iterrows()))
    if K is not None: print(f"       full survey: HS-or-less {(K.edu<=2).mean():.0%}, BA+ {(K.edu>=6).mean():.0%}, age {K.age.mean():.0f}, attention {K.attn.mean():.2f}")
# same split within right-leaning shows only and within left-leaning shows only (so it is not just side)
for side,lab in [(1.0,'right-leaning shows'),(0.0,'left-leaning shows')]:
    d=R[R.share_right==side]; 
    if d.x.nunique()<6: continue
    d=d.copy(); d['t']=pd.qcut(d.x,2,labels=['lower-address half','higher-address half']); g=d.groupby('t').agg(n=('resp','size'),shows=('show_id','nunique'),hs=('edu',lambda s:(s<=2).mean()),ba=('edu',lambda s:(s>=6).mean()))
    print(f"  {lab}: "+' | '.join(f"{i}: HS-or-less {r.hs:.0%}, BA+ {r.ba:.0%}, n {r.n} ({r.shows} shows)" for i,r in g.iterrows()))
# binary outcome: HS-or-less as outcome, address net one-sidedness, CR2
R['hs']=(R.edu<=2).astype(float); R['ba']=(R.edu>=6).astype(float)
for y,lab in [('hs','P(HS or less)'),('ba','P(BA+)')]:
    for yl,d in [('pooled',R),('Y1',R[R.year==1]),('Y2',R[R.year==2])]:
        r=fit(d,y,'x',('inten_z',)); print(f"  {lab} ~ address | one-sided, {yl:<6} {r['b']*100:+.1f} points per SD ({r['p2']:.2f}/{r['pw']:.2f})")
