import sys; sys.argv=['x']; exec(open('step7_audience/9_education_probes/edu_consequences.py').read().split('a_edu=fit(')[0])
from statsmodels.stats.multitest import multipletests
K1f=pd.read_csv('data/external/kettering/KETTERING_DATA_Y1_WEIGHTED.csv',encoding='cp1252',low_memory=False).reset_index().rename(columns={'index':'resp'}); K2f=pd.read_pickle('data/external/kettering/y2/y2_raw.pkl')
STATES=["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","DC","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming","Puerto Rico"]
SOUTH={"Delaware","DC","Florida","Georgia","Maryland","North Carolina","South Carolina","Virginia","West Virginia","Alabama","Kentucky","Mississippi","Tennessee","Arkansas","Louisiana","Oklahoma","Texas"}
def extra(K,year):
    X=pd.DataFrame({'resp':K.resp})
    X['mail']=(pd.to_numeric(K.MODE,errors='coerce')==2).astype(float)
    st=pd.to_numeric(K['SAMP_TYPE' if year==1 else 'SAMPLE_TYPE'],errors='coerce'); X['optin']=(st==3).astype(float)
    X['black']=(pd.to_numeric(K.RACE_2,errors='coerce')==1).astype(float); X['hispanic']=(pd.to_numeric(K.ETHNICITY,errors='coerce')==1).astype(float)
    X['white_nh']=((pd.to_numeric(K.RACE_1,errors='coerce')==1)&(X.hispanic==0)&(X.black==0)).astype(float)
    X['lgbt']=(K[['Q50_2','Q50_3','Q50_4','Q50_5']].apply(pd.to_numeric,errors='coerce')==1).any(axis=1).astype(float)
    X['female']=(pd.to_numeric(K.GENDER,errors='coerce')==2).astype(float)
    if year==1:
        ru=pd.to_numeric(K.Urban_Rural_Continuum_Code,errors='coerce'); X['rucc']=ru; X['nonmetro']=(ru>=4).astype(float).where(ru.notna())
        X['south']=(pd.to_numeric(K.DEMO_REGION,errors='coerce')==3).astype(float).where(K.DEMO_REGION.astype(str).str.strip()!='')
        X['open_answer']=(~K.Q28.astype(str).str.strip().isin(['-98','','nan'])).astype(float); X['open_len']=np.log1p(K.Q28.astype(str).str.len().where(X.open_answer==1))
    else:
        import pyreadstat; _,meta=pyreadstat.read_sav('data/external/kettering/y2/KETTERING_2026_weighted_final_client.sav',metadataonly=True); vl=meta.variable_value_labels['STATE']; s=pd.to_numeric(K.STATE,errors='coerce'); name=s.map(lambda v: vl.get(v) if pd.notna(v) else None).replace({'District of Columbia':'DC'}); X['south']=name.isin(SOUTH).astype(float).where(name.notna())
        X['open_answer']=(~K.Q64.astype(str).str.strip().isin(['-98','','nan'])).astype(float); X['open_len']=np.log1p(K.Q64.astype(str).str.len().where(X.open_answer==1))
        dur=(pd.to_datetime(K.EndDate)-pd.to_datetime(K.StartDate)).dt.total_seconds()/60; X['log_minutes']=np.log(dur.where((dur>2)&(dur<180)))
    att=[c for c in K.columns if re.match(r'^Q(3[0-4][A-J]?|31|22|23|29)$',c)]; v=K[att].apply(pd.to_numeric,errors='coerce'); X['nonresp']=(v.isna()|(v<0)).sum(axis=1)/len(att)
    return X
E1=extra(K1f,1); E2=extra(K2f,2); E2['resp']+=10**6
R=R.merge(pd.concat([E1,E2]),on='resp',how='left')
print(f"Y2 state check: most common codes {pd.to_numeric(K2f.STATE,errors='coerce').value_counts().head(3).index.tolist()} (expect 5=California, 44=Texas, 10=Florida)")
print(f"\n=== A. never-scanned respondent variables ~ address, net of one-sidedness (pooled 1,200/63 unless Y1/Y2-only) ===")
rows=[]
for v,lab in [('mail','answered by MAIL (not web)'),('optin','opt-in (not panel) sample'),('black','Black'),('hispanic','Hispanic'),('white_nh','white non-Hispanic'),('lgbt','LGBT'),('female','female'),('nonmetro','non-metro county (Y1)'),('rucc','rural-urban code 1-9 (Y1)'),('south','South region'),('open_answer','answered the open-ended item'),('open_len','log length of open answer'),('log_minutes','log interview minutes (Y2)'),('nonresp','share of attitude items skipped')]:
    d=R.dropna(subset=[v]); out=[]
    for yl,dd in [('pooled',d),('Y1',d[d.year==1]),('Y2',d[d.year==2])]:
        if dd[v].nunique()<2 or len(dd)<100: out.append(f"{'--':>22}"); continue
        r=fit(dd,v,'x',('inten_z',),B=600); rows.append(dict(var=v,sample=yl,b=r['b'],p=r['p2'],pw=r['pw'],N=r['N'])); out.append(f"{r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
    print(f"  {lab:<34} "+' | '.join(f"{o:>22}" for o in out)+f"   mean {d[v].mean():.2f}")
A=pd.DataFrame(rows); A['q']=np.nan; m=A['sample']=='pooled'; A.loc[m,'q']=multipletests(A.loc[m,'p'],method='fdr_bh')[1]; print("  pooled BH q<.10:",A[m&(A.q<.10)][['var','b','p','q']].values.tolist())
print("  same, one-sidedness net of address (pooled):",', '.join(f"{v} {fit(R.dropna(subset=[v]),v,'inten_z',('x',),B=400)['b']:+.3f}" for v in ['mail','optin','black','hispanic','white_nh','lgbt','female','south','open_answer','nonresp']))
print("\n  does mail/opt-in explain education? edu ~ address | one-sided:",f"{fit(R,'edu_z','x',('inten_z',))['b']:+.3f}"," + mail + optin:",f"{fit(R,'edu_z','x',('inten_z','mail','optin'))['b']:+.3f}"," web-only:",f"{fit(R[R.mail==0],'edu_z','x',('inten_z',))['b']:+.3f}")

print("\n=== B. address x LISTENER'S OWN PARTY (simple slopes, net of one-sidedness) ===")
R['own']=R.pid.map({'R':'Rep','leanR':'Rep','D':'Dem','leanD':'Dem','I':'Ind'})
for y,lab in [('edu_z','education'),('elec_z','election distrust'),('cyn4_z','cynicism'),('Q1_z','life ladder'),('Q3_z','income difficulty'),('Q19_z','attention'),('Q21_z','social media time'),('Q29_z','democracy doing well'),('Q22_z','people like me valued')]:
    out=[]
    for g in ('Dem','Ind','Rep'):
        r=fit(R[R.own==g],y,'x',('inten_z',),B=400); out.append(f"{g} {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f}) n{r['N']}")
    print(f"  {lab:<22} "+' | '.join(out))
print("  cross-pressure: address among listeners whose named shows lean AGAINST their party (share_right vs own):")
R['cross']=((R.own=='Dem')&(R.share_right>0.5))|((R.own=='Rep')&(R.share_right<0.5)); print(f"    cross-pressured listeners {int(R.cross.sum())}; P(cross-pressured) ~ address: {100*fit(R[R.own!='Ind'],'cross','x',('inten_z',))['b']:+.1f} pts ({fit(R[R.own!='Ind'],'cross','x',('inten_z',),B=400)['p2']:.2f})")

print("\n=== C. multi-show namers ===")
y1=pd.read_csv('data/external/kettering/y1_match_maximal.csv',dtype={'show_id':str}); y2=pd.read_csv('data/external/kettering/y2/y2_match_maximal.csv',dtype={'show_id':str}); y2['resp']+=10**6
P=pd.concat([y1,y2]).merge(X[['show_id','lean','mfte_z']],on='show_id'); g=P.groupby('resp').agg(n_shows=('show_id','nunique'),both=('lean',lambda s:(('L' in set(s)) and ('R' in set(s)))),spread=('mfte_z',lambda s:s.max()-s.min())).reset_index()
R=R.merge(g,on='resp',how='left'); print(f"  listeners naming 2+ corpus shows: {int((R.n_shows>=2).sum())} of {len(R)}; both sides: {int(R.both.sum())}")
print(f"  P(names 2+ corpus shows) ~ address: {100*fit(R,'n_shows','x',('inten_z',),B=400)['b']:+.1f} (shows per SD) p {fit(R,'n_shows','x',('inten_z',),B=400)['p2']:.2f}")
R['both_f']=R.both.astype(float); mm=R[R.n_shows>=2]; r=fit(mm,'both_f','x',('inten_z',),B=400); print(f"  among multi-show namers, P(both sides) ~ address: {100*r['b']:+.1f} pts ({r['p2']:.2f}/{r['pw']:.2f}) n{r['N']}")
r=fit(R[R.n_shows==1],'edu_z','x',('inten_z',)); print(f"  education link among single-show namers only: {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")

print("\n=== D. ALL address factors in one model (education, pooled, net of one-sidedness) ===")
I=pd.read_csv('step7_audience/inputs/mfte_imperative_show_scores.csv',dtype={'show_id':str}); M=pd.read_csv('step7_audience/inputs/mfte_show_scores.csv',dtype={'show_id':str})[['show_id','vimp_z','pp2_z']]
F=M.merge(I[['show_id']+[c for c in I.columns if c.endswith('_z')]],on='show_id'); fac=[c for c in F.columns if c!='show_id' and c not in ('VIMP_OCCUR_z','VIMP_EXIST_z','lets_z')]
P2=pd.concat([y1,y2]).merge(F,on='show_id'); agg=P2.groupby('resp')[fac].mean().reset_index(); R=R.merge(agg,on='resp',how='left')
import statsmodels.api as sm
def joint(y,facs,B=None):
    d=R.dropna(subset=[y]+facs+['inten_z','attn','age']); Xm=pd.get_dummies(d.pid,prefix='pid',drop_first=True).astype(float); Xm['attn']=d.attn; Xm['age']=d.age; Xm['share_right']=d.share_right; Xm['year2']=(d.year==2).astype(float); Xm['inten_z']=d.inten_z
    for f in facs: Xm[f]=d[f]
    m=sm.OLS(d[y].values,sm.add_constant(Xm).values).fit(cov_type='cluster',cov_kwds={'groups':pd.factorize(d.show_id)[0]}); names=['const']+list(Xm.columns)
    return {f:(m.params[names.index(f)],m.pvalues[names.index(f)],np.nan) for f in facs}, m.rsquared
print(f"  controls only R2 {joint('edu_z',[])[1]:.3f}; composite: {joint('edu_z',['x'])[0]['x'][0]:+.2f} (p {joint('edu_z',['x'])[0]['x'][1]:.2f}) R2 {joint('edu_z',['x'])[1]:.3f}   [cluster-robust CR1 p-values, 63 clusters]")
for lab,facs in [('two components',['vimp_z','pp2_z']),('components + you know',['vimp_z','pp2_z','youknow_z']),('all Biber classes + pp2 + you know',['pp2_z','youknow_z','VIMP_ACT_z','VIMP_COMM_z','VIMP_MENTAL_z','VIMP_CAUSE_z','VIMP_ASPECT_z','VIMP_DOAUX_z','VIMP_NONE_z'])]:
    out,r2=joint('edu_z',facs); print(f"  {lab} (R2 {r2:.3f}): "+', '.join(f"{f.replace('VIMP_','').replace('_z','')} {b:+.2f} ({p:.2f}/{pw:.2f})" for f,(b,p,pw) in out.items()))
for y,lab in [('elec_z','election distrust'),('Q1_z','life ladder'),('Q3_z','income difficulty'),('Q21_z','social media time'),('Q19_z','attention')]:
    out,r2=joint(y,['vimp_z','pp2_z','youknow_z']); print(f"  {lab}: "+', '.join(f"{f.replace('_z','')} {b:+.2f} ({p:.2f}/{pw:.2f})" for f,(b,p,pw) in out.items()))
R.to_pickle('step7_audience/outputs/R_pooled_full.pkl')
print("\n=== E. check on the Democrat-listener election-distrust slope ===")
dem=R[R.own=='Dem']; bs=[]
for s in dem.show_id.unique():
    r=fit(dem[dem.show_id!=s],'elec_z','x',('inten_z',),B=2); bs.append((r['b'],r['p2'],s))
bs.sort(); nm=S.set_index('show_id').show.to_dict(); print(f"  LOSO among Democrats: b in [{bs[0][0]:+.3f}, {bs[-1][0]:+.3f}]; CR2 p<.05 in {sum(p<.05 for b,p,s in bs)}/{len(bs)} drops; weakest without {nm.get(bs[0][2],bs[0][2])}, strongest without {nm.get(bs[-1][2],bs[-1][2])}")
for yl in (1,2):
    r=fit(dem[dem.year==yl],'elec_z','x',('inten_z',),B=600); print(f"  Y{yl} Democrats: {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f}) n{r['N']}")
top=dem.groupby('show_id').agg(n=('resp','size'),x=('x','first'),elec=('elec_z','mean')).sort_values('n',ascending=False).head(8); top['show']=top.index.map(nm); print(top.round(2).to_string())
