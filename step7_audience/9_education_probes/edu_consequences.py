"""What follows from address -> less-educated audience? (a) indirect paths via education, (b) address x education
moderation, (c) reach: does address bring in low-education listeners who are otherwise disengaged. Pooled maximal match."""
import pandas as pd, numpy as np, re, sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'step7_audience'); from common import cr_all
from statsmodels.stats.multitest import multipletests
z=lambda s:(s-s.mean())/s.std(ddof=0); num=lambda g,c: pd.to_numeric(g[c],errors='coerce').where(lambda s:s>0) if c in g else pd.Series(np.nan,index=g.index)
B=600
S=pd.read_csv('step7_audience/inputs/directive_final.csv')[['show_id','show','inten','lean']]; S['show_id']=S.show_id.astype(str)
M=pd.read_csv('step7_audience/inputs/mfte_show_scores.csv')[['show_id','mfte_z']]; M['show_id']=M.show_id.astype(str); X=S.merge(M,on='show_id'); mu,sd=X.inten.mean(),X.inten.std()
MAIN=r"cnn|fox|msnbc|nbc|abc|cbs|npr|bbc|new york times|nyt|washington post|wapo|wall street|wsj|associated press|\bap\b|reuters|pbs|usa today|local news|newspaper|the hill|politico|axios|bloomberg|the guardian|the economist|the atlantic|time magazine|newsweek|huffpost|yahoo news|google news|apple news"
ITEMS={'Q1':'life ladder now','Q2':'life ladder in 5 yrs','Q3':'income difficulty (higher=worse)','Q10':'attends community events','Q18A':'trust faith leaders','Q18B':'trust business leaders','Q18C':'trust elected officials','Q18D':'trust education officials','Q18E':'trust family/friends','Q19':'political attention','Q21':'social media time','Q22':'people like me valued','Q23':'citizens have power','Q29':'democracy doing well','Q30A':'democracy best form','Q30B':'people committed to democracy','Q30C':'leaders committed to democracy','Q31':'leaders held accountable','Q32C':'election officials acted improperly','Q32H':'rich-poor gap','Q33A':'trust courts','Q33B':'trust Congress','Q33C':'trust presidency','Q33D':'trust people\'s role','Q33E':'trust election administration','Q33F':'trust criminal justice','Q33G':'trust military','Q33H':'trust media (Q33H)','Q34B':'Q34B','Q36A':'people (not govt) responsible for needs','Q36C':'leaders should NOT compromise','Q36E':'hard to get along w/ different beliefs','Q36G':'diversity makes US weaker','Q36K':'violence sometimes okay','Q37':'young people: less likely better life','Q44':'church attendance (higher=never)','Q47':'registered to vote (higher=no)','Q49':'household income','Q4':'lonely (Y1)','Q13':'volunteered (Y1; higher=no)','Q15A':'cannot afford basics (Y1)','Q16':'crime concern (Y1)','Q24A':'voting (Y1)','Q24B':'contacting officials (Y1)','Q24C':'protest (Y1)','Q24D':'campaigning (Y1)','Q24E':'donating (Y1)','Q24F':'town halls (Y1)','Q32B':'extreme people should not vote (Y1)','Q32G':'more power to presidents (Y1)','Q35A':'Q35A (Y1)','n_src':'number of news sources named (0-3)','mainstream':'names a mainstream outlet','elec':'election distrust composite','cyn4':'cynicism composite (4 items)'}
def prep(K):
    K=K.copy(); K['edu']=num(K,'EDU'); K['edu_z']=z(K.edu); K['age']=num(K,'AGE'); K['attn']=num(K,'Q19'); K['w']=pd.to_numeric(K.WEIGHT,errors='coerce')
    q41,q43=num(K,'Q41'),num(K,'Q43'); K['pid']=np.where(q41==1,'R',np.where(q41==2,'D',np.where(q43==2,'leanR',np.where(q43==1,'leanD','I'))))
    e=6-num(K,'Q33E'); f=num(K,'Q32C'); K['elec']=z(((e-e.mean())/e.std()+(f-f.mean())/f.std())/2)
    Z4=pd.DataFrame({c:-z(num(K,c)) for c in ['Q31','Q33E','Q33H','Q34B']}); K['cyn4']=z(Z4.mean(axis=1).where(Z4.notna().sum(axis=1)>=3))
    src=K[['Q17_1','Q17_2','Q17_3']].fillna('').astype(str).replace({'nan':'','-99':'','-98':'','-97':''}); K['n_src']=(src.apply(lambda s:s.str.strip().str.len()>1)).sum(axis=1); K['mainstream']=(src.iloc[:,0]+' '+src.iloc[:,1]+' '+src.iloc[:,2]).str.lower().str.contains(MAIN,regex=True).astype(float)
    for q in ITEMS:
        if q in ('elec','cyn4','n_src','mainstream'): continue
        v=num(K,q)
        if q=='Q47': v=v.where(v<=2)
        if q=='Q13': v=v.where(v<=2)
        K[q+'_z']=z(v) if v.notna().sum()>100 else np.nan
    K['n_src_z']=z(K.n_src); K['mainstream_z']=z(K.mainstream); K['elec_z']=K.elec; K['cyn4_z']=K.cyn4
    K['edu_grp']=pd.cut(K.edu,[0,2,5,9],labels=['HS or less','some college','BA+'])
    return K
def build(K,pairs,year):
    p=pairs.merge(X,on='show_id'); p['inten_z']=(p.inten-mu)/sd; p['right']=(p.lean=='R').astype(float); prim=p.sort_values('show_id').groupby('resp').first().reset_index()[['resp','show_id']]
    R=p.groupby('resp').agg(x=('mfte_z','mean'),inten_z=('inten_z','mean'),share_right=('right','mean')).reset_index().merge(prim,on='resp').merge(K,on='resp'); R['year']=year; return R
K1=prep(pd.read_csv('data/external/kettering/KETTERING_DATA_Y1_WEIGHTED.csv',encoding='cp1252',low_memory=False).reset_index().rename(columns={'index':'resp'})); K2=prep(pd.read_pickle('data/external/kettering/y2/y2_raw.pkl'))
R1=build(K1,pd.read_csv('data/external/kettering/y1_match_maximal.csv',dtype={'show_id':str}),1); R2=build(K2,pd.read_csv('data/external/kettering/y2/y2_match_maximal.csv',dtype={'show_id':str}),2); R2['resp']+=10**6; R=pd.concat([R1,R2],ignore_index=True)
print(f"pooled listeners {len(R)} in {R.show_id.nunique()} shows | education groups: {R.edu_grp.value_counts().to_dict()} | full-survey: Y1 {K1.edu_grp.value_counts(normalize=True).round(2).to_dict()}")
def fit(d,y,x,extra=(),drop=(),B=B):
    d=d.dropna(subset=[y,x]+list(extra)); Xm=pd.get_dummies(d.pid,prefix='pid',drop_first=True).astype(float)
    if 'attn' not in drop and y!='Q19_z': Xm['attn']=d.attn
    if y!='age': Xm['age']=d.age
    if d.share_right.nunique()>1: Xm['share_right']=d.share_right
    if d.year.nunique()>1: Xm['year2']=(d.year==2).astype(float)
    for c in extra: Xm[c]=d[c]
    Xm=pd.concat([pd.Series(1.0,index=d.index,name='const'),Xm],axis=1); Xm['x']=d[x]; m=pd.concat([Xm,d[[y,'show_id']]],axis=1).dropna()
    if len(m)<60 or m.show_id.nunique()<8: return dict(b=np.nan,p2=np.nan,pw=np.nan,N=len(m),G=m.show_id.nunique())
    r=cr_all(m[y].values,m[Xm.columns].values,m.show_id.values,list(Xm.columns).index('x'),B=B); r['N']=len(m); r['G']=m.show_id.nunique(); return r
def coef_of(d,y,x,c,extra=()):
    """coefficient of covariate c in the model of y on x + extra (CR2 p)"""
    d=d.dropna(subset=[y,x,c]+list(extra)); Xm=pd.get_dummies(d.pid,prefix='pid',drop_first=True).astype(float)
    if y!='Q19_z': Xm['attn']=d.attn
    Xm['age']=d.age; Xm['share_right']=d.share_right
    if d.year.nunique()>1: Xm['year2']=(d.year==2).astype(float)
    for e in extra: Xm[e]=d[e]
    Xm[c]=d[c]; Xm=pd.concat([pd.Series(1.0,index=d.index,name='const'),Xm],axis=1); Xm['x']=d[x]; m=pd.concat([Xm,d[[y,'show_id']]],axis=1).dropna()
    r=cr_all(m[y].values,m[Xm.columns].values,m.show_id.values,list(Xm.columns).index(c),B=2); return r['b'],r['p2']
a_edu=fit(R,'edu_z','x',('inten_z',)); print(f"\npath a: address -> education (net one-sidedness) = {a_edu['b']:+.3f} ({a_edu['p2']:.2f}/{a_edu['pw']:.2f})")
pooled=[q for q in ITEMS if '(Y1)' not in ITEMS[q]]; y1only=[q for q in ITEMS if '(Y1)' in ITEMS[q]]
print("\n=== (A) INDIRECT PATH address -> education -> outcome  (pooled 1,200 / 63; address net of one-sidedness) ===")
print(f"{'outcome':<40} {'total':>7} {'p':>9}  {'direct|edu':>10} {'p':>9}  {'edu->y':>7} {'p':>6}  {'indirect a*b':>12} {'% of total':>10}")
rowsA=[]
for q in pooled:
    y=q+'_z'; d=R if q in pooled else R[R.year==1]
    if y not in d or d[y].notna().sum()<100: continue
    t=fit(d,y,'x',('inten_z',)); dr=fit(d,y,'x',('inten_z','edu_z')); be,pe=coef_of(d,y,'x','edu_z',('inten_z',)); ind=a_edu['b']*be
    rowsA.append(dict(item=q,label=ITEMS[q],total=t['b'],p_total=t['p2'],pw_total=t['pw'],direct=dr['b'],p_direct=dr['p2'],pw_direct=dr['pw'],edu_b=be,edu_p=pe,indirect=ind,N=t['N']))
    print(f"{ITEMS[q][:40]:<40} {t['b']:+7.3f} {t['p2']:4.2f}/{t['pw']:4.2f}  {dr['b']:+10.3f} {dr['p2']:4.2f}/{dr['pw']:4.2f}  {be:+7.3f} {pe:6.3f}  {ind:+12.3f} {(100*ind/t['b'] if abs(t['b'])>.02 else np.nan):9.0f}%")
A=pd.DataFrame(rowsA); A['q_total']=multipletests(A.p_total,method='fdr_bh')[1]; A['q_direct']=multipletests(A.p_direct,method='fdr_bh')[1]
print("\n  BH q<.10 on total:",A[A.q_total<.10].label.tolist()); print("  BH q<.10 on direct (net of education):",A[A.q_direct<.10].label.tolist())
print("  education itself predicts (|b|>.10, p<.05):",[(r.label,round(r.edu_b,2)) for r in A.itertuples() if abs(r.edu_b)>.10 and r.edu_p<.05])
A.to_csv('scans/address_edu_indirect_paths.csv',index=False)

print("\n=== (B) MODERATION: does address relate to outcomes differently for less-educated listeners?  simple slopes of address (net one-sidedness) within education group ===")
grps=['HS or less','some college','BA+']
print(f"{'outcome':<40} "+' '.join(f"{g:>22}" for g in grps)+f"  {'interaction x*edu':>18}")
rowsB=[]
for q in pooled+y1only:
    y=q+'_z'; d=R if q in pooled else R[R.year==1]
    if y not in d or d[y].notna().sum()<100: continue
    out=[]; row=dict(item=q,label=ITEMS[q])
    for g in grps:
        r=fit(d[d.edu_grp==g],y,'x',('inten_z',),B=400); out.append(f"{r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f}) n{r['N']}" if not np.isnan(r['b']) else '   --'); row[g+'_b']=r['b']; row[g+'_p']=r['p2']; row[g+'_pw']=r['pw']; row[g+'_N']=r['N']
    dd=d.dropna(subset=[y,'edu_z']).copy(); dd['xe']=dd.x*dd.edu_z; ri=fit(dd,y,'xe',('inten_z','edu_z','x'),B=400); row['inter_b']=ri['b']; row['inter_p']=ri['p2']; row['inter_pw']=ri['pw']
    rowsB.append(row); print(f"{ITEMS[q][:40]:<40} "+' '.join(f"{o:>22}" for o in out)+f"  {ri['b']:+.3f} ({ri['p2']:.2f}/{ri['pw']:.2f})")
Bd=pd.DataFrame(rowsB); Bd['q_inter']=multipletests(Bd.inter_p.fillna(1),method='fdr_bh')[1]; Bd['q_low']=multipletests(Bd['HS or less_p'].fillna(1),method='fdr_bh')[1]
print("\n  interactions at BH q<.10:",Bd[Bd.q_inter<.10][['label','inter_b','inter_p']].values.tolist()); print("  HS-or-less slopes at BH q<.10:",Bd[Bd.q_low<.10][['label','HS or less_b','HS or less_p']].values.tolist())
Bd.to_csv('scans/address_by_education_moderation.csv',index=False)

print("\n=== (C) REACH: are low-education listeners of address-heavy shows otherwise engaged or disengaged? ===")
for K,lab in [(K1,'Y1'),(K2,'Y2')]:
    Rk=R[R.year==(1 if lab=='Y1' else 2)]; lo=K[K.edu<=2]; lo_l=Rk[Rk.edu<=2]; hi_t=lo_l[lo_l.x>=lo_l.x.quantile(2/3)]; lo_t=lo_l[lo_l.x<=lo_l.x.quantile(1/3)]
    nonl=lo[~lo.resp.isin(Rk.resp)]
    def desc(g): return f"attention {g.attn.mean():.2f} | sources named {g.n_src.mean():.2f} | mainstream outlet {g.mainstream.mean():.0%} | registered {(g.Q47_z.notna() & (num(g,'Q47')==1)).sum()/max(num(g,'Q47').where(lambda s:s<=2).notna().sum(),1):.0%} | age {g.age.mean():.0f} | n {len(g)}"
    print(f"  {lab} HS-or-less respondents:  ALL non-listeners     {desc(nonl)}")
    print(f"  {lab}                          corpus listeners      {desc(lo_l)}")
    print(f"  {lab}                          low-address-show      {desc(lo_t)}")
    print(f"  {lab}                          high-address-show     {desc(hi_t)}")
print("\n  within HS-or-less listeners (pooled), address (net one-sidedness) ->")
lo_l=R[R.edu<=2]
for y,lab in [('Q19_z','political attention'),('n_src_z','sources named'),('mainstream_z','names mainstream outlet'),('Q47_z','NOT registered'),('Q10_z','community events'),('Q23_z','citizens have power'),('Q22_z','people like me valued'),('Q21_z','social media time'),('Q1_z','life ladder'),('Q3_z','income difficulty'),('elec_z','election distrust'),('cyn4_z','cynicism')]:
    r=fit(lo_l,y,'x',('inten_z',)); print(f"    {lab:<26} {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f}) N={r['N']} G={r['G']}")
print("  within BA+ listeners (pooled), same:")
hi_l=R[R.edu>=6]
for y,lab in [('Q19_z','political attention'),('n_src_z','sources named'),('mainstream_z','names mainstream outlet'),('Q1_z','life ladder'),('Q3_z','income difficulty'),('elec_z','election distrust'),('cyn4_z','cynicism')]:
    r=fit(hi_l,y,'x',('inten_z',)); print(f"    {lab:<26} {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f}) N={r['N']} G={r['G']}")

print("\n=== (D) SHOW LEVEL: address vs audience education profile (64 shows, listener means, n>=5) ===")
per=R.groupby('show_id').agg(n=('resp','size'),x=('x','first'),inten=('inten_z','first'),edu=('edu','mean'),share_lo=('edu',lambda s:(s<=2).mean()),share_ba=('edu',lambda s:(s>=6).mean()),attn=('attn','mean'),nsrc=('n_src','mean'),main=('mainstream','mean'),right=('share_right','first')).reset_index(); per=per[per.n>=5]
import statsmodels.api as sm
for y,lab in [('edu','mean education'),('share_lo','share HS or less'),('share_ba','share BA+'),('attn','mean attention'),('nsrc','sources named'),('main','share naming mainstream outlet')]:
    m=sm.WLS(per[y],sm.add_constant(per[['x','inten','right']]),weights=np.sqrt(per.n)).fit(cov_type='HC1'); print(f"  {lab:<32} address b={m.params.x:+.3f} (p {m.pvalues.x:.3f}) | one-sided b={m.params.inten:+.3f} (p {m.pvalues.inten:.3f}) | right b={m.params.right:+.3f} | shows {len(per)}")
print(f"  full-survey shares: HS or less Y1 {(K1.edu<=2).mean():.0%} Y2 {(K2.edu<=2).mean():.0%}; BA+ Y1 {(K1.edu>=6).mean():.0%} Y2 {(K2.edu>=6).mean():.0%}; all corpus listeners: HS or less {(R.edu<=2).mean():.0%}, BA+ {(R.edu>=6).mean():.0%}")
print("  top-address-tercile shows: HS-or-less share",f"{per[per.x>=per.x.quantile(2/3)].share_lo.mean():.0%}","| bottom tercile",f"{per[per.x<=per.x.quantile(1/3)].share_lo.mean():.0%}")
