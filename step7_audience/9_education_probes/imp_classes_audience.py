import sys; sys.argv=['x']; src=open('step7_audience/9_education_probes/edu_consequences.py').read().split('a_edu=fit(')[0]
exec(src)
I=pd.read_csv('step7_audience/inputs/mfte_imperative_show_scores.csv',dtype={'show_id':str}); zc=[c for c in I.columns if c.endswith('_z')]
XI=X.merge(I[['show_id']+zc],on='show_id')
print("show level (194): r with address, with DIME, right-left gap (SD), r with one-sidedness")
D=pd.read_csv('step7_audience/inputs/directive_final.csv')[['show_id','avg_host_cfscore']]; D['show_id']=D.show_id.astype(str); XI=XI.merge(D,on='show_id',how='left')
for c in zc:
    a,b=XI[XI.lean=='R'][c],XI[XI.lean=='L'][c]; gap=(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)
    print(f"  {c:<16} r(address) {XI[c].corr(XI.mfte_z):+.2f} | r(DIME) {XI[c].corr(XI.avg_host_cfscore):+.2f} | R−L gap {gap:+.2f} | r(one-sided) {XI[c].corr(XI.inten):+.2f} | mean rate/10k {I.loc[I.show_id.isin(XI.show_id),c.replace('_z','_rate')].mean():.1f}")
def build2(K,pairs,year):
    p=pairs.merge(XI,on='show_id'); p['inten_z']=(p.inten-mu)/sd; p['right']=(p.lean=='R').astype(float); prim=p.sort_values('show_id').groupby('resp').first().reset_index()[['resp','show_id']]
    agg={'x':('mfte_z','mean'),'inten_z':('inten_z','mean'),'share_right':('right','mean')}; agg.update({c:(c,'mean') for c in zc})
    R=p.groupby('resp').agg(**agg).reset_index().merge(prim,on='resp').merge(K,on='resp'); R['year']=year; return R
R1=build2(K1,pd.read_csv('data/external/kettering/y1_match_maximal.csv',dtype={'show_id':str}),1); R2=build2(K2,pd.read_csv('data/external/kettering/y2/y2_match_maximal.csv',dtype={'show_id':str}),2); R2['resp']+=10**6; R=pd.concat([R1,R2],ignore_index=True)
print(f"\npooled {len(R)} / {R.show_id.nunique()} shows. EDUCATION ~ command class (z), net of one-sidedness; then + ad-free... (b, CR2 p / wild p)")
print(f"{'exposure':<18} {'pooled':>22} {'Year 1':>22} {'Year 2':>22} {'pooled + address total':>24}")
for c in ['x']+zc:
    out=[]
    for d in (R,R[R.year==1],R[R.year==2]):
        r=fit(d,'edu_z',c,('inten_z',)); out.append(f"{r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
    r=fit(R,'edu_z',c,('inten_z','x')) if c!='x' else dict(b=np.nan,p2=np.nan,pw=np.nan); out.append(f"{r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})" if c!='x' else '')
    print(f"{c:<18} "+' '.join(f"{o:>22}" for o in out))
print("\nELECTION DISTRUST ~ command class, net of one-sidedness (pooled):")
for c in ['x']+zc:
    r=fit(R,'elec_z',c,('inten_z',)); print(f"  {c:<18} {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
print("\nHS-or-less share (points per SD) ~ class, net of one-sidedness, pooled:")
R['hs']=(R.edu<=2).astype(float)
for c in ['x']+zc:
    r=fit(R,'hs',c,('inten_z',)); print(f"  {c:<18} {100*r['b']:+.1f} ({r['p2']:.2f}/{r['pw']:.2f})")
print("\nFIXED head-to-head: EDUCATION ~ class + total address (as 'addr') + one-sidedness, pooled")
R['addr']=R.x
for c in zc:
    r=fit(R,'edu_z',c,('inten_z','addr')); ra=coef_of(R,'edu_z',c,'addr',('inten_z',)); print(f"  {c:<16} class {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f}) | address total alongside {ra[0]:+.3f} (p {ra[1]:.2f})")
print("\nall classes together (ACT, COMM, MENTAL, CAUSE, ASPECT, DOAUX, NONE, youknow), education, pooled — CR2 p per term:")
cl=['VIMP_ACT_z','VIMP_COMM_z','VIMP_MENTAL_z','VIMP_CAUSE_z','VIMP_ASPECT_z','VIMP_DOAUX_z','VIMP_NONE_z','youknow_z']
for c in cl:
    b,p=coef_of(R,'edu_z',c,c,tuple(['inten_z']+[k for k in cl if k!=c])) if False else (None,None)
# simpler: fit with exposure c and the other classes as extras
for c in cl:
    r=fit(R,'edu_z',c,tuple(['inten_z']+[k for k in cl if k!=c]),B=400); print(f"  {c:<16} {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
print("\nshow-level correlations among the classes (194):"); print(XI[cl].corr().round(2).to_string())
print("\n'you know' by year, education, net of one-sidedness AND address:")
for yl,d in [('pooled',R),('Y1',R[R.year==1]),('Y2',R[R.year==2])]:
    r=fit(d,'edu_z','youknow_z',('inten_z','addr')); print(f"  {yl:<7} {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
