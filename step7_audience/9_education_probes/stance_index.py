import sys; sys.argv=['x']; exec(open('step7_audience/9_education_probes/edu_consequences.py').read().split('a_edu=fit(')[0])
import statsmodels.api as sm
I=pd.read_csv('step7_audience/inputs/mfte_imperative_show_scores.csv',dtype={'show_id':str}); Mz=pd.read_csv('step7_audience/inputs/mfte_show_scores.csv',dtype={'show_id':str})[['show_id','vimp_z','pp2_z']]
F=Mz.merge(I[['show_id','youknow_z','VIMP_ACT_z','VIMP_COMM_z','VIMP_MENTAL_z','VIMP_CAUSE_z','VIMP_ASPECT_z','VIMP_DOAUX_z','VIMP_NONE_z']],on='show_id'); fac=[c for c in F.columns if c!='show_id']
F['doing_z']=z(F[['VIMP_ACT_z','VIMP_CAUSE_z','VIMP_ASPECT_z','VIMP_DOAUX_z','VIMP_NONE_z']].mean(axis=1)); F['thinking_z']=z(F[['VIMP_MENTAL_z','VIMP_COMM_z']].mean(axis=1))
y1=pd.read_csv('data/external/kettering/y1_match_maximal.csv',dtype={'show_id':str}); y2=pd.read_csv('data/external/kettering/y2/y2_match_maximal.csv',dtype={'show_id':str}); y2['resp']+=10**6
agg=pd.concat([y1,y2]).merge(F,on='show_id').groupby('resp')[fac+['doing_z','thinking_z']].mean().reset_index(); R=R.merge(agg,on='resp',how='left')
print("=== 3. GUIDE-vs-PEER STANCE INDEX, built on YEAR 1 only, frozen, tested on YEAR 2 ===")
R1,R2=R[R.year==1].dropna(subset=['edu_z','attn','age','inten_z','share_right']+fac),R[R.year==2].dropna(subset=['edu_z','attn','age','inten_z','share_right']+fac)
def design(d,cols):
    Xm=pd.get_dummies(d.pid,prefix='pid',drop_first=True).astype(float); Xm['attn']=d.attn; Xm['age']=d.age; Xm['share_right']=d.share_right; Xm['inten_z']=d.inten_z
    for c in cols: Xm[c]=d[c]
    return sm.add_constant(Xm)
# (a) simple a-priori index: doing commands minus "you know" (equal weights) — no fitting
R['stance_simple']=z(R.doing_z-R.youknow_z)
# (b) Year-1-fitted weights on (vimp, pp2, youknow) -> frozen linear index
m1=sm.OLS(R1.edu_z.values,design(R1,['vimp_z','pp2_z','youknow_z']).values).fit(); names=list(design(R1,['vimp_z','pp2_z','youknow_z']).columns); w={c:m1.params[names.index(c)] for c in ['vimp_z','pp2_z','youknow_z']}
print(f"  Year-1 weights (education per SD): vimp {w['vimp_z']:+.3f}, pp2 {w['pp2_z']:+.3f}, you know {w['youknow_z']:+.3f}")
R['stance_fit']=z(sum(R[c]*w[c] for c in w)*-1)   # sign so that higher = more guide-stance (less education)
print(f"  r(stance_simple, address) = {R.stance_simple.corr(R.x):+.2f}; r(stance_fit, address) = {R.stance_fit.corr(R.x):+.2f}; r(simple, fit) = {R.stance_simple.corr(R.stance_fit):+.2f}")
print(f"\n  {'exposure':<34} {'Y1 (fit sample for b)':>24} {'Y2 (out of sample)':>24} {'pooled':>24}")
for c,lab in [('x','address composite (vimp+pp2)'),('stance_simple','doing commands minus you know'),('stance_fit','Y1-fitted index (vimp,pp2,you know)'),('doing_z','doing commands only'),('thinking_z','thinking/communication commands'),('youknow_z','you know only')]:
    out=[]
    for d in (R[R.year==1],R[R.year==2],R):
        r=fit(d,'edu_z',c,('inten_z',),B=800); out.append(f"{r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
    print(f"  {lab:<34} "+' '.join(f"{o:>24}" for o in out))
R2=R.loc[R2.index]
print("\n  out-of-sample fit on Year 2 (adjusted R2 of education model with controls + one-sidedness + exposure):")
for c,lab in [(None,'controls only'),('x','address composite'),('stance_simple','doing minus you know'),('stance_fit','Y1-fitted index')]:
    d=R2.copy(); cols=[c] if c else []; m=sm.OLS(d.edu_z.values,design(d,cols).values).fit(); print(f"    {lab:<24} adj R2 {m.rsquared_adj:.3f}")
print("\n  Y2 head-to-head: education ~ stance_simple + address composite + one-sidedness:")
d=R2.copy(); r=fit(d,'edu_z','stance_simple',('inten_z','x'),B=800); r2=fit(d,'edu_z','x',('inten_z','stance_simple'),B=800); print(f"    stance {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f}) | address alongside {r2['b']:+.3f} ({r2['p2']:.2f}/{r2['pw']:.2f})")
print("\n  stance index vs other Y2 outcomes (should be null if it is composition-specific):")
for y,lab in [('elec_z','election distrust'),('cyn4_z','cynicism'),('Q1_z','life ladder'),('Q19_z','attention')]:
    d=R[R.year==2]; r=fit(d,y,'stance_simple',('inten_z',),B=400); print(f"    {lab:<20} {r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
