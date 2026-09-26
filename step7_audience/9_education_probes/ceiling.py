import sys; sys.argv=['x']; exec(open('step7_audience/9_education_probes/edu_consequences.py').read().split('a_edu=fit(')[0])
import statsmodels.api as sm, statsmodels.formula.api as smf
print("=== 1. CEILING: how much of each outcome varies BETWEEN shows at all, net of party/attention/age/education/year? (pooled 1,200 / 63) ===")
print(f"{'outcome':<26} {'between-show var %':>18} {'explained by one-sided':>22} {'by address':>11} {'by both':>8} {'by all 9 address factors':>24}")
I=pd.read_csv('step7_audience/inputs/mfte_imperative_show_scores.csv',dtype={'show_id':str}); Mz=pd.read_csv('step7_audience/inputs/mfte_show_scores.csv',dtype={'show_id':str})[['show_id','vimp_z','pp2_z']]
F=Mz.merge(I[['show_id','youknow_z','VIMP_ACT_z','VIMP_COMM_z','VIMP_MENTAL_z','VIMP_CAUSE_z','VIMP_ASPECT_z','VIMP_DOAUX_z','VIMP_NONE_z']],on='show_id'); fac=[c for c in F.columns if c!='show_id']
y1=pd.read_csv('data/external/kettering/y1_match_maximal.csv',dtype={'show_id':str}); y2=pd.read_csv('data/external/kettering/y2/y2_match_maximal.csv',dtype={'show_id':str}); y2['resp']+=10**6
agg=pd.concat([y1,y2]).merge(F,on='show_id').groupby('resp')[fac].mean().reset_index(); R=R.merge(agg,on='resp',how='left')
rows=[]
for y,lab in [('edu_z','education'),('elec_z','election distrust'),('cyn4_z','cynicism'),('Q1_z','life ladder'),('Q3_z','income difficulty'),('Q21_z','social media time'),('Q19_z','attention'),('Q22_z','people like me valued'),('Q29_z','democracy doing well'),('Q33H_z','trust media'),('Q31_z','leaders accountable'),('age','age')]:
    d=R.dropna(subset=[y,'inten_z','attn','age','edu_z']+fac).copy(); ctrl=['attn','age']+(['edu_z'] if y!='edu_z' else []); ctrl=[c for c in ctrl if c!=y]
    Xc=pd.get_dummies(d.pid,prefix='pid',drop_first=True).astype(float); 
    for c in ctrl: Xc[c]=d[c]
    Xc['year2']=(d.year==2).astype(float); Xc=sm.add_constant(Xc)
    res=sm.OLS(d[y].values,Xc.values).fit().resid; d['res']=res
    # show fixed effects on residual: between-show share = R2 of show dummies (adjusted)
    m_fe=smf.ols('res ~ C(show_id)',d).fit(); between=max(m_fe.rsquared_adj,0)
    def expl(cols):
        m=smf.ols('res ~ '+' + '.join(cols),d).fit(); return max(m.rsquared_adj,0)
    e1,e2,e3,e4=expl(['inten_z']),expl(['x']),expl(['inten_z','x']),expl(['inten_z']+fac)
    rows.append(dict(outcome=lab,between=between,one=e1,addr=e2,both=e3,all9=e4,N=len(d)))
    print(f"{lab:<26} {100*between:>17.1f}% {100*e1:>21.1f}% {100*e2:>10.1f}% {100*e3:>7.1f}% {100*e4:>23.1f}%")
print("  (between-show var % = adjusted R2 of show fixed effects on the residual after controls; a show-level measure cannot exceed it)")
pd.DataFrame(rows).to_csv('scans/between_show_ceiling.csv',index=False)

print("\n=== 2. TIME-MATCHED ADDRESS: address from episodes aired in the 12 months before each wave (Y1 fielded 2025; Y2 Apr-May 2026) ===")
E=pd.read_csv('step7_audience/inputs/ep_dates.csv',dtype={'show_id':str,'episode_id':str}); E=E[E.date.astype(str).str.len()==10]; E['dt']=pd.to_datetime(E.date,errors='coerce')
M=pd.read_csv('step7_audience/inputs/mfte_episode_rates.csv',dtype={'show_id':str,'episode_id':str}).merge(E[['show_id','episode_id','dt']],on=['show_id','episode_id'])
ref=pd.read_csv('data/output/dirz_reference_scale.csv'); ref['show_id']=ref.show_id.astype(str)
def showz(df,minn=5):
    g=df.groupby('show_id').apply(lambda g: pd.Series({'vimp':np.average(g.vimp,weights=g.words),'pp2':np.average(g.pp2,weights=g.words),'n':len(g)}),include_groups=False).reset_index(); g=g[g.n>=minn]
    for c in ('vimp','pp2'): mu,sd=g.loc[g.show_id.isin(ref.show_id),c].agg(['mean',lambda s:s.std(ddof=0)]).values; g[c+'_z']=(g[c]-mu)/sd
    g['addr_t']=g[['vimp_z','pp2_z']].mean(axis=1); return g[['show_id','addr_t','n']]
W={1:('2024-04-01','2025-06-30'),2:('2025-04-01','2026-05-31')}
parts=[]
for yr,(a,b) in W.items():
    g=showz(M[(M.dt>=a)&(M.dt<=b)]); pairs=(y1 if yr==1 else y2).merge(g,on='show_id'); ex=pairs.groupby('resp').addr_t.mean().reset_index(); ex['year']=yr; parts.append(ex)
    print(f"  Y{yr}: window {a}..{b}: {g.show_id.nunique()} shows with >=5 dated episodes; r(time-matched, all-time address) = {g.merge(X,on='show_id')[['addr_t','mfte_z']].corr().iloc[0,1]:.2f}")
T=pd.concat(parts); R=R.merge(T,on=['resp','year'],how='left'); d=R.dropna(subset=['addr_t']); print(f"  listeners with time-matched exposure: {len(d)} / {len(R)} in {d.show_id.nunique()} shows")
for y,lab in [('edu_z','education'),('elec_z','election distrust'),('cyn4_z','cynicism'),('Q1_z','life ladder'),('Q3_z','income difficulty'),('Q21_z','social media'),('Q19_z','attention')]:
    ra=fit(d,y,'x',('inten_z',),B=600); rt=fit(d,y,'addr_t',('inten_z',),B=600); print(f"  {lab:<20} all-time address {ra['b']:+.3f} ({ra['p2']:.2f}/{ra['pw']:.2f}) | time-matched {rt['b']:+.3f} ({rt['p2']:.2f}/{rt['pw']:.2f})  same {len(d)} listeners")
