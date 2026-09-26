import pyreadstat, glob, re, sys, warnings, pandas as pd, numpy as np; warnings.filterwarnings('ignore'); sys.path.insert(0,'step7_audience'); from common import cr_all
from scipy import stats; import statsmodels.formula.api as smf
S=pd.read_csv('step7_audience/inputs/mfte_show_scores.csv',dtype={'show_id':str}); D=pd.read_csv('step7_audience/inputs/directive_final.csv',dtype={'show_id':str})[['show_id','inten','side']]; ID=pd.read_csv('step7_audience/inputs/inten_distilled.csv',dtype={'show_id':str})[['show_id','inten_use']]
S=S.merge(D,on='show_id',how='left').merge(ID,on='show_id',how='left'); S['inten_any']=S.inten.fillna(S.inten_use); mu,sd=D.inten.mean(),D.inten.std(); S['inten_z']=(S.inten_any-mu)/sd
A=S.set_index('show_id')
print("=== 1. PEW W165 (Mar 2025, n 9,482): regular audiences of named sources vs OUR address score ===")
MAP={'The Joe Rogan Experience':'HOLDOUT_rogan','Tucker Carlson Network':'HOLDOUT_tucker','The Daily Wire':'HOLDOUT_shapiro','Breitbart':'1592116227','NPR':'1057255460','Fox News':'1303660358','NBC News':'75462585','The New York Times':'HOLDOUT_thedaily','The Atlantic':'1258635512','Politico':'1480605295'}
NOTE={'The Daily Wire':'Ben Shapiro Show (flagship)','Breitbart':'Breitbart News Daily','NPR':'NPR Politics Podcast','Fox News':'Fox News Rundown','NBC News':'NBC Nightly News podcast','The New York Times':'The Daily','The Atlantic':'Radio Atlantic','Politico':'POLITICO Energy (weak proxy)'}
T=pd.read_csv('data/output/pew_w165_source_audiences.csv'); T['show_id']=T.source.map(MAP); T=T.dropna(subset=['show_id']); T['address']=T.show_id.map(A.mfte_z); T['one_sided']=T.show_id.map(A.inten_z); T['side']=T.show_id.map(A.side); T['proxy']=T.source.map(NOTE).fillna('')
print(T[['source','proxy','n','ba_w','hs_or_less_w','rep_w','address','one_sided']].round(2).sort_values('address').to_string(index=False))
for lab,d in [('all matched',T),('without POLITICO Energy proxy',T[T.source!='Politico'])]:
    r,p=stats.spearmanr(d.address,d.ba_w); r2,p2=stats.pearsonr(d.address,d.ba_w); rh,ph=stats.spearmanr(d.address,d.hs_or_less_w)
    m=smf.ols('ba_w ~ address + rep_w',d).fit(); m2=smf.ols('ba_w ~ address + one_sided',d.dropna(subset=['one_sided'])).fit()
    print(f"  {lab} (n={len(d)}): Spearman r(address, BA+ share) = {r:+.2f} (p {p:.2f}); Pearson {r2:+.2f} (p {p2:.2f}); r(address, HS-or-less share) = {rh:+.2f}; address net of Republican share b={100*m.params.address:+.1f} pts/SD (p {m.pvalues.address:.2f}); net of one-sidedness b={100*m2.params.address:+.1f} (p {m2.pvalues.address:.2f})")
print("  side validation: r(our side score, Pew Republican share) =",f"{stats.pearsonr(T.dropna(subset=['side']).side,T.dropna(subset=['side']).rep_w)[0]:+.2f} over {T.side.notna().sum()} sources")

print("\n=== 2. PEW W118 (Dec 2022, n 5,132): 'podcast you listen to most' -> felt connection to host, by OUR address ===")
f=glob.glob('data/external/pew_w118/W118_Dec22/*.sav')[0]; d,m=pyreadstat.read_sav(f); vl=m.variable_value_labels['PODMAIN_CODES_W118']; d['podmain']=d.PODMAIN_CODES_W118.map(vl)
PM={'The Joe Rogan Experience':'HOLDOUT_rogan','The Daily (NYT)':'HOLDOUT_thedaily','The Dan Bongino Show':'HOLDOUT_bongino','The Ben Shapiro Show':'HOLDOUT_shapiro','Pod Save America':'1192761536','Timcast IRL':'1362265400','Breaking Points with Krystal and Saagar':'1570045623','The Glenn Beck Program':'HOLDOUT_beck','The Mark Levin Show':'1818658022','Mea Culpa with Michael Cohen':'1714009198','Daily Wire (general reference)':'HOLDOUT_shapiro','Morning Wire':'HOLDOUT_shapiro'}
d['show_id']=d.podmain.map(PM); P=d.dropna(subset=['show_id']).copy(); P['address']=P.show_id.map(A.mfte_z); P['one_sided']=P.show_id.map(A.inten_z)
v=lambda s: s.where(s<99); P['conn']=6-v(P.PODMHOST_W118)   # 1 not at all .. 5 extremely
P['conn_hi']=(v(P.PODMHOST_W118)<=2).astype(float).where(v(P.PODMHOST_W118).notna())
P['ba']=(P.F_EDUCCAT==1).astype(float).where(P.F_EDUCCAT<99); P['edu6']=v(P.F_EDUCCAT2); P['rep']=(P.F_PARTYSUM_FINAL==1).astype(float); P['dem']=(P.F_PARTYSUM_FINAL==2).astype(float); P['age']=v(P.F_AGECAT); P['female']=(P.F_GENDER==2).astype(float)
P['opinions']=6-v(P.PODCOMM_W118) if 'PODCOMM_W118' in P else np.nan; P['trust']=v(P.PODTRUST_W118); P['convo']=v(P.PODCONVO_W118); P['recommend']=(v(P.PODREC2_W118)==1).astype(float)
P['w']=P.WEIGHT_W118
g=P.groupby('podmain').agg(n=('QKEY','size'),address=('address','first'),one_sided=('one_sided','first'),conn_mean=('conn','mean'),very_conn=('conn_hi','mean'),ba=('ba','mean'),rep=('rep','mean')).sort_values('address')
print(g.round(2).to_string()); print(f"  {len(P)} respondents, {P.show_id.nunique()} scored shows (Daily Wire/Morning Wire mapped to Shapiro)")
def fit(y,x,extra=(),B=2000):
    dd=P.dropna(subset=[y,x,'edu6','age']+list(extra)); X=pd.DataFrame({'const':1.0,'rep':dd.rep,'dem':dd.dem,'age':dd.age,'female':dd.female}); 
    if y not in ('ba','edu6'): X['edu']=dd.edu6
    
    for c in extra: X[c]=dd[c]
    X['x']=dd[x]; r=cr_all(dd[y].values,X.values,dd.show_id.values,list(X.columns).index('x'),B=B); r['N']=len(dd); r['G']=dd.show_id.nunique(); return r
print("\n  outcome ~ address of main podcast (per SD), controls party, education, age, gender; cluster = show (CR2 p / wild p)")
for y,lab in [('conn','felt connection to host (1-5)'),('conn_hi','very/extremely connected'),('opinions','host shares political opinions (freq)'),('trust','trusts podcast news (lower=more)'),('convo','discusses podcast with others'),('recommend','has recommended a podcast'),('ba','listener has BA+'),('edu6','listener education (1-6)')]:
    r1=fit(y,'address'); r2=fit(y,'address',('one_sided',)); r3=fit(y,'one_sided',('address',))
    print(f"    {lab:<40} address {r1['b']:+.3f} ({r1['p2']:.2f}/{r1['pw']:.2f}) | net one-sided {r2['b']:+.3f} ({r2['p2']:.2f}/{r2['pw']:.2f}) | one-sided net address {r3['b']:+.3f} ({r3['p2']:.2f}/{r3['pw']:.2f})  N={r1['N']} G={r1['G']}")
print(f"\n  note: PODMHOST scale 1 extremely .. 5 not at all (reversed here); overall share very/extremely connected among all who named a podcast: {(v(d.PODMHOST_W118)<=2).mean():.0%}")

print("\n  W118 composition, show level (weighted by n): Spearman r(address, share BA+ among a show's main listeners) over the 10 shows =", f"{stats.spearmanr(g.address, g.ba)[0]:+.2f}" , "| r(one-sided, BA+) =", f"{stats.spearmanr(g.one_sided, g.ba)[0]:+.2f}")
gg=g[g.n>=7]; print(f"  shows with n>=7 ({len(gg)}): Spearman r(address, BA+) = {stats.spearmanr(gg.address,gg.ba)[0]:+.2f}; r(address, very connected) = {stats.spearmanr(gg.address,gg.very_conn)[0]:+.2f}; r(one-sided, very connected) = {stats.spearmanr(gg.one_sided,gg.very_conn)[0]:+.2f}")
