import sys; sys.argv=['x']; exec(open('step7_audience/9_education_probes/edu_consequences.py').read().split('a_edu=fit(')[0])
def addcols(K):
    K=K.copy(); K['inc']=num(K,'Q49').where(lambda s:s<=10); K['inc_z']=z(K.inc)
    e=num(K,'Q48'); K['emp_ft']=(e==1).astype(float).where(e.notna()); K['emp_pt']=(e==2).astype(float).where(e.notna()); K['not_emp_nolook']=(e==3).astype(float).where(e.notna()); K['unemployed']=(e==4).astype(float).where(e.notna())
    for q in ['Q12F','Q12D']: v=num(K,q); K[q+'_yes']=(v==1).astype(float).where(v.notna())
    v=num(K,'Q9D'); K['schools_sat_z']=z(v.where(v<=4)); K['Q18D_z']=z(num(K,'Q18D'))
    for q in ['Q25','Q26','Q8','Q5','Q6']:
        if q in K: K[q+'_z']=z(num(K,q))
    K['noncollege_highinc']=((K.edu<=5)&(K.inc>=7)).astype(float).where(K.edu.notna()&K.inc.notna())   # no bachelor's, household income >= $90k
    K['college_lowinc']=((K.edu>=6)&(K.inc<=5)).astype(float).where(K.edu.notna()&K.inc.notna())
    K['edu_resid']=K.edu_z-np.polyval(np.polyfit(K.inc_z.fillna(0),K.edu_z.fillna(0),1),K.inc_z.fillna(0))
    return K
K1=addcols(K1); K2=addcols(K2)
new=['inc','inc_z','emp_ft','emp_pt','not_emp_nolook','unemployed','Q12F_yes','Q12D_yes','schools_sat_z','Q18D_z','Q25_z','Q26_z','Q8_z','Q5_z','Q6_z','noncollege_highinc','college_lowinc','edu_resid']
R=pd.concat([R[R.year==1].merge(K1[['resp']+[c for c in new if c in K1]],on='resp'),R[R.year==2].assign(resp0=lambda d:d.resp-10**6).merge(K2[['resp']+[c for c in new if c in K2]].rename(columns={'resp':'resp0'}),on='resp0').drop(columns='resp0')],ignore_index=True)
def line(lab,d,y,x='x',ex=('inten_z',),B=800):
    out=[]
    for dd in (d,d[d.year==1],d[d.year==2]):
        if y not in dd or dd[y].notna().sum()<80: out.append(f"{'--':>22}"); continue
        r=fit(dd,y,x,ex,B=B); out.append(f"{r['b']:+.3f} ({r['p2']:.2f}/{r['pw']:.2f})")
    print(f"  {lab:<58} "+' | '.join(f"{o:>22}" for o in out))
print("=== 1. EDUCATION vs INCOME (pooled | Y1 | Y2; address net of one-sidedness) ===")
line("education ~ address",R,'edu_z'); line("education ~ address + household INCOME",R,'edu_z',ex=('inten_z','inc_z'))
line("income ~ address",R,'inc_z'); line("income ~ address + EDUCATION",R,'inc_z',ex=('inten_z','edu_z'))
line("education residual on income (schooling beyond income)",R,'edu_resid')
line("P(no bachelor's AND income >= $90k)  [pts/SD]",R,'noncollege_highinc'); line("P(bachelor's AND income < $60k)  [pts/SD]",R,'college_lowinc')
R['terc']=pd.cut(R.x,[-9,R.x.quantile(1/3),R.x.quantile(2/3),9],labels=['low','mid','high'])
g=R.groupby('terc').agg(n=('resp','size'),edu=('edu','mean'),inc=('inc','mean'),noncol_hi=('noncollege_highinc','mean'),col_lo=('college_lowinc','mean'),ft=('emp_ft','mean'),nolook=('not_emp_nolook','mean')); print(g.round(2).to_string()); print(f"  public: edu {pd.concat([K1.edu,K2.edu]).mean():.2f}, income {pd.concat([K1.inc,K2.inc]).mean():.2f}, non-college high-income {pd.concat([K1.noncollege_highinc,K2.noncollege_highinc]).mean():.2f}, college low-income {pd.concat([K1.college_lowinc,K2.college_lowinc]).mean():.2f}")
print("\n=== 2. EMPLOYMENT (pts per SD) ===")
for y,lab in [('emp_ft','employed full time'),('emp_pt','part time'),('not_emp_nolook','not employed, not looking (retired/homemaker/student)'),('unemployed','not employed, looking')]: line(lab,R,y)
line("not employed/not looking, net of AGE already in; + education",R,'not_emp_nolook',ex=('inten_z','edu_z'))
print("\n=== 3. CIVIC SOCIALISATION & INFORMATION (net one-sided; then + education) ===")
for y,lab in [('Q25_z','formal civics education received (Y1)'),('Q26_z','parents/relatives encouraged participation (Y1)'),('Q8_z','feels overloaded by information (Y1)'),('Q12F_yes','barrier: not knowing enough about issues'),('Q12D_yes','barrier: unsure how to get involved'),('Q18D_z','trust education officials'),('schools_sat_z','satisfied with schools'),('Q5_z','free time weekday (Y1)'),('Q6_z','free time weekend (Y1)')]:
    line(lab,R,y); line("   ... + education",R,y,ex=('inten_z','edu_z'))
print("\n=== 4. is the education gap concentrated at a particular level? P(each level) ~ address, pooled, pts/SD ===")
for lo,hi,lab in [(1,1,'less than HS'),(2,2,'HS diploma/GED'),(3,3,'vocational/trade'),(4,4,'some college, no degree'),(5,5,'associate'),(6,6,"bachelor's"),(7,8,'graduate/professional')]:
    R['lvl']=((R.edu>=lo)&(R.edu<=hi)).astype(float).where(R.edu.notna()); r=fit(R,'lvl','x',('inten_z',),B=600); print(f"  {lab:<26} share {R.lvl.mean():.0%}  {100*r['b']:+5.1f} pts ({r['p2']:.2f}/{r['pw']:.2f})")
