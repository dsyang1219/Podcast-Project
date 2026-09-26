"""Build the paper's seven figures from existing tables. Output: figures/fig*.png (300 dpi) and .pdf.
Palette: left #2a78d6 / right #e34948 (diverging pair); address #2a78d6 vs one-sidedness #eb6834; education ramp (blue steps)."""
import sys, re, warnings, glob; warnings.filterwarnings('ignore')
sys.argv=['x']; exec(open('step7_audience/9_education_probes/edu_consequences.py').read().split('a_edu=fit(')[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.patches import Patch; from matplotlib.lines import Line2D
from scipy import stats; import statsmodels.formula.api as smf, statsmodels.api as sm, pyreadstat, textstat
OUT='figures/'; L,Rc,GR,AD,OS='#2a78d6','#e34948','#9a9992','#2a78d6','#eb6834'; BAND='#e6e5e1'; INK,INK2='#0b0b0b','#52514e'
RAMP=['#86b6ef','#5598e7','#2a78d6','#184f95']
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','Nimbus Roman','DejaVu Serif'],'mathtext.fontset':'stix','font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#c8c7c2','axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'axes.titlecolor':INK,'axes.titlesize':11,'axes.titleweight':'normal','axes.titlelocation':'left','grid.color':'#ebeae6','grid.linewidth':0.6,'axes.grid':True,'axes.axisbelow':True,'legend.frameon':False,'figure.dpi':150,'savefig.dpi':300,'pdf.fonttype':42})
def save(fig,name): fig.savefig(OUT+name+'.png',bbox_inches='tight'); fig.savefig(OUT+name+'.pdf',bbox_inches='tight'); plt.close(fig); print('saved',name)
def ci(r): t=stats.t.ppf(.975,max(r.get('df',30),1.5)); return r['b']-t*r['se2'],r['b']+t*r['se2']
lean=X.set_index('show_id').lean; col=lambda l: L if l=='L' else (Rc if l=='R' else GR)

# ---------------- Fig 1: address by show
S=X.sort_values('mfte_z').reset_index(drop=True); S['c']=S.lean.map(col)
fig,ax=plt.subplots(figsize=(7.2,3.6)); ax.scatter(range(len(S)),S.mfte_z,s=14,c=S.c,edgecolor='white',linewidth=0.4,zorder=3); ax.axhline(0,color='#c8c7c2',lw=0.8,zorder=1)
ax.set_ylabel('Listener-directed address (SD from corpus mean)'); ax.set_xlabel('194 shows, ranked by address'); ax.set_xticks([])
for i,dy in [(0,0),(1,-5),(2,6),(3,0),(len(S)-1,0),(len(S)-2,0),(len(S)-4,-2),(len(S)-7,-2)]:
    r=S.iloc[i]; ax.annotate(r.show[:34],(i,r.mfte_z),xytext=(6 if i<10 else -7,dy),textcoords='offset points',ha='left' if i<10 else 'right',va='center',fontsize=8,color=INK2)
mL,mR=S[S.lean=='L'].mfte_z.mean(),S[S.lean=='R'].mfte_z.mean()
ax.axhline(mL,color=L,lw=1,ls=(0,(4,3)),zorder=2); ax.axhline(mR,color=Rc,lw=1,ls=(0,(4,3)),zorder=2)
ax.text(len(S)*0.30,mR+0.1,f'Right-leaning mean {mR:+.2f}',color=Rc,fontsize=9,ha='left'); ax.text(len(S)*0.62,mL-0.32,f'Left-leaning mean {mL:+.2f}',color=L,fontsize=9,ha='left')
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=L,label='Left-leaning'),Line2D([],[],marker='o',ls='',color=Rc,label='Right-leaning')],loc='upper left',bbox_to_anchor=(0,0.92))
ax.set_title('Listener-directed address across 194 shows'); save(fig,'fig1_address_by_show')

# ---------------- Fig 2: stability (event study + quarterly trajectories)
ev=[('2020 election',0.006),('6 January',0.022),('Dobbs',0.009),('2022 midterm',0.048),('Assassination attempt',0.012),('Biden withdrawal',0.024),('2024 election',-0.058),('2025 inauguration',0.024)]; lo95,hi95=-0.094,0.089
E=pd.read_csv('step7_audience/inputs/ep_dates.csv',dtype={'show_id':str,'episode_id':str}); E=E[E.date.astype(str).str.len()==10]; E['dt']=pd.to_datetime(E.date,errors='coerce')
M=pd.read_csv('step7_audience/inputs/mfte_episode_rates.csv',dtype={'show_id':str,'episode_id':str}).merge(E[['show_id','episode_id','dt']],on=['show_id','episode_id']); M['a']=(z(M.vimp)+z(M.pp2))/2; M=M[M.dt>='2018-01-01']; M['q']=M.dt.dt.to_period('Q').dt.to_timestamp()
cnt=M.groupby('show_id').agg(n=('a','size'),nq=('q','nunique')).query('n>=150 and nq>=24'); cand=X[X.show_id.isin(cnt.index)].sort_values('mfte_z'); qs=[0.02,0.35,0.65,0.98]; pick=cand.iloc[[int(q*(len(cand)-1)) for q in qs]]
fig,axs=plt.subplots(1,2,figsize=(8.4,3.3),gridspec_kw={'width_ratios':[1,1.35],'wspace':0.38})
ax=axs[0]; ax.axhspan(lo95,hi95,color=BAND,zorder=0); ax.axhline(0,color='#c8c7c2',lw=0.8); ax.scatter(range(len(ev)),[v for _,v in ev],s=28,color=INK,zorder=3)
ax.set_xticks(range(len(ev))); ax.set_xticklabels([e for e,_ in ev],rotation=40,ha='right',fontsize=8); ax.set_ylabel('Shift in address after vs before (SD)'); ax.set_ylim(-0.15,0.15); ax.text(0.03,0.96,'Grey band: 95% range\nof 300 placebo dates',transform=ax.transAxes,fontsize=8,color=INK2,va='top'); ax.set_title('Eight political shocks')
ax=axs[1]
for _,r in pick.iterrows():
    g=M[M.show_id==r.show_id].groupby('q').a.mean().rolling(3,center=True,min_periods=1).mean(); ax.plot(g.index,g.values,lw=1.8,color=col(r.lean),alpha=0.95); ax.annotate(r.show[:24],(g.index[-1],g.values[-1]),xytext=(4,0),textcoords='offset points',fontsize=8,va='center',color=INK2)
ax.set_ylabel('Quarterly address (SD, 3-quarter average)'); ax.set_title('Four shows, quarter by quarter'); ax.set_xlim(pd.Timestamp('2018-01-01'),pd.Timestamp('2028-06-01')); ax.set_xticks([pd.Timestamp(f'{y}-01-01') for y in (2018,2020,2022,2024,2026)]); ax.set_xticklabels(['2018','2020','2022','2024','2026'])
axs[1].legend(handles=[Line2D([],[],color=L,lw=1.8,label='Left-leaning'),Line2D([],[],color=Rc,lw=1.8,label='Right-leaning')],loc='upper left',ncol=2); save(fig,'fig2_stability')

# ---------------- Fig 3: style vs stance scatter
fig,ax=plt.subplots(figsize=(4.6,4.2)); d=X.dropna(subset=['inten']); iz=(d.inten-mu)/sd
ax.scatter(iz,d.mfte_z,s=18,c=d.lean.map(col),edgecolor='white',linewidth=0.4,zorder=3); r=np.corrcoef(iz,d.mfte_z)[0,1]
ax.set_xlabel('One-sidedness (SD)'); ax.set_ylabel('Listener-directed address (SD)'); ax.axhline(0,color='#c8c7c2',lw=0.8); ax.axvline(0,color='#c8c7c2',lw=0.8)
ax.text(0.98,0.97,f'Across shows r = {r:.2f}\nWithin shows, passage level r = −0.01',transform=ax.transAxes,va='top',ha='right',fontsize=9,color=INK2)
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=L,label='Left-leaning'),Line2D([],[],marker='o',ls='',color=Rc,label='Right-leaning')],loc='lower right'); ax.set_title('Address and one-sidedness across shows'); save(fig,'fig3_style_vs_stance')

# ---------------- Fig 4: who listens (education composition)
def levels(s): s=s.dropna(); return np.array([(s<=2).mean(),((s>=3)&(s<=5)).mean(),(s==6).mean(),(s>=7).mean()])
R['terc']=pd.cut(R.x,[-9,R.x.quantile(1/3),R.x.quantile(2/3),9],labels=['Low-address\nthird','Middle\nthird','High-address\nthird']); pub=pd.concat([K1.edu,K2.edu])
groups=[('Low-address\nthird',R[R.terc=='Low-address\nthird'].edu),('Middle\nthird',R[R.terc=='Middle\nthird'].edu),('High-address\nthird',R[R.terc=='High-address\nthird'].edu),('General\npublic',pub)]
fig,axs=plt.subplots(1,2,figsize=(8.2,3.5),gridspec_kw={'width_ratios':[1.25,1],'wspace':0.12}); labs=['High school or less','Some college / associate','Bachelor\'s','Graduate degree']
def stacked(ax,groups,title):
    x=np.arange(len(groups)); bottom=np.zeros(len(groups))
    for k in range(4):
        vals=np.array([levels(g)[k] for _,g in groups]); ax.bar(x,vals,bottom=bottom,color=RAMP[k],width=0.62,edgecolor='white',linewidth=1.5,label=labs[k])
        for i,v in enumerate(vals):
            if v>=0.08: ax.text(x[i],bottom[i]+v/2,f'{v:.0%}',ha='center',va='center',fontsize=8.5,color='white' if k>=2 else INK)
        bottom+=vals
    ax.set_xticks(x); ax.set_xticklabels([n for n,_ in groups],fontsize=9); ax.set_ylim(0,1); ax.set_yticks([0,.25,.5,.75,1]); ax.set_yticklabels(['0','25%','50%','75%','100%']); ax.grid(axis='x',visible=False); ax.set_title(title)
stacked(axs[0],groups,'By show address, and the general public')
left=R[R.share_right<0.5]; right=R[R.share_right>=0.5]
g2=[('Left shows,\nlower half',left[left.x<left.x.median()].edu),('Left shows,\nupper half',left[left.x>=left.x.median()].edu),('Right shows,\nlower half',right[right.x<right.x.median()].edu),('Right shows,\nupper half',right[right.x>=right.x.median()].edu)]
stacked(axs[1],g2,'Within each side'); axs[1].set_yticklabels([]); axs[1].set_xticklabels([n for n,_ in g2],fontsize=8)
axs[0].set_ylabel('Share of listeners'); fig.legend(handles=[Patch(color=RAMP[k],label=labs[k]) for k in range(4)],loc='lower center',bbox_to_anchor=(0.5,-0.07),ncol=4); save(fig,'fig4_who_listens')

# ---------------- Fig 5: replication across sources (2 x 2)
fig,axs=plt.subplots(2,2,figsize=(7.4,6.6),gridspec_kw={'wspace':0.32,'hspace':0.42}); axs=axs.ravel()
# (a) Kettering show level
y1=pd.read_csv('data/external/kettering/y1_match_maximal.csv',dtype={'show_id':str}); y2=pd.read_csv('data/external/kettering/y2/y2_match_maximal.csv',dtype={'show_id':str}); y2['resp']+=10**6
Pk=pd.concat([y1.assign(year=1),y2.assign(year=2)]).merge(R[['resp','edu']],on='resp').merge(X[['show_id','mfte_z','lean']],on='show_id'); gk=Pk.groupby('show_id').agg(n=('resp','nunique'),edu=('edu','mean'),x=('mfte_z','first'),lean=('lean','first')).query('n>=5')
ax=axs[0]; ax.scatter(gk.x,gk.edu,s=np.sqrt(gk.n)*7,c=gk.lean.map(col),alpha=0.85,edgecolor='white',linewidth=0.5); bb=np.polyfit(gk.x,gk.edu,1,w=np.sqrt(gk.n)); xs=np.linspace(gk.x.min(),gk.x.max(),10); ax.plot(xs,np.polyval(bb,xs),color=INK,lw=1.2)
ax.set_title(f'Kettering–Gallup ({len(gk)} shows)'); ax.set_ylabel('Mean listener education (1–8)'); ax.set_xlabel('Address (SD)'); ax.text(0.97,0.95,f'r = {gk.x.corr(gk.edu):.2f}',transform=ax.transAxes,fontsize=9,color=INK2,ha='right',va='top')
ax.legend(handles=[Line2D([],[],marker='o',ls='',color=L,label='Left-leaning'),Line2D([],[],marker='o',ls='',color=Rc,label='Right-leaning')],loc='lower left',fontsize=8,handletextpad=0.3)
# (b) Pew W165
T=pd.read_csv('data/output/pew_w165_source_audiences.csv'); MAP={'The Joe Rogan Experience':'HOLDOUT_rogan','Tucker Carlson Network':'HOLDOUT_tucker','The Daily Wire':'HOLDOUT_shapiro','Breitbart':'1592116227','NPR':'1057255460','Fox News':'1303660358','NBC News':'75462585','The New York Times':'HOLDOUT_thedaily','The Atlantic':'1258635512'}
Sm=pd.read_csv('step7_audience/inputs/mfte_show_scores.csv',dtype={'show_id':str}).set_index('show_id').mfte_z; T['show_id']=T.source.map(MAP); T=T.dropna(subset=['show_id']); T['x']=T.show_id.map(Sm)
NAME={'The Joe Rogan Experience':'Joe Rogan','Tucker Carlson Network':'Tucker Carlson','The New York Times':'New York Times','The Daily Wire':'Daily Wire'}
POS={'The Atlantic':(6,4,'left'),'NPR':(6,2,'left'),'The New York Times':(-6,-4,'right'),'The Daily Wire':(6,3,'left'),'Breitbart':(-6,0,'right'),'NBC News':(7,6,'left'),'Fox News':(0,-10,'center'),'Tucker Carlson Network':(0,9,'center'),'The Joe Rogan Experience':(0,-10,'center')}
ax=axs[1]; ax.scatter(T.x,T.ba_w,s=34,color=INK,zorder=3)
for _,r in T.iterrows(): dx,dy,ha=POS[r.source]; ax.annotate(NAME.get(r.source,r.source),(r.x,r.ba_w),xytext=(dx,dy),textcoords='offset points',fontsize=8,color=INK2,ha=ha,va='center')
ax.set_title(f'Pew named sources, 2025 ({len(T)})'); ax.set_ylabel('Audience with a bachelor\'s degree'); ax.set_xlabel('Address (SD)'); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0,decimals=0)); ax.text(0.97,0.95,f'Spearman r = {stats.spearmanr(T.x,T.ba_w)[0]:.2f}',transform=ax.transAxes,fontsize=9,color=INK2,ha='right',va='top'); ax.set_xlim(T.x.min()-0.45,T.x.max()+0.45); ax.set_ylim(0.2,0.7)
# (c) Pew W118
f=glob.glob('data/external/pew_w118/W118_Dec22/*.sav')[0]; d8,m8=pyreadstat.read_sav(f); vl=m8.variable_value_labels['PODMAIN_CODES_W118']; d8['podmain']=d8.PODMAIN_CODES_W118.map(vl)
PM={'The Joe Rogan Experience':'HOLDOUT_rogan','The Daily (NYT)':'HOLDOUT_thedaily','The Dan Bongino Show':'HOLDOUT_bongino','The Ben Shapiro Show':'HOLDOUT_shapiro','Pod Save America':'1192761536','Timcast IRL':'1362265400','Breaking Points with Krystal and Saagar':'1570045623','The Glenn Beck Program':'HOLDOUT_beck','The Mark Levin Show':'1818658022','Mea Culpa with Michael Cohen':'1714009198','Daily Wire (general reference)':'HOLDOUT_shapiro','Morning Wire':'HOLDOUT_shapiro'}
d8['show_id']=d8.podmain.map(PM); P8=d8.dropna(subset=['show_id']).copy(); P8['ba']=(P8.F_EDUCCAT==1).astype(float).where(P8.F_EDUCCAT<99); g8=P8.groupby('show_id').agg(n=('QKEY','size'),ba=('ba','mean')).reset_index(); g8['x']=g8.show_id.map(Sm)
N8={'HOLDOUT_rogan':'Joe Rogan','HOLDOUT_thedaily':'The Daily','HOLDOUT_bongino':'Dan Bongino','HOLDOUT_shapiro':'Ben Shapiro','1192761536':'Pod Save America','1362265400':'Timcast IRL','1570045623':'Breaking Points','HOLDOUT_beck':'Glenn Beck','1818658022':'Mark Levin','1714009198':'Mea Culpa'}; g8['name']=g8.show_id.map(N8)
P8pos={'Joe Rogan':(-7,-2,'right'),'The Daily':(7,2,'left'),'Dan Bongino':(0,-10,'center'),'Ben Shapiro':(7,-4,'left'),'Pod Save America':(7,0,'left'),'Timcast IRL':(7,3,'left'),'Breaking Points':(7,2,'left'),'Glenn Beck':(7,3,'left'),'Mark Levin':(7,2,'left'),'Mea Culpa':(7,0,'left')}
ax=axs[2]; ax.scatter(g8.x,g8.ba,s=np.sqrt(g8.n)*7,color=INK,alpha=0.85,zorder=3)
for _,r in g8.iterrows(): dx,dy,ha=P8pos[r['name']]; ax.annotate(r['name'],(r.x,r.ba),xytext=(dx,dy),textcoords='offset points',fontsize=8,color=INK2,ha=ha,va='center')
ax.set_title(f'Pew podcast wave, 2022 ({len(g8)} shows)'); ax.set_ylabel('Listeners with a bachelor\'s degree'); ax.set_xlabel('Address (SD)'); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0,decimals=0)); ax.text(0.97,0.95,f'Spearman r = {stats.spearmanr(g8.x,g8.ba)[0]:.2f}',transform=ax.transAxes,fontsize=9,color=INK2,ha='right',va='top'); ax.set_xlim(g8.x.min()-0.6,g8.x.max()+0.5); ax.set_ylim(0.15,0.99)
# (d) reviews
O=pd.read_csv('data/output/show_level_outcomes.csv',dtype={'show_id':str}); Rv=pd.read_csv('data/output/apple_reviews.csv',dtype={'show_id':str}).drop_duplicates(['show_id','review_id']); Rv['text']=Rv.text.astype(str); Rv=Rv[Rv.text.str.split().str.len()>=15]; Rv['fk']=Rv.text.map(textstat.flesch_kincaid_grade).clip(0,20)
gr=Rv.groupby('show_id').agg(n=('fk','size'),fk=('fk','mean')).reset_index().merge(O[['show_id','addr_noad','lean']],on='show_id').query('n>=20'); gr['x']=z(gr.addr_noad)
ax=axs[3]; ax.scatter(gr.x,gr.fk,s=np.sqrt(gr.n)*2.4,c=gr.lean.map(col),alpha=0.8,edgecolor='white',linewidth=0.4); bb=np.polyfit(gr.x,gr.fk,1,w=np.sqrt(gr.n)); xs=np.linspace(gr.x.min(),gr.x.max(),10); ax.plot(xs,np.polyval(bb,xs),color=INK,lw=1.2)
ax.set_title(f'Apple reviews ({len(gr)} shows)'); ax.set_ylabel('Reading grade of listener reviews'); ax.set_xlabel('Address (SD)'); ax.text(0.97,0.95,f'r = {gr.x.corr(gr.fk):.2f}',transform=ax.transAxes,fontsize=9,color=INK2,ha='right',va='top')
for ax in axs: ax.title.set_fontsize(10)
save(fig,'fig5_replication')

# ---------------- Fig 6: what follows style, what follows stance
rows=[]
for y,lab in [('edu_z','Listener education'),('elec_z','Election distrust'),('cyn4_z','Institutional cynicism')]:
    ra=fit(R,y,'x',('inten_z',),B=2); ro=fit(R,y,'inten_z',('x',),B=2); rows.append((lab,'Kettering–Gallup, 1,200 listeners',ra['b'],*ci(ra),ro['b'],*ci(ro)))
O['R']=(O.lean=='R').astype(float); dd=O[O.n_reviews>=20].copy(); dd['y']=z(dd.share5); dd['a']=z(dd.addr_noad); dd['o']=z(dd.inten)
m=smf.wls('y ~ a + o + R + solo_ + log_epw',dd,weights=np.sqrt(dd.n_reviews)).fit(cov_type='HC1'); c=m.conf_int(); rows.append(('Share of 5-star reviews','Apple reviews, 119 shows',m.params.a,c.loc['a',0],c.loc['a',1],m.params.o,c.loc['o',0],c.loc['o',1]))
Sx=pd.read_csv('step7_audience/inputs/mfte_show_scores.csv',dtype={'show_id':str})[['show_id','mfte_z']]; D=pd.read_csv('step7_audience/inputs/directive_final.csv',dtype={'show_id':str})[['show_id','inten']]; ID=pd.read_csv('step7_audience/inputs/inten_distilled.csv',dtype={'show_id':str})[['show_id','inten_use']]
A=Sx.merge(D,on='show_id',how='left').merge(ID,on='show_id',how='left'); A['inten_z']=(A.inten.fillna(A.inten_use)-mu)/sd; A=A.set_index('show_id')
P8=P8.copy(); P8['address']=P8.show_id.map(A.mfte_z); P8['one']=P8.show_id.map(A.inten_z); v8=lambda s: s.where(s<99); P8['conn']=z(6-v8(P8.PODMHOST_W118)); P8['rep']=(P8.F_PARTYSUM_FINAL==1).astype(float); P8['dem']=(P8.F_PARTYSUM_FINAL==2).astype(float); P8['edu6']=v8(P8.F_EDUCCAT2); P8['age']=v8(P8.F_AGECAT); P8['female']=(P8.F_GENDER==2).astype(float)
def fit8(y,x,ex):
    d=P8.dropna(subset=[y,x,'edu6','age']+list(ex)); Xm=pd.DataFrame({'const':1.0,'rep':d.rep,'dem':d.dem,'edu':d.edu6,'age':d.age,'female':d.female})
    for c_ in ex: Xm[c_]=d[c_]
    Xm['x']=d[x]; return cr_all(d[y].values,Xm.values,d.show_id.values,list(Xm.columns).index('x'),B=2)
ra=fit8('conn','address',('one',)); ro=fit8('conn','one',('address',)); rows.append(('Felt connection to host','Pew podcast wave, 205 listeners',ra['b'],*ci(ra),ro['b'],*ci(ro)))
fig,ax=plt.subplots(figsize=(7.2,3.4)); yy=np.arange(len(rows))[::-1]
for i,(lab,src,ba,la,ha,bo,lo_,ho) in zip(yy,rows):
    ax.plot([la,ha],[i+0.16,i+0.16],color=AD,lw=1.6); ax.scatter([ba],[i+0.16],s=34,color=AD,zorder=3,edgecolor='white',linewidth=0.6)
    ax.plot([lo_,ho],[i-0.16,i-0.16],color=OS,lw=1.6); ax.scatter([bo],[i-0.16],s=34,color=OS,zorder=3,edgecolor='white',linewidth=0.6)
    ax.text(-1.02,i,f'{lab}\n',va='center',ha='left',fontsize=8.5,color=INK,transform=ax.get_yaxis_transform()) if False else None
ax.set_yticks(yy); ax.set_yticklabels([f'{lab}\n' for lab,*_ in rows],fontsize=10,color=INK)
for i,(lab,src,*_) in zip(yy,rows): ax.text(-0.02,i-0.24,src,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8,color=INK2)
ax.axvline(0,color='#c8c7c2',lw=0.9); ax.set_xlabel('Standardized coefficient (SD of outcome per SD of show measure), each net of the other'); ax.grid(axis='y',visible=False)
ax.legend(handles=[Line2D([],[],marker='o',color=AD,lw=1.6,label='Listener-directed address'),Line2D([],[],marker='o',color=OS,lw=1.6,label='One-sidedness')],loc='lower center',bbox_to_anchor=(0.5,-0.3),ncol=2); ax.set_title('Address and one-sidedness as predictors of five audience outcomes'); save(fig,'fig6_style_vs_stance_outcomes')

# ---------------- Fig 7: which commands carry it
I=pd.read_csv('step7_audience/inputs/mfte_imperative_show_scores.csv',dtype={'show_id':str}); zc=['VIMP_ACT_z','VIMP_DOAUX_z','VIMP_CAUSE_z','VIMP_ASPECT_z','VIMP_NONE_z','VIMP_COMM_z','VIMP_MENTAL_z','youknow_z']
agg=pd.concat([y1,y2]).merge(I[['show_id']+zc],on='show_id').groupby('resp')[zc].mean().reset_index(); R2=R.merge(agg,on='resp',how='left')
names={'VIMP_ACT_z':'Activity commands (make, take, check)','VIMP_DOAUX_z':'"Do" / "don\'t" commands','VIMP_CAUSE_z':'"Let\'s" / "let me"','VIMP_ASPECT_z':'Keep / start / stop','VIMP_NONE_z':'Other commands (look, go, find)','VIMP_COMM_z':'Communication commands (tell, explain)','VIMP_MENTAL_z':'Thinking commands (think, remember)','youknow_z':'"You know"'}
res=[(names[c],fit(R2,'edu_z',c,('inten_z',),B=2)) for c in zc]; res=[(n,r['b'],*ci(r)) for n,r in res]
fig,ax=plt.subplots(figsize=(6.4,3.4)); yy=np.arange(len(res))[::-1]
for i,(n,b,lo_,hi_) in zip(yy,res):
    cc=AD if b<0 else OS; ax.plot([lo_,hi_],[i,i],color=cc,lw=1.6); ax.scatter([b],[i],s=34,color=cc,zorder=3,edgecolor='white',linewidth=0.6)
ax.set_yticks(yy); ax.set_yticklabels([n for n,*_ in res],fontsize=10,color=INK); ax.axvline(0,color='#c8c7c2',lw=0.9); ax.grid(axis='y',visible=False)
ax.set_xlabel('Listener education (SD) per SD of the feature, net of one-sidedness'); ax.set_title('Education coefficient by type of command'); save(fig,'fig7_command_types')
print('done')
