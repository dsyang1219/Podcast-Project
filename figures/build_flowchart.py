"""Pipeline flow chart for the talk / paper (figures/fig0_pipeline.png/.pdf). Same typeface and palette as the other figures."""
import sys; TALK=len(sys.argv)>1 and sys.argv[1]=='talk'
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],'pdf.fonttype':42})
INK,INK2,BLUE,ORANGE,GRAY,BG='#0b0b0b','#52514e','#2a78d6','#eb6834','#9a9992','#f2f1ee'
fig,ax=plt.subplots(figsize=(11,6.2)); ax.set_xlim(0,11); ax.set_ylim(0,6.2); ax.axis('off')
def box(x,y,w,h,title,body,edge=GRAY,fill='white',tw='bold',fs=10.5,bfs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',linewidth=1.4,edgecolor=edge,facecolor=fill))
    ax.text(x+w/2,y+h-0.28,title,ha='center',va='center',fontsize=fs,fontweight=tw,color=INK)
    ax.text(x+w/2,y+h/2-0.17,body,ha='center',va='center',fontsize=bfs,color=INK2,linespacing=1.35)
def arrow(x1,y1,x2,y2,color=GRAY,style='-|>'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=14,linewidth=1.4,color=color,shrinkA=2,shrinkB=2))
# column 1: corpus
box(0.3,3.9,2.3,1.75,'Shows','Apple Podcasts US Politics chart,\n13 July 2026: 250 shows\n→ 6 inclusion rules → 219\n→ 194 analysed')
box(0.3,1.35,2.3,1.75,'Episodes','Priority-ladder sample\n(census before 2018,\nthen k per show per quarter)\n31,191 episodes, 25,900 hours\ntranscribed with Whisper')
arrow(1.45,3.9,1.45,3.1)
# column 2: measures
box(3.3,3.9,2.6,1.75,'Listener-directed address','Commands and "you" per 10,000\nwords, from a published grammatical\ntagger; z-scored on 194 shows\n(style)',edge=BLUE)
box(3.3,1.35,2.6,1.75,'One-sidedness','Share of political passages that\ntake a side, from a published\ntwo-stage LLM rubric (Much et al.)\n(stance)',edge=ORANGE)
arrow(2.6,4.77,3.3,4.77,BLUE); arrow(2.6,2.22,3.3,2.22,ORANGE)
ax.text(4.6,0.98,'validated against hosts\' donations (DIME),\nBrookings labels and listeners\' party',ha='center',va='center',fontsize=8.5,color=INK2,style='italic')
# column 3: audiences
box(6.6,3.75,2.1,2.05,'Who listens','Kettering–Gallup\nDemocracy for All,\ntwo waves (n 20,338; 23,683):\nrespondents name their\nown sources → 1,247 listeners\nof 64 shows',fs=10.5,bfs=8.6)
box(6.6,1.2,2.1,2.05,'Independent checks','Pew ATP 2025: audiences\nof 30 named sources\nPew ATP 2022: podcast\nlistened to most\nApple reviews: 16,434\nacross 141 shows',fs=10.5,bfs=8.6)
arrow(5.9,4.77,6.6,4.77,GRAY); arrow(5.9,2.22,6.6,2.22,GRAY)
# column 4: findings
box(9.2,3.75,1.6,2.05,'Style →\nwho is reached','Audiences of\naddress-heavy shows\nlook like the public\non education',edge=BLUE,fill='#eef4fc',fs=10,bfs=8.6)
box(9.2,1.2,1.6,2.05,'Style →\ncynicism','Listeners of\naddress-heavy shows\nare more cynical about\ninstitutions (Year 1),\nnet of one-sidedness',edge=ORANGE,fill='#fdf0ea',fs=10,bfs=8.6) if TALK else box(9.2,1.2,1.6,2.05,'Stance →\nwhat is believed','Election distrust\nfollows one-sidedness,\nnot address',edge=ORANGE,fill='#fdf0ea',fs=10,bfs=8.6)
arrow(8.7,4.77,9.2,4.77,BLUE); arrow(8.7,2.22,9.2,2.22,ORANGE)
ax.text(0.3,0.42,'Each audience model holds constant the listener\'s party, attention, age and education, the lean of the shows named, and the survey year;\ninference clusters by show (CR2 + wild cluster bootstrap).',fontsize=8.5,color=INK2,va='center')
name='figures/fig0_pipeline_talk' if TALK else 'figures/fig0_pipeline'; fig.savefig(name+'.png',dpi=300,bbox_inches='tight'); fig.savefig(name+'.pdf',bbox_inches='tight'); print('saved',name)
