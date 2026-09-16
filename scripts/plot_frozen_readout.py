"""Plot measured staged-training results and the fixed validation trajectory."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'results/plasticity/staged'
GROUPS=['joint_65k','joint_262k','readout_262k','staged_262k']
LABELS=['Joint 65k\n(start)','Joint 262k','Readout-only\n262k','Joint then frozen\n262k']
COLORS=['#ACB7C2','#BD6864','#718AAB','#277A72']


def main():
    data=json.loads((OUT/'comparison.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for i,(group,label,color) in enumerate(zip(GROUPS,LABELS,COLORS)):
        s=data['groups'][group]['success_rate']
        axes[0].bar(i,s['mean']*100,yerr=s['sample_sd']*100,color=color,width=.6,capsize=3)
        axes[0].scatter(i+np.array([-.1,0,.1]),np.asarray(s['values'])*100,s=18,c='#263849',zorder=3)
        axes[0].text(i,(s['mean']+s['sample_sd'])*100+3,f"{s['mean']*100:.1f}%",ha='center',fontsize=10)
    axes[0].set(xticks=range(4),xticklabels=LABELS,ylabel='Goal reached (%)',ylim=(0,100),title='New test: 200 identical maps per seed')
    steps=np.asarray(data['steps'])
    for group,label,color in zip(GROUPS[1:],['Joint PPO','Readout-only PPO','Joint then frozen'],COLORS[1:]):
        s=[data['validation'][group][str(step)] for step in steps]
        mean=np.array([x['mean'] for x in s])*100;sd=np.array([x['sample_sd'] for x in s])*100
        axes[1].plot(steps,mean,'o-',label=label,color=color)
        axes[1].fill_between(steps,mean-sd,mean+sd,color=color,alpha=.13)
    axes[1].set(xlabel='Cumulative environment steps',ylabel='Goal reached (%)',ylim=(0,100),title='Fixed validation: three prespecified milestones')
    axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Freeze learned brain connections at 65,536 steps · 3 training seeds',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.6,4.4),layout='constrained')
    base=np.zeros(4)
    for metric,color,label in [('successes','#277A72','Success'),('collisions','#BD6864','Collision'),('timeouts','#BDA06A','Timeout')]:
        heights=np.array([data['groups'][g][metric]['mean']/2 for g in GROUPS])
        ax.bar(range(4),heights,bottom=base,color=color,label=label,width=.65);base+=heights
    ax.set(xticks=range(4),xticklabels=LABELS,ylabel='Test episodes (%)',ylim=(0,100),title='Outcomes on the same 200 new maps per seed')
    ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(1,1))
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'failure-types.png',dpi=180);plt.close(fig)
    print(OUT/'comparison.png')


if __name__=='__main__':main()
