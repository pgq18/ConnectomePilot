"""Render success across fixed budgets and measured failure types."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'results/plasticity/staged-4m'


def main():
    data=json.loads((OUT/'comparison.json').read_text())
    steps=np.array(data['steps']);x=steps/1000000
    fig,axes=plt.subplots(1,2,figsize=(13.2,4.8),layout='constrained')
    for ax,key,title in zip(axes,['fresh','validation'],['New held-out test: 500 maps / seed','Fixed validation: 200 maps / seed']):
        points=[data[key][str(step)]['success_rate'] for step in steps]
        mean=np.array([p['mean'] for p in points])*100;sd=np.array([p['sample_sd'] for p in points])*100
        for i,(seed,color) in enumerate(zip(data['seeds'],['#718AAB','#B47A42','#6F9F8D'])):
            ax.plot(x,[p['values'][i]*100 for p in points],'o--',lw=1,alpha=.75,color=color,label=f'Seed {seed}')
        ax.plot(x,mean,'o-',color='#174F55',lw=2.2,label='Mean ± sample SD')
        ax.fill_between(x,mean-sd,mean+sd,color='#277A72',alpha=.12)
        for a,b in zip(x,mean):ax.text(a,97.5,f'Avg {b:.1f}%',ha='center',fontsize=8,color='#174F55')
        ax.set(xticks=x,xticklabels=[f'{v:.2f}' for v in x],xlabel='Cumulative environment steps (millions)',ylabel='Goal reached (%)',ylim=(70,100),title=title)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for i,choice in enumerate(data['selected_checkpoints']):
        step=choice['total_steps']
        for ax,key in zip(axes,['fresh','validation']):
            value=data[key][str(step)]['success_rate']['values'][i]*100
            ax.scatter(step/1000000,value,marker='*',s=180,color=['#718AAB','#B47A42','#6F9F8D'][i],edgecolor='white',linewidth=.7,zorder=10)
    axes[0].legend(frameon=False,fontsize=9,loc='lower right')
    fig.suptitle('Frozen-brain PPO to 4M · stars: checkpoints selected by validation',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained')
    base=np.zeros(len(steps))
    for key,color,label in [('successes','#277A72','Success'),('collisions','#BD6864','Collision'),('timeouts','#BDA06A','Timeout')]:
        heights=np.array([data['fresh'][str(step)][key]['mean']/5 for step in steps])
        ax.bar(range(len(steps)),heights,bottom=base,color=color,label=label,width=.65);base+=heights
    ax.set(xticks=range(len(steps)),xticklabels=[f'{v:.2f}M' for v in x],xlabel='Training steps',ylabel='Test episodes (%)',ylim=(0,100),title='Same 500 new maps for every checkpoint and training seed')
    ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(1,1));ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'failure-types.png',dpi=180);plt.close(fig)
    print(OUT/'comparison.png')


if __name__=='__main__':main()
