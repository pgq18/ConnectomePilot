"""Render success across fixed budgets and measured failure types."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'results/plasticity/staged-long'


def main():
    data=json.loads((OUT/'comparison.json').read_text())
    steps=np.array(data['steps']);x=steps/1000
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.5),layout='constrained')
    for ax,key,title in zip(axes,['fresh','validation'],['New held-out test: 200 maps / seed','Fixed validation: 100 maps / seed']):
        points=[data[key][str(step)]['success_rate'] for step in steps]
        mean=np.array([p['mean'] for p in points])*100;sd=np.array([p['sample_sd'] for p in points])*100
        for i,(seed,color) in enumerate(zip(data['seeds'],['#718AAB','#B47A42','#6F9F8D'])):
            ax.plot(x,[p['values'][i]*100 for p in points],'o--',lw=1,alpha=.75,color=color,label=f'Seed {seed}')
        ax.plot(x,mean,'o-',color='#174F55',lw=2.2,label='Mean')
        ax.fill_between(x,mean-sd,mean+sd,color='#277A72',alpha=.12)
        for a,b in zip(x,mean):ax.annotate(f'{b:.1f}%',(a,b),xytext=(0,-18),textcoords='offset points',ha='center',fontsize=9)
        ax.set(xticks=x,xticklabels=['262k','524k','786k','1,049k'],xlabel='Cumulative environment steps',ylabel='Goal reached (%)',ylim=(70,100),title=title)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=9,loc='lower right')
    fig.suptitle('Continue readout learning · brain connections frozen since 65,536 steps',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.2),layout='constrained')
    base=np.zeros(4)
    for key,color,label in [('successes','#277A72','Success'),('collisions','#BD6864','Collision'),('timeouts','#BDA06A','Timeout')]:
        heights=np.array([data['fresh'][str(step)][key]['mean']/2 for step in steps])
        ax.bar(range(4),heights,bottom=base,color=color,label=label,width=.65);base+=heights
    ax.set(xticks=range(4),xticklabels=['262,144','524,288','786,432','1,048,576'],xlabel='Training steps',ylabel='Test episodes (%)',ylim=(0,100),title='Same 200 new maps for every checkpoint and training seed')
    ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(1,1));ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'failure-types.png',dpi=180);plt.close(fig)
    print(OUT/'comparison.png')


if __name__=='__main__':main()
