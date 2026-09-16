"""Render the fixed-budget continuation, including failure types."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'results/plasticity/long'
MODES=['readout','ppo_edges','three_factor']
LABELS=['Readout PPO','Edge PPO','Local plasticity']
COLORS=['#718AAB','#277A72','#B47A42']


def main():
    result=json.loads((OUT/'comparison.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.6),layout='constrained')
    x=np.arange(3)
    for offset,stage,label,color in [(-.18,'before','65,536 steps','#A0B3C8'),(.18,'after','262,144 steps','#277A72')]:
        values=[result['groups'][m]['fresh_test'][stage]['success_rate'] for m in MODES]
        axes[0].bar(x+offset,[v['mean']*100 for v in values],.34,yerr=[v['sample_sd']*100 for v in values],label=label,color=color,capsize=3)
        for i,v in enumerate(values):axes[0].scatter(np.full(3,i+offset),np.asarray(v['values'])*100,s=15,color='#233440',zorder=4)
    axes[0].set(xticks=x,xticklabels=LABELS,ylabel='Goal reached (%)',ylim=(0,100),title='New held-out test: 200 paired maps / seed')
    axes[0].legend(frameon=False)
    steps=np.asarray(result['steps'])
    for mode,label,color in zip(MODES,LABELS,COLORS):
        values=[result['groups'][mode]['validation'][str(s)]['success_rate'] for s in steps]
        mean=np.array([v['mean'] for v in values])*100;sd=np.array([v['sample_sd'] for v in values])*100
        axes[1].plot(steps,mean,'o-',label=label,color=color)
        axes[1].fill_between(steps,mean-sd,mean+sd,color=color,alpha=.13)
    axes[1].set(xlabel='Cumulative environment steps',ylabel='Goal reached (%)',ylim=(0,100),title='Fixed validation maps: learning trend')
    axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Longer reward training · same real subgraph and hyperparameters · 3 seeds',fontsize=12)
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.8),layout='constrained')
    for ax,mode,label in zip(axes,MODES,LABELS):
        base=np.zeros(2)
        for metric,color,name in [('successes','#277A72','Success'),('collisions','#BD6864','Collision'),('timeouts','#BDA06A','Timeout')]:
            heights=np.array([result['groups'][mode]['fresh_test'][stage][metric]['mean']/2 for stage in ['before','after']])
            ax.bar([0,1],heights,bottom=base,color=color,label=name);base+=heights
        ax.set(xticks=[0,1],xticklabels=['65,536','262,144'],ylim=(0,100),title=label,xlabel='Training steps')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Test episodes (%)');axes[-1].legend(frameon=False,loc='upper left',bbox_to_anchor=(1,1))
    fig.suptitle('Success, collision and timeout on the same 200 new maps per seed')
    fig.savefig(OUT/'failure-types.png',dpi=180);plt.close(fig)
    print(OUT/'comparison.png')


if __name__=='__main__':main()
