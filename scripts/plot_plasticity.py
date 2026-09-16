"""Publication-independent static plots from audited measured results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[1]/'results/plasticity'
MODES=['readout','ppo_edges','three_factor']
LABELS=['PPO: readout only','PPO: edges + readout','Local reward plasticity']
COLORS=['#718AAB','#277A72','#B47A42']


def main():
    data=json.loads((ROOT/'comparison.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    x=np.arange(3)
    for ax,conditions,title in [(axes[0],['initial','clean'],'Before and after reward learning'),
                                 (axes[1],['clean','noise','delay'],'Final policy under sensor changes')]:
        width=.32 if len(conditions)==2 else .24
        offsets=(np.arange(len(conditions))-(len(conditions)-1)/2)*width
        for offset,condition,color in zip(offsets,conditions,['#91A8BE','#277A72','#B47A42']):
            vals=[data['groups'][m]['initial'] if condition=='initial' else data['groups'][m]['conditions'][condition]['success_rate'] for m in MODES]
            ax.bar(x+offset,[v['mean']*100 for v in vals],width*.92,yerr=[v['sample_sd']*100 for v in vals],
                   label={'initial':'Initial','clean':'Clean','noise':'Noise: 0.05 m','delay':'Delay: 100 ms'}[condition],color=color,capsize=3)
            for i,v in enumerate(vals):
                ax.scatter(np.full(3,i+offset),np.asarray(v['values'])*100,s=15,color='#253340',zorder=4)
        ax.set(xticks=x,xticklabels=['Readout PPO','Edge PPO','Local plasticity'],ylabel='Goal reached (%)',ylim=(0,100),title=title)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Real 4,096-neuron subgraph · 65,536 steps / seed · 3 seeds · 100 held-out maps',fontsize=12)
    fig.savefig(ROOT/'comparison.png',dpi=180);fig.savefig(ROOT/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for mode,label,color in zip(MODES,LABELS,COLORS):
        curves=[];returns=[]
        for seed in [71,72,73]:
            episodes=json.loads((ROOT/f'main/{mode}-{seed}/training.json').read_text())['episodes']
            bins=[ [r for r in episodes if step-4096<r['environment_steps']<=step] for step in range(4096,65537,4096)]
            curves.append([np.mean([r['status']=='success' for r in rows]) for rows in bins])
            returns.append([np.mean([r['task_return'] for r in rows]) for rows in bins])
        steps=np.arange(4096,65537,4096)
        for ax,values,scale in [(axes[0],curves,100),(axes[1],returns,1)]:
            values=np.asarray(values)*scale;avg=values.mean(0);sd=values.std(0,ddof=1)
            ax.plot(steps,avg,label=label,color=color);ax.fill_between(steps,avg-sd,avg+sd,color=color,alpha=.13)
            ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);ax.set_xlabel('Environment steps')
    axes[0].set(ylabel='Training episodes successful (%)',ylim=(0,100));axes[1].set_ylabel('Training task return')
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Exploratory training trajectories · mean ± sample SD across 3 seeds')
    fig.savefig(ROOT/'training-curves.png',dpi=180);plt.close(fig)
    if (ROOT/'paths-seed71.json').exists():
        data=json.loads((ROOT/'paths-seed71.json').read_text())
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for ax,mode,label,color in zip(axes,MODES,LABELS,COLORS):
            state=data['groups'][mode];path=np.asarray(state['path'])
            for ox,oy,r in state['obstacles']:
                ax.add_patch(Circle((ox,oy),r,color='#BFC6CE'))
                ax.add_patch(Circle((ox,oy),r+state['radius'],fill=False,color='#DDE1E5',lw=.7))
            ax.plot(path[:,0],path[:,1],color=color,lw=2)
            ax.scatter(*path[0],s=25,color='#263746');ax.scatter(*state['goal'],marker='*',s=100,color='#DDAE38')
            ax.set(xlim=(0,12),ylim=(0,8),aspect='equal',title=f"{label}\n{state['status']} · {state['steps']} steps")
        fig.suptitle('First held-out map 5300000 · training seed 71 · no outcome filtering')
        fig.savefig(ROOT/'paths.png',dpi=180);plt.close(fig)
    print(ROOT/'comparison.png');print(ROOT/'training-curves.png')


if __name__=='__main__':main()
