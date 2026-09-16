"""Render measured learning-stage scores and preselected navigation paths."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[1]


def main():
    root=ROOT/'results/learning'
    result=json.loads((root/'comparison.json').read_text())
    kinds=['mlp','gru','fly','shuffled'];labels=['MLP','GRU','Fly connectome','Shuffled connectome']
    fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
    x=np.arange(4)
    for stage,offset,color,label in [('imitation',-.18,'#91A8BE','Imitation + DAgger'),('ppo',.18,'#277A72','Then PPO')]:
        values=[result['groups'][k][stage]['clean']['success_rate'] for k in kinds]
        ax.bar(x+offset,[v['mean']*100 for v in values],.34,yerr=[v['sample_sd']*100 for v in values],
               color=color,label=label,capsize=4)
    ax.set(xticks=x,xticklabels=labels,ylabel='Goal reached (%)',ylim=(0,105),
           title='Held-out obstacle avoidance · 3 training seeds, 100 shared maps')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)
    paths=json.loads((root/'paths.json').read_text())
    fig,axes=plt.subplots(1,4,figsize=(13.5,3.3),layout='constrained')
    colors={'success':'#277A72','collision':'#B34E45','timeout':'#9C7D32'}
    for ax,kind,label in zip(axes,kinds,labels):
        world=paths[kind][0]
        for ox,oy,r in world['obstacles']:
            ax.add_patch(Circle((ox,oy),r,color='#BFC6CE'))
            ax.add_patch(Circle((ox,oy),r+world['radius'],fill=False,edgecolor='#DDE1E5',lw=.7))
        path=np.asarray(world['path']);ax.plot(path[:,0],path[:,1],color=colors[world['status']],lw=2)
        ax.scatter(*path[0],marker='o',s=25,color='#313B45');ax.scatter(*world['goal'],marker='*',s=80,color='#DEAB39')
        ax.set(xlim=(0,12),ylim=(0,8),aspect='equal',title=f"{label}\n{world['status']} · {world['steps']} steps")
    fig.suptitle('Preselected map 4300000 · PPO checkpoints from training seed 61')
    fig.savefig(root/'paths.png',dpi=180);plt.close(fig)
    print(root/'comparison.png');print(root/'paths.png')


if __name__=='__main__':main()
