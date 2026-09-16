"""Exploratory DN dependence check and a fixed-map replay after the main trial.

These checks are not used to choose checkpoints or tune training. They were
added after the first readout run was seen, and are labeled post hoc explicitly.
"""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flylab.plasticity import PlasticPolicy, PlasticityWorlds
from scripts.train_plasticity import evaluate
from scripts.train_population import write_json


@torch.no_grad()
def main():
    torch.set_num_threads(4); torch.cuda.set_device(0)
    root=ROOT/'results/plasticity'; result={};paths={}
    for mode in ['readout','ppo_edges','three_factor']:
        checkpoint=torch.load(root/f'main/{mode}-71/final.pt',map_location='cuda:0',weights_only=True)
        model=PlasticPolicy(mode).to('cuda:0');model.load_state_dict(checkpoint['model'])
        env=PlasticityWorlds(1,0,[5300000]);raw=env.observe();hidden=model.initial(1)
        latents=[];actions=[];speeds=[]
        for _ in range(600):
            mean,_,hidden,_=model(torch.as_tensor(raw,device='cuda:0'),hidden)
            action=mean.tanh().cpu().numpy()
            latents.append(mean.cpu().numpy()[0]);actions.append(action[0])
            raw,_,done,_=env.step(action,autoreset=False)
            speeds.append(env.worlds[0].last_action.linear)
            if done.all():break
        paths[mode]=env.worlds[0].state()
        paths[mode]['action_diagnostics']={'mean_abs_latent':np.abs(latents).mean(0).tolist(),
            'fraction_linear_residual_below_minus_0_95':float((np.asarray(actions)[:,0]<-.95).mean()),
            'fraction_speed_below_0_05_mps':float((np.asarray(speeds)<.05).mean()),
            'mean_speed_mps':float(np.mean(speeds))}
        model.actor.weight[:,:len(model.descending)]=0
        result[mode]=evaluate(model,list(range(5300000,5300100)))
    write_json(root/'silent-dn-seed71.json',{'status':'post-hoc exploratory dependence check',
        'intervention':'Zero actor weights on calibrated DN channels, retain learned context weights and bias; no learning.',
        'seed':71,'groups':result})
    write_json(root/'paths-seed71.json',{'map':5300000,'selection':'first test map, no success filtering','groups':paths})
    print(json.dumps({mode:r['summary'] for mode,r in result.items()},indent=2))


if __name__=='__main__':main()
