"""Reload final models, test reliance on the brain, and export fixed-map paths."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flylab.learning import LearningPolicy,LearningWorlds
from scripts.train_learning import evaluate
from scripts.train_population import write_json,evaluate as evaluate_old
from flylab.population_ppo import Policy,RunningNorm


@torch.no_grad()
def paths(model):
    seeds=[4300000,4300001,4300002]
    worlds=LearningWorlds(3,0,seeds);raw=worlds.observe();hidden=model.initial(3)
    starts=torch.ones(3,dtype=torch.bool,device=model.obs_mean.device);finished=np.zeros(3,bool)
    for _ in range(600):
        mean,_,hidden=model(torch.as_tensor(raw,device=model.obs_mean.device),hidden,starts)
        raw,_,done,_=worlds.step(torch.tanh(mean).cpu().numpy(),autoreset=False)
        finished|=done
        if finished.all():break
        starts=torch.as_tensor(done,device=model.obs_mean.device)
    return [w.state() for w in worlds.worlds]


def main():
    torch.set_num_threads(1);torch.cuda.set_device(1)
    root=ROOT/'results/learning';mechanism={};trajectories={};validation=[]
    for kind in ('mlp','gru','fly','shuffled'):
        file=root/'main'/f'{kind}-61/policy.pt'
        checkpoint=torch.load(file,map_location='cpu',weights_only=False)
        assert checkpoint['config']['status']=='complete'
        model=LearningPolicy(kind).to('cuda:1');model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        if model.brain_info:
            assert model.brain_info['weight_digest']==checkpoint['config']['brain']['weight_digest']
        trajectories[kind]=paths(model)
        if kind in ('fly','shuffled'):
            results=[]
            for condition in ('disconnected','silent'):
                result=evaluate(model,list(range(4300000,4300100)),8,condition)
                results.append(result);print(kind,condition,result['summary'],flush=True)
            initial=torch.load(root/'main'/f'{kind}-61/initial.pt',map_location='cpu',weights_only=False)['model']
            with torch.no_grad():
                for name in ('gain_raw','bias','leak_raw'):getattr(model,name).copy_(initial[name].to('cuda:1'))
            result=evaluate(model,list(range(4300000,4300100)),8)
            result['condition']='restore_initial_dynamics';results.append(result)
            mechanism[kind]=results
            print(kind,'restore_initial_dynamics',result['summary'],flush=True)
        validation.append({'run':f'{kind}-61','strict_checkpoint_reload':True,'fixed_scene_rollouts':3})
        del model,checkpoint;torch.cuda.empty_cache()
    write_json(root/'mechanism.json',{'seed':61,'results':mechanism,'caution':'Interventions shift the feature distribution; single training seed.'})
    write_json(root/'paths.json',trajectories);write_json(root/'reload-check.json',validation)
    old=[]
    for seed in (11,22,33):
        checkpoint=torch.load(ROOT/f'results/population/main/fly-{seed}/policy.pt',map_location='cpu',weights_only=False)
        config=checkpoint['config'];model=Policy('fly',config['observation_dim']).to('cuda:1')
        model.load_state_dict(checkpoint['policy']);norm=RunningNorm(config['observation_dim']).to('cuda:1')
        norm.load_state_dict(checkpoint['normalizer'])
        result=evaluate_old(model,norm,'fly','cuda:1',seed,list(range(4300000,4300100)),'clutter',32)
        old.append({'seed':seed,**result});print('old fly',seed,result['summary'],flush=True)
        del model,norm,checkpoint;torch.cuda.empty_cache()
    write_json(root/'old-fly-test.json',old)


if __name__=='__main__':main()
