"""Record one preselected map with the highest observed 4M policy, without training."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flylab.plasticity import PlasticPolicy,PlasticityWorlds,GRAPH
from flylab.controllers import base_action,compose
from flylab.world import Action


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'results/plasticity/staged-4m/main/staged-72/final.pt')
    parser.add_argument('--scene',type=int,default=5900000)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    checkpoint=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    assert checkpoint['config']['seed']==72 and checkpoint['total_steps']==4194304
    assert checkpoint['config']['graph_sha256']==sha(GRAPH)
    model=PlasticPolicy('readout').to(args.device)
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    before=model.edge_weight.detach().clone()
    env=PlasticityWorlds(1,0,[args.scene])
    raw=env.observe();hidden=model.initial(1)
    starts=torch.ones(1,dtype=torch.bool,device=args.device)
    world=env.worlds[0]
    initial=world.state()
    snapshots=[initial];commands=[];activity=[];episode=None
    started=time.perf_counter()
    for _ in range(600):
        mean,_,hidden,_=model(torch.as_tensor(raw,device=args.device),hidden,starts)
        action=mean.tanh().cpu().numpy()
        base=base_action(world.observe())
        final,residual=compose(base,Action(float(action[0,0])*.75,float(action[0,1])*1.8))
        commands.append({'base':[base.linear,base.angular],'residual':[residual.linear,residual.angular],
                         'final':[final.linear,final.angular],'policy_action':action[0].tolist()})
        activity.append(hidden[0,model.descending].cpu().numpy().copy())
        raw,_,done,episodes=env.step(action,autoreset=False)
        assert world.last_action==final
        state=world.state();state.pop('path');snapshots.append(state)
        if done[0]:
            episode=episodes[0];episode.pop('slot');break
        starts=torch.as_tensor(done,device=args.device)
    assert episode is not None and torch.equal(before,model.edge_weight)
    recorded={'scene_seed':args.scene,'scenario':'clutter','scene_choice':'fixed before this single rollout',
              'rollout_attempts':1,'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint),
              'training_seed':72,'training_steps':4194304,'method':'joint 65536 then frozen brain; trained readout',
              'graph_sha256':sha(GRAPH),'neurons':model.n,'edges':len(model.source),'descending':len(model.descending),
              'condition':'clean','action_selection':'tanh of Gaussian mean; no exploration noise',
              'device':args.device,'torch_version':str(torch.__version__),'control_dt':world.dt,
              'inference_seconds':time.perf_counter()-started,'result':episode,'states':snapshots,'commands':commands,
              'final_path':world.path,'source_sha256':sha(__file__)}
    (args.output/'rollout.json').write_text(json.dumps(recorded,ensure_ascii=False,indent=2))
    np.savez_compressed(args.output/'neural-activity.npz',descending=np.asarray(activity,dtype=np.float32))
    print(json.dumps({'output':str(args.output),'result':episode,'inference_seconds':recorded['inference_seconds']},ensure_ascii=False))


if __name__=='__main__':main()
