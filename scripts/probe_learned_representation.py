"""Diagnose raw vs per-frame-normalized neural signals; no policy updates."""
from pathlib import Path
import argparse
import hashlib
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flylab.learning import LearningPolicy
from scripts.probe_sensory import metrics
from scripts.train_population import write_json


@torch.no_grad()
def collect(model,path):
    data=np.load(path);offsets=data['offsets'];features=[];targets=[];actions=[];labels=[]
    for begin in range(0,len(offsets)-1,8):
        episodes=[data['observations'][offsets[i]:offsets[i+1]] for i in range(begin,min(begin+8,len(offsets)-1))]
        hidden=model.initial(len(episodes))
        for t in range(max(map(len,episodes))):
            raw=np.zeros((len(episodes),23),np.float32);valid=[]
            for i,episode in enumerate(episodes):
                if t<len(episode):raw[i]=episode[t];valid.append(i)
            mean,_,hidden=model(torch.as_tensor(raw,device=model.obs_mean.device),hidden)
            actions.append(torch.tanh(mean[valid]).cpu())
            labels.append(torch.as_tensor(np.stack([data['actions'][offsets[begin+i]+t] for i in valid])))
            if t%3==0:
                features.append(hidden[valid][:,model.descending].cpu().clone())
                targets.append(torch.as_tensor(raw[valid,:18]).clone())
        print(path.name,begin+len(episodes),'episodes',flush=True)
    a,y=torch.cat(actions),torch.cat(labels)
    return torch.cat(features),torch.cat(targets),{
        'action_mse':float((a-y).square().mean()),
        'action_mse_per_channel':(a-y).square().mean(0).tolist(),
        'zero_residual_mse':float(y.square().mean()),'frames':len(y)}


def predict(x,y,xt):
    mean=x.mean(0);std=x.std(0).clamp_min(1e-7)
    a=torch.cat([(x-mean)/std,torch.ones(len(x),1)],1)
    b=torch.cat([(xt-mean)/std,torch.ones(len(xt),1)],1)
    penalty=torch.eye(a.shape[1])*10;penalty[-1,-1]=0
    return b@torch.linalg.solve(a.T@a+penalty,a.T@y)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    root=ROOT/'results/learning';file=root/'main/fly-61/imitation.pt'
    checkpoint=torch.load(file,map_location='cpu',weights_only=False)
    model=LearningPolicy('fly').to(args.device);model.load_state_dict(checkpoint['model']);model.eval()
    assert model.brain_info['weight_digest']==checkpoint['config']['brain']['weight_digest']
    started=time.perf_counter();x,y,train_action=collect(model,root/'train.npz');xt,yt,val_action=collect(model,root/'validation.npz')
    normalized=torch.nn.functional.layer_norm(x,(x.shape[-1],))
    normalized_t=torch.nn.functional.layer_norm(xt,(xt.shape[-1],))
    amplitude=torch.stack([x.mean(1),x.std(1)],1);amplitude_t=torch.stack([xt.mean(1),xt.std(1)],1)
    result={}
    for name,a,b in [('raw_dn',x,xt),('normalized_dn',normalized,normalized_t),
                     ('amplitude_only',amplitude,amplitude_t),
                     ('normalized_plus_amplitude',torch.cat([normalized,amplitude],1),torch.cat([normalized_t,amplitude_t],1))]:
        result[name]=metrics(predict(a,y,b),yt);print(name,result[name],flush=True)
    write_json(root/'learned-representation-probe.json',{'checkpoint':str(file.relative_to(ROOT)),
        'stage':'imitation','training_seed':61,'results':result,'seconds':time.perf_counter()-started,
        'device':args.device,'ridge_lambda':10,'stride':3,'feature_std_floor':1e-7,
        'train_episodes':128,'validation_episodes':32,'layer_norm_eps':1e-5,
        'checkpoint_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
        'dataset_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in ('train.npz','validation.npz')},
        'per_frame_dn_std_mean':float(x.std(1).mean()),'across_frames_dn_std_mean':float(x.std(0).mean()),
        'normalized_across_frames_dn_std_mean':float(normalized.std(0).mean()),
        'teacher_trajectory_action_error':{'train':train_action,'validation':val_action},
        'note':'Post hoc linear probes on whole held-out validation episodes; not a policy intervention.'})


if __name__=='__main__':main()
