"""Shared imitation / DAgger / recurrent PPO experiment for four controllers."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flylab.learning import LearningPolicy, LearningWorlds
from flylab.population_ppo import advantages
from flylab.teacher import GeometryTeacher
from scripts.train_population import write_json, summarize


class Demonstrations:
    def __init__(self, path):
        data = np.load(path)
        self.episodes = [(data['observations'][a:b], data['actions'][a:b])
                         for a, b in zip(data['offsets'][:-1], data['offsets'][1:])]

    def sample(self, rng, batch, length, burn):
        x = np.zeros((burn+length,batch,23),np.float32)
        y = np.zeros((burn+length,batch,2),np.float32)
        starts = np.zeros((burn+length,batch),bool)
        mask = np.zeros((burn+length,batch),np.float32)
        for i in range(batch):
            observations, actions = self.episodes[int(rng.integers(len(self.episodes)))]
            offset = int(rng.integers(len(observations)))
            for t in range(burn+length):
                j = offset-burn+t
                if 0 <= j < len(observations):
                    x[t,i] = observations[j]; y[t,i] = actions[j]; mask[t,i] = 1
                    starts[t,i] = j == 0
        return x,y,starts,mask


def save(folder, name, model, config, optimizer=None):
    state={'model':model.state_dict(),'config':config}
    if optimizer is not None: state['optimizer']=optimizer.state_dict()
    path=folder/f'{name}.pt'; temporary=path.with_suffix('.pending')
    torch.save(state,temporary);temporary.replace(path)


def parameter_delta(model, original):
    return {name:float((p.detach().cpu()-original[name]).norm()) for name,p in model.named_parameters()}


@torch.no_grad()
def evaluate(model, scenes, batch, condition='clean'):
    rows=[]; device=model.obs_mean.device
    for offset in range(0,len(scenes),batch):
        selected=scenes[offset:offset+batch]
        worlds=LearningWorlds(len(selected),0,selected,condition)
        raw=worlds.observe(); hidden=model.initial(len(selected))
        starts=torch.ones(len(selected),device=device,dtype=torch.bool)
        finished=np.zeros(len(selected),bool)
        for _ in range(600):
            mean,_,hidden=model(torch.as_tensor(raw,device=device),hidden,starts,condition)
            raw,_,done,episodes=worlds.step(torch.tanh(mean).cpu().numpy(),autoreset=False)
            for row in episodes: row.pop('slot');rows.append(row)
            finished|=done
            if finished.all():break
            starts=torch.as_tensor(done,device=device)
    assert len(rows)==len(scenes)
    return {'condition':condition,'summary':summarize(rows),'episodes':rows}


def imitate(model, data, optimizer, rng, updates, args, log, stage):
    device=model.obs_mean.device;start=time.perf_counter()
    for update in range(updates):
        arrays=data.sample(rng,args.batch,args.sequence,8)
        x,y,starts,mask=[torch.as_tensor(a,device=device) for a in arrays]
        hidden=model.initial(args.batch)
        with torch.no_grad(): _,_,hidden=model.sequence(x[:8],hidden,starts[:8])
        mean,_,_=model.sequence(x[8:],hidden.detach(),starts[8:])
        # Equal sampling objective across models; turns receive bounded emphasis.
        importance=(1+2*y[8:,:,1].abs())*mask[8:]
        error=(torch.tanh(mean)-y[8:]).square().mean(-1)
        loss=(error*importance).sum()/importance.sum().clamp_min(1)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite imitation loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if update%32==0 or update==updates-1:
            torch.cuda.synchronize();row={'stage':stage,'update':update+1,'loss':float(loss.detach()),
                'seconds':time.perf_counter()-start,'gpu_peak_mib':torch.cuda.max_memory_allocated()/2**20}
            log.append(row);write_json(args.output/'progress.json',row);print(json.dumps(row),flush=True)


@torch.no_grad()
def dagger(model, data, args):
    """Student visits states; teacher labels those states. No teacher action execution."""
    device=model.obs_mean.device;rows=[];added=[]
    seeds=list(range(10100000,10100000+args.dagger_episodes))
    for offset in range(0,len(seeds),args.batch):
        selected=seeds[offset:offset+args.batch]
        worlds=LearningWorlds(len(selected),args.seed,selected)
        teachers=[GeometryTeacher(w) for w in worlds.worlds]
        raw=worlds.observe();hidden=model.initial(len(selected));starts=torch.ones(len(selected),device=device,dtype=torch.bool)
        buffers=[([],[]) for _ in selected];finished=np.zeros(len(selected),bool)
        for _ in range(600):
            for i,w in enumerate(worlds.worlds):
                if not finished[i]:
                    try: label=teachers[i].action(w)
                    except RuntimeError:
                        # Record failure and stop this labeling episode explicitly.
                        raise RuntimeError(f'Teacher could not label student state in scene {w.seed}')
                    buffers[i][0].append(raw[i].copy());buffers[i][1].append(label)
            mean,_,hidden=model(torch.as_tensor(raw,device=device),hidden,starts)
            raw,_,done,episodes=worlds.step(torch.tanh(mean).cpu().numpy(),autoreset=False)
            for row in episodes:row.pop('slot');rows.append(row)
            finished|=done
            if finished.all():break
            starts=torch.as_tensor(done,device=device)
        added.extend([(np.asarray(x),np.asarray(y)) for x,y in buffers])
    data.episodes.extend(added)
    offsets=np.cumsum([0]+[len(x) for x,_ in added])
    np.savez_compressed(args.output/'dagger.npz',observations=np.concatenate([x for x,_ in added]),
                        actions=np.concatenate([y for _,y in added]),offsets=offsets,scene_seeds=np.asarray(seeds))
    write_json(args.output/'dagger-collection.json',{'summary':summarize(rows),'episodes':rows})


def train_ppo(model, args, log):
    device=model.obs_mean.device;T,B,L=args.rollout,args.batch,args.sequence
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,eps=1e-5)
    worlds=LearningWorlds(B,args.seed);raw=worlds.observe();obs=torch.as_tensor(raw,device=device)
    hidden=model.initial(B);starts=torch.ones(B,dtype=torch.bool,device=device)
    episodes=[]; start=time.perf_counter(); history=[]
    for update in range(args.ppo_steps//(T*B)):
        storage={k:[] for k in ('obs','hidden','starts','latent','logp','value','reward','done')}
        for t in range(T):
            storage['obs'].append(obs);storage['hidden'].append(hidden.detach());storage['starts'].append(starts)
            with torch.no_grad():
                mean,value,hidden=model(obs,hidden,starts)
                latent=model.distribution(mean).sample();logp=model.log_prob(mean,latent)
            raw,reward,done,rows=worlds.step(torch.tanh(latent).cpu().numpy());episodes.extend(rows)
            for key,value_ in [('latent',latent),('logp',logp),('value',value),
                    ('reward',torch.as_tensor(reward,device=device)),('done',torch.as_tensor(done,device=device))]:storage[key].append(value_)
            obs=torch.as_tensor(raw,device=device);starts=torch.as_tensor(done,device=device)
        data={k:torch.stack(v) for k,v in storage.items()}
        with torch.no_grad():
            _,last_value,_=model(obs,hidden,starts)
            adv,target=advantages(data['reward'],data['value'],data['done'],last_value)
            data['adv']=(adv-adv.mean())/(adv.std()+1e-8);data['target']=target
        chunks={k:v.reshape(T//L,L,B,*v.shape[2:]).transpose(1,2).reshape(-1,L,*v.shape[2:]) for k,v in data.items()}
        kl=[];losses=[]
        for epoch in range(2):
            epoch_kl=[]
            for indices in torch.randperm(len(chunks['obs']),device=device).chunk(4):
                mb={k:v[indices].transpose(0,1) for k,v in chunks.items()}
                mean,value,_=model.sequence(mb['obs'],mb['hidden'][0],mb['starts'])
                logratio=model.log_prob(mean,mb['latent'])-mb['logp'];ratio=logratio.exp()
                actor=torch.maximum(-mb['adv']*ratio,-mb['adv']*ratio.clamp(.8,1.2)).mean()
                loss=actor+.5*(value-mb['target']).square().mean()-.001*model.distribution(mean).entropy().sum(-1).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
                k=float(((ratio-1)-logratio).mean().detach());kl.append(k);epoch_kl.append(k);losses.append(float(loss.detach()))
            if np.mean(epoch_kl)>.03:break
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        row={'stage':'ppo','update':update+1,'steps':(update+1)*T*B,'seconds':elapsed,
             'loss':float(np.mean(losses)),'approx_kl':float(np.mean(kl)),
             'recent_success_rate':float(np.mean([e['status']=='success' for e in episodes[-100:]])) if episodes else 0,
             'gpu_peak_mib':torch.cuda.max_memory_allocated()/2**20}
        history.append(row);write_json(args.output/'progress.json',row)
        if update%16==0:print(json.dumps(row),flush=True)
    write_json(args.output/'ppo-training.json',history);write_json(args.output/'ppo-episodes.json',episodes)
    log.append(row)
    return optimizer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['mlp','gru','fly','shuffled'],required=True);p.add_argument('--seed',type=int,default=61)
    p.add_argument('--device',default='cuda:1');p.add_argument('--batch',type=int,default=8)
    p.add_argument('--sequence',type=int,default=16);p.add_argument('--rollout',type=int,default=32)
    p.add_argument('--bc-updates',type=int,default=256);p.add_argument('--dagger-updates',type=int,default=128)
    p.add_argument('--dagger-episodes',type=int,default=32);p.add_argument('--ppo-steps',type=int,default=32768)
    p.add_argument('--eval-episodes',type=int,default=100);p.add_argument('--eval-start',type=int,default=4300000)
    p.add_argument('--conditions',nargs='+',default=['clean','noise','delay']);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory.')
    if args.batch%4 or args.rollout%args.sequence or args.ppo_steps%(args.batch*args.rollout):p.error('Invalid batch/sequence/step divisibility')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False;rng=np.random.default_rng(args.seed)
    dataset=ROOT/'results/learning/train.npz';data=Demonstrations(dataset);raw=np.load(dataset)['observations']
    model=LearningPolicy(args.kind,raw.mean(0),raw.std(0)).to(args.device)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-5)
    files=[ROOT/'flylab/learning.py',ROOT/'flylab/teacher.py',ROOT/'flylab/world.py',ROOT/'flylab/population_ppo.py',Path(__file__)]
    config={**vars(args),'output':str(args.output),'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
            'parameters':sum(p.numel() for p in model.parameters()),'brain':model.brain_info,
            'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'source_hashes':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            'torch':torch.__version__,'ppo_learning_rate':1e-4,'bc_learning_rate':1e-3,
            'train_map_range':[10000000,10999999],'validation_maps':[4200000,4200031],
            'test_maps':[args.eval_start,args.eval_start+args.eval_episodes-1],
            'critic':'separate raw-observation MLP, same for all variants','brain_actor_range_bypass':False}
    write_json(args.output/'run.json',config);save(args.output,'initial',model,config)
    log=[];start=time.perf_counter()
    try:
        imitate(model,data,optimizer,rng,args.bc_updates,args,log,'bc')
        save(args.output,'bc',model,config,optimizer)
        validation=[{'stage':'bc',**evaluate(model,list(range(4200000,4200032)),args.batch)}]
        write_json(args.output/'validation.json',validation)
        dagger(model,data,args)
        imitate(model,data,optimizer,rng,args.dagger_updates,args,log,'dagger')
        save(args.output,'imitation',model,config,optimizer)
        validation.append({'stage':'imitation',**evaluate(model,list(range(4200000,4200032)),args.batch)})
        write_json(args.output/'validation.json',validation)
        # Predeclared final imitation and final PPO checkpoints, no test selection.
        scenes=list(range(args.eval_start,args.eval_start+args.eval_episodes))
        results=[]
        for condition in args.conditions:
            results.append({'stage':'imitation',**evaluate(model,scenes,args.batch,condition)})
            write_json(args.output/'evaluation.json',results)
        optimizer=train_ppo(model,args,log)
        save(args.output,'policy',model,config,optimizer)
        for condition in args.conditions:
            result={'stage':'ppo',**evaluate(model,scenes,args.batch,condition)}
            results.append(result);write_json(args.output/'evaluation.json',results)
            print(json.dumps({'stage':'ppo','condition':condition,**result['summary']}),flush=True)
        if model.brain_info:
            config['edge_weights_unchanged']=model.edge_digest()==model.brain_info['weight_digest']
            assert config['edge_weights_unchanged']
        config.update(status='complete',elapsed_seconds=time.perf_counter()-start,
                      parameter_changes=parameter_delta(model,initial),gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20)
        write_json(args.output/'training.json',log);write_json(args.output/'run.json',config);save(args.output,'policy',model,config,optimizer)
    except BaseException as exc:
        config.update(status='failed',error=repr(exc));write_json(args.output/'run.json',config);raise


if __name__=='__main__':main()
