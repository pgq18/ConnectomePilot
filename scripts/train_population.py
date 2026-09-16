"""Train and evaluate one reproducible population-control comparison run."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flylab.population_ppo import WorldBatch, FeaturePipe, Policy, RunningNorm, advantages


def write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending'); tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)); tmp.replace(path)


def save(path,policy,norm,config,optimizer=None):
    data={'policy':policy.state_dict(),'normalizer':norm.state_dict(),'config':config}
    if optimizer is not None: data['optimizer']=optimizer.state_dict()
    tmp=path.with_suffix('.pending'); torch.save(data,tmp); tmp.replace(path)


@torch.no_grad()
def evaluate(policy,norm,kind,device,seed,scenes,scenario,batch=32,condition='clean'):
    rows=[]
    pipe=FeaturePipe(kind,batch,device,seed+100000)
    for offset in range(0,len(scenes),batch):
        selected=scenes[offset:offset+batch]
        padded=selected+[selected[-1]]*(batch-len(selected))
        worlds=WorldBatch(batch,seed,scenario,padded,condition)
        if pipe.brain: pipe.brain.reset()
        obs=norm(pipe(worlds.observe(),condition=condition))
        hidden=torch.zeros((batch,policy.width),device=device)
        starts=torch.ones(batch,device=device,dtype=torch.bool)
        finished=np.zeros(batch,bool)
        for _ in range(600):
            mean,_,hidden=policy(obs,hidden,starts)
            raw,_,done,episodes=worlds.step(torch.tanh(mean).cpu().numpy(),autoreset=False)
            for ep in episodes:
                slot=ep.pop('slot')
                if slot<len(selected) and not finished[slot]: rows.append(ep)
            finished|=done
            if finished.all(): break
            obs=norm(pipe(raw,condition=condition)); starts=torch.as_tensor(done,device=device)
    del pipe; gc.collect(); torch.cuda.empty_cache()
    return {'condition':condition,'episodes':rows,'summary':summarize(rows)}


def summarize(rows):
    return {'episodes':len(rows),'successes':sum(r['status']=='success' for r in rows),
        'collisions':sum(r['status']=='collision' for r in rows),
        'timeouts':sum(r['status']=='timeout' for r in rows),
        'success_rate':float(np.mean([r['status']=='success' for r in rows])),
        'mean_task_return':float(np.mean([r['task_return'] for r in rows])),
        'mean_angular_action_change':float(np.mean([r['mean_angular_action_change'] for r in rows]))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['mlp','gru','fly','shuffled'],required=True)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--steps',type=int,default=100352)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--rollout',type=int,default=64)
    p.add_argument('--sequence',type=int,default=32)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--scenario',default='clutter',choices=['clutter','single','slalom'])
    p.add_argument('--eval-episodes',type=int,default=100)
    p.add_argument('--eval-start',type=int,default=3000000)
    p.add_argument('--conditions',nargs='+',default=['clean'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=1800)
    args=p.parse_args()
    if args.output.exists(): p.error('Use a new output directory; existing runs are never overwritten.')
    if args.rollout%args.sequence or args.batch<4 or args.batch%4 or args.steps<args.batch*args.rollout:
        p.error('Require rollout divisible by sequence, batch divisible by 4, and at least one rollout.')
    if args.eval_start<1000000 or args.eval_episodes<1: p.error('Evaluation seeds must be held out.')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    worlds=WorldBatch(args.batch,args.seed,args.scenario)
    pipe=FeaturePipe(args.kind,args.batch,args.device,args.seed)
    policy=Policy(args.kind,pipe.size).to(args.device)
    norm=RunningNorm(pipe.size).to(args.device)
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4,eps=1e-5)
    brain_info=pipe.brain.info() if pipe.brain else None
    config={**vars(args),'output':str(args.output),'status':'training','schema':1,
        'started_at':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,
        'brain':brain_info,'observation_dim':pipe.size,'hidden_width':policy.width,
        'trainable_parameters':sum(p.numel() for p in policy.parameters()),
        'train_scene_range':[0,999999],'gamma':.99,'gae_lambda':.95,'learning_rate':3e-4,
        'clip':.2,'entropy_coefficient':.001,'value_coefficient':.5,'gradient_clip':.5,
        'minibatches':4,'finite_horizon_timeout_terminal':True,
        'source_hashes':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [ROOT/'flylab/population.py',ROOT/'flylab/population_ppo.py',Path(__file__)]}}
    write_json(args.output/'run.json',config)
    save(args.output/'initial.pt',policy,norm,config)
    raw=worlds.observe(); feats=pipe(raw); norm.update(feats); obs=norm(feats)
    hidden=torch.zeros((args.batch,policy.width),device=args.device)
    starts=torch.ones(args.batch,dtype=torch.bool,device=args.device)
    T,B,L=args.rollout,args.batch,args.sequence
    updates=math_ceil(args.steps/(T*B))
    history=[]; episode_rows=[]
    torch.cuda.synchronize(); start=time.perf_counter()
    try:
        for update in range(updates):
            storage={name:[] for name in ['obs','hidden','starts','latent','logp','value','reward','done']}
            for t in range(T):
                storage['obs'].append(obs); storage['hidden'].append(hidden.detach())
                storage['starts'].append(starts)
                with torch.no_grad():
                    mean,value,hidden=policy(obs,hidden,starts)
                    latent=policy.distribution(mean).sample()
                    action=torch.tanh(latent)
                    logp=policy.log_prob(mean,latent)
                raw,reward,done,episodes=worlds.step(action.cpu().numpy())
                episode_rows.extend(episodes)
                for name,v in [('latent',latent),('logp',logp),('value',value),
                               ('reward',torch.as_tensor(reward,device=args.device)),
                               ('done',torch.as_tensor(done,device=args.device))]: storage[name].append(v)
                feats=pipe(raw,done); norm.update(feats); obs=norm(feats)
                starts=torch.as_tensor(done,device=args.device)
            data={k:torch.stack(v) for k,v in storage.items()}
            with torch.no_grad():
                _,last_value,_=policy(obs,hidden,starts)
                adv,target=advantages(data['reward'],data['value'],data['done'],last_value)
                data['adv']=(adv-adv.mean())/(adv.std()+1e-8); data['target']=target
            # Every sample belongs to exactly one contiguous sequence. Stored
            # hidden state initializes each truncated-BPTT sequence.
            chunks={k:v.reshape(T//L,L,B,*v.shape[2:]).transpose(1,2).reshape(-1,L,*v.shape[2:])
                    for k,v in data.items()}
            count=len(chunks['obs']); kl=[]; losses=[]; clipfractions=[]
            for epoch in range(args.epochs):
                order=torch.randperm(count,device=args.device)
                epoch_kl=[]
                for indices in order.chunk(4):
                    mb={k:v[indices].transpose(0,1) for k,v in chunks.items()}
                    mean,value=policy.sequence(mb['obs'],mb['hidden'][0],mb['starts'])
                    logp=policy.log_prob(mean,mb['latent'])
                    logratio=logp-mb['logp']; ratio=logratio.exp()
                    pg=torch.maximum(-mb['adv']*ratio,-mb['adv']*ratio.clamp(.8,1.2)).mean()
                    vl=.5*(value-mb['target']).square().mean()
                    entropy=policy.distribution(mean).entropy().sum(-1).mean()
                    loss=pg+.5*vl-.001*entropy
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(),.5); optimizer.step()
                    k=float(((ratio-1)-logratio).mean().detach())
                    epoch_kl.append(k); kl.append(k); losses.append(float(loss.detach()))
                    clipfractions.append(float(((ratio-1).abs()>.2).float().mean()))
                if np.mean(epoch_kl)>.03: break
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            steps=(update+1)*T*B
            recent=episode_rows[-100:]
            row={'update':update+1,'steps':steps,'seconds':elapsed,'steps_per_second':steps/elapsed,
                 'loss':float(np.mean(losses)),'approx_kl':float(np.mean(kl)),
                 'clip_fraction':float(np.mean(clipfractions)),
                 'recent_success_rate':float(np.mean([r['status']=='success' for r in recent])) if recent else 0.,
                 'episodes':len(episode_rows),'gpu_peak_mib':torch.cuda.max_memory_allocated()/2**20}
            history.append(row); write_json(args.output/'progress.json',row)
            if (update+1)%5==0 or update==0:
                print(json.dumps({'kind':args.kind,'seed':args.seed,**row}),flush=True)
                save(args.output/'checkpoint.pt',policy,norm,{**config,'steps_actual':steps},optimizer)
            if elapsed>args.max_seconds: raise TimeoutError('Training time budget exceeded; checkpoint retained')
        config.update(status='evaluating',steps_actual=steps,training_seconds=elapsed,
                      gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        save(args.output/'policy.pt',policy,norm,config,optimizer)
        with (args.output/'training.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
        write_json(args.output/'training-episodes.json',episode_rows)
        if pipe.brain:
            config['brain_hash_after']=pipe.brain.current_digest()
            config['brain_weights_unchanged']=config['brain_hash_after']==config['brain']['weight_digest']
            assert config['brain_weights_unchanged']
        del pipe,worlds,storage,data,chunks; gc.collect(); torch.cuda.empty_cache()
        scenes=list(range(args.eval_start,args.eval_start+args.eval_episodes))
        results=[]
        for condition in args.conditions:
            result=evaluate(policy,norm,args.kind,args.device,args.seed,scenes,args.scenario,args.batch,condition)
            results.append(result); write_json(args.output/'evaluation.json',results)
            print(json.dumps({'condition':condition,**result['summary']}),flush=True)
        config['status']='complete'; write_json(args.output/'run.json',config)
        save(args.output/'policy.pt',policy,norm,config,optimizer)
    except BaseException as exc:
        config.update(status='failed',error=repr(exc))
        write_json(args.output/'run.json',config)
        raise


def math_ceil(x): return int(np.ceil(x))

if __name__=='__main__': main()
