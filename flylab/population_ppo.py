"""Shared PPO implementation for MLP, GRU and frozen population features.

The GRU is replayed on contiguous sequences with stored initial states and
episode masks. All policies optimize the same squashed Gaussian action law.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from .controllers import base_action, compose
from .world import NavigationWorld, Action
from .population import PopulationBrain, observation_vector


class WorldBatch:
    def __init__(self, batch, seed, scenario='clutter', scene_seeds=None, condition='clean'):
        self.batch, self.scenario, self.condition = batch, scenario, condition
        self.rngs=[np.random.default_rng(np.random.SeedSequence([seed,i])) for i in range(batch)]
        self.worlds=[NavigationWorld() for _ in range(batch)]
        self.previous=[None]*batch
        self.delayed=[None]*batch
        self.jitter=np.zeros(batch)
        self.steps=np.zeros(batch,int)
        self.reset(np.ones(batch,bool),scene_seeds)

    def reset(self, mask, scene_seeds=None):
        for i in np.flatnonzero(mask):
            seed=int(scene_seeds[i]) if scene_seeds is not None else int(self.rngs[i].integers(0,1000000))
            self.worlds[i].reset(seed,self.scenario)
            self.previous[i]=None; self.delayed[i]=None
            self.jitter[i]=0; self.steps[i]=0
        return self.observe()

    def observe(self):
        features=[]
        for i,w in enumerate(self.worlds):
            raw=w.observe()
            if self.condition=='noise':
                # Sensor noise is deterministic for a given scene and step.
                rng=np.random.default_rng(np.random.SeedSequence([w.seed,w.steps,773]))
                raw['rays']=np.clip(np.asarray(raw['rays'])+rng.normal(0,.05,9),0,3).tolist()
            elif self.condition=='delay':
                old=self.delayed[i]
                self.delayed[i]=list(raw['rays'])
                if old is not None: raw['rays']=old
            features.append(observation_vector(raw,self.previous[i]))
            self.previous[i]=list(raw['rays'])
        return np.stack(features)

    def step(self, actions, autoreset=True):
        rewards=np.zeros(self.batch,np.float32)
        done=np.zeros(self.batch,bool)
        episodes=[]
        for i,(w,a) in enumerate(zip(self.worlds,actions)):
            if w.status!='running': continue
            before=w.last_action
            action,_=compose(base_action(w.observe()),Action(float(a[0])*.75,float(a[1])*1.8))
            _,r,terminated,truncated,_=w.step(action)
            rewards[i]=r-.005*float(np.dot(a,a))
            self.jitter[i]+=abs(action.angular-before.angular)
            self.steps[i]+=1
            done[i]=terminated or truncated
            if done[i]:
                episodes.append({'slot':i,'scene_seed':w.seed,'status':w.status,'steps':w.steps,
                    'task_return':w.reward,'path_length':w.path_length,'min_clearance':w.min_clearance,
                    'mean_angular_action_change':float(self.jitter[i]/self.steps[i])})
        if autoreset:
            # Reset only completed worlds; observing every slot happens once below.
            for i in np.flatnonzero(done):
                self.worlds[i].reset(int(self.rngs[i].integers(0,1000000)),self.scenario)
                self.previous[i]=None; self.delayed[i]=None; self.jitter[i]=0; self.steps[i]=0
        return self.observe(),rewards,done,episodes


class RunningNorm(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.register_buffer('mean',torch.zeros(size))
        self.register_buffer('var',torch.ones(size))
        self.register_buffer('count',torch.tensor(1e-4))

    @torch.no_grad()
    def update(self,x):
        n=x.shape[0]; total=self.count+n
        delta=x.mean(0)-self.mean
        self.var.copy_((self.var*self.count+x.var(0,unbiased=False)*n+
                       delta.square()*self.count*n/total)/total)
        self.mean.add_(delta*n/total); self.count.copy_(total)

    def forward(self,x):
        return ((x-self.mean)/torch.sqrt(self.var+1e-5)).clamp(-5,5)


class Policy(nn.Module):
    def __init__(self,kind,features):
        super().__init__()
        self.kind=kind
        # Approximately matched trainable counts (~350k), rather than giving
        # the population model a much larger readout than the baselines.
        width=240 if kind=='gru' else 576 if kind=='mlp' else 128
        self.width=width
        self.input=nn.Linear(features,width)
        self.core=nn.GRUCell(width,width) if kind=='gru' else nn.Linear(width,width)
        self.actor=nn.Linear(width,2); self.critic=nn.Linear(width,1)
        self.log_std=nn.Parameter(torch.full((2,),-.5))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.orthogonal_(m.weight,math.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight,.01)
        nn.init.orthogonal_(self.critic.weight,1.)

    def forward(self,x,hidden=None,starts=None):
        y=torch.tanh(self.input(x))
        if self.kind=='gru':
            if hidden is None: hidden=torch.zeros((len(x),self.width),device=x.device)
            if starts is not None: hidden=hidden*(1-starts.float()).unsqueeze(-1)
            hidden=self.core(y,hidden); y=hidden
        else:
            y=torch.tanh(self.core(y))
        return self.actor(y),self.critic(y).squeeze(-1),hidden

    def sequence(self,x,hidden,starts):
        if self.kind!='gru':
            mean,value,_=self(x.reshape(-1,x.shape[-1]))
            return mean.reshape(*x.shape[:2],2),value.reshape(x.shape[:2])
        means=[]; values=[]
        for xt,st in zip(x,starts):
            mean,value,hidden=self(xt,hidden,st)
            means.append(mean); values.append(value)
        return torch.stack(means),torch.stack(values)

    def distribution(self,mean):
        return torch.distributions.Normal(mean,self.log_std.clamp(-4,1).exp())

    def log_prob(self,mean,latent):
        # Stable log(1-tanh(z)^2), identical during rollout and update.
        correction=2*(math.log(2)-latent-torch.nn.functional.softplus(-2*latent))
        return (self.distribution(mean).log_prob(latent)-correction).sum(-1)


def advantages(rewards,values,dones,last_value,gamma=.99,lam=.95):
    """Finite-horizon task: collision, success and 600-step expiry end returns."""
    result=torch.zeros_like(rewards); last=torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        next_value=last_value if t==len(rewards)-1 else values[t+1]
        alive=1-dones[t].float()
        delta=rewards[t]+gamma*next_value*alive-values[t]
        last=delta+gamma*lam*alive*last; result[t]=last
    return result,result+values


class FeaturePipe:
    def __init__(self,kind,batch,device,seed):
        self.device=device
        self.brain=PopulationBrain(batch,device,wiring='shuffled' if kind=='shuffled' else 'real',seed=seed) \
            if kind in ('fly','shuffled') else None
        self.size=2*len(self.brain.dn)+5 if self.brain else 23

    @torch.no_grad()
    def __call__(self,raw,done=None,condition='clean'):
        x=torch.as_tensor(raw,device=self.device)
        if self.brain:
            if done is not None:
                self.brain.reset(torch.as_tensor(done,device=self.device))
            x=self.brain.advance(x,disconnected=condition=='disconnected')
            if condition=='silent': x[:,:-5]=0
        return x
