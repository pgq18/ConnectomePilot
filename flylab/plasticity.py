"""Three learning rules on one real graph, inspired by FlyDoom b047fab.

Independent robotics adaptation: fixed directional input, continuous Gaussian
residuals, calibrated linear readouts, fixed node biases, batched eligibility.
This is neither a biological fly nor an unmodified FlyDoom reproduction.
"""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from .population import observation_vector
from .population_ppo import WorldBatch

GRAPH = Path(__file__).resolve().parents[1] / 'data/plasticity/graph.npz'


class PlasticityWorlds(WorldBatch):
    def __init__(self, batch, seed, scene_seeds=None, condition='clean'):
        if scene_seeds is None:
            scene_seeds = [int(np.random.default_rng(np.random.SeedSequence([seed, i, 901])).integers(20000000, 21000000))
                           for i in range(batch)]
        super().__init__(batch, seed, 'clutter', scene_seeds, condition)

    def step(self, actions, autoreset=True):
        raw, rewards, done, episodes = super().step(actions, autoreset=False)
        if autoreset:
            for i in np.flatnonzero(done):
                self.worlds[i].reset(int(self.rngs[i].integers(20000000, 21000000)), 'clutter')
                self.previous[i] = None; self.delayed[i] = None; self.jitter[i] = 0; self.steps[i] = 0
                observation = self.worlds[i].observe()
                raw[i] = observation_vector(observation)
                self.previous[i] = observation['rays']
        return raw, rewards, done, episodes


class PlasticPolicy(nn.Module):
    def __init__(self, mode, graph=GRAPH):
        super().__init__()
        if mode not in ('readout', 'ppo_edges', 'three_factor'): raise ValueError(mode)
        self.mode = mode
        g = np.load(graph, allow_pickle=False) if isinstance(graph, (str, Path)) else graph
        self.n = len(g['global_indices'])
        for name in ('source', 'target', 'inputs', 'channels', 'descending'):
            self.register_buffer(name, torch.as_tensor(g[name], dtype=torch.long))
        w = torch.as_tensor(g['weight'], dtype=torch.float32)
        self.edge_weight = nn.Parameter(w.clone(), requires_grad=mode == 'ppo_edges')
        self.register_buffer('initial_weight', w.clone())
        self.register_buffer('sign', w.sign())
        self.register_buffer('dn_mean', torch.zeros(len(self.descending)))
        self.register_buffer('dn_std', torch.ones(len(self.descending)))
        self.actor = nn.Linear(len(self.descending) + 5, 2)
        self.critic = nn.Linear(len(self.descending) + 5, 1)
        nn.init.normal_(self.actor.weight, std=.01); nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.weight); nn.init.zeros_(self.critic.bias)
        self.std = .4
        if mode == 'three_factor':
            for p in self.parameters(): p.requires_grad_(False)

    def initial(self, batch):
        return self.edge_weight.new_zeros((batch, self.n))

    def forward(self, observations, hidden, starts=None):
        if starts is not None: hidden = hidden * (~starts).to(hidden.dtype).unsqueeze(1)
        sensory = torch.cat(((1-observations[:, :9]).clamp(0, 1), observations[:, 9:18].clamp(-1, 1)), 1)
        external = torch.zeros_like(hidden)
        external[:, self.inputs] = .9 * sensory[:, self.channels]
        for _ in range(5):
            messages = hidden[:, self.source] * self.edge_weight
            recurrent = torch.zeros_like(hidden).index_add(1, self.target, messages)
            hidden = torch.tanh(.5 * hidden + recurrent + external)
        features = self.features(observations, hidden)
        return self.actor(features), self.critic(features).squeeze(-1), hidden, features

    def features(self, observations, hidden):
        dn = ((hidden[:, self.descending] - self.dn_mean) / self.dn_std).clamp(-5, 5)
        # Keep aggregate feature energy comparable to five unscaled context inputs.
        return torch.cat((dn / math.sqrt(len(self.descending)), observations[:, 18:]), 1)

    def log_prob(self, mean, latent):
        correction = 2 * (math.log(2) - latent - torch.nn.functional.softplus(-2 * latent))
        return (torch.distributions.Normal(mean, self.std).log_prob(latent) - correction).sum(-1)

    @torch.no_grad()
    def project(self):
        # Boolean advanced indexing returns a copy: explicitly write into the base.
        self.edge_weight.copy_(self.sign * (self.sign * self.edge_weight).clamp(0, 1))


class ThreeFactor:
    """Per-environment traces; TD modulation, no autograd or PPO optimizer.

Gaussian score replaces FlyDoom's categorical action surprise. Eligibility uses
current pre/post rates as upstream does, not a spike-timing model. A scalar TD
signal broadcasts to all retained edges; no anatomical DAN or compartment model.
"""
    def __init__(self, model, batch, edge_lr=1e-4, actor_lr=2e-3, value_lr=5e-3):
        self.model = model
        self.edge_lr, self.actor_lr, self.value_lr = edge_lr, actor_lr, value_lr
        self.edge_trace = model.edge_weight.new_zeros((batch, len(model.source)))
        self.actor_trace = model.edge_weight.new_zeros((batch, 2, model.actor.in_features))
        self.bias_trace = model.edge_weight.new_zeros((batch, 2))

    @torch.no_grad()
    def reset(self, done):
        self.edge_trace[done] = 0; self.actor_trace[done] = 0; self.bias_trace[done] = 0

    @torch.no_grad()
    def update(self, hidden, features, mean, value, latent, reward, next_value, done):
        m = self.model
        delta = (.1 * reward + .99 * (~done) * next_value - value).clamp(-5, 5)
        post = hidden[:, m.target]
        hebb = hidden[:, m.source] * (post - post.mean(1, keepdim=True))
        self.edge_trace.mul_(.95).add_(hebb).clamp_(-5, 5)
        change = self.edge_lr * (delta[:, None] * self.edge_trace).mean(0)
        m.edge_weight.add_(change)
        m.edge_weight.add_(1e-6 * (m.initial_weight - m.edge_weight))
        m.project()
        score = (latent - mean) / m.std**2
        self.actor_trace.mul_(.95).add_(score[:, :, None] * features[:, None, :])
        self.bias_trace.mul_(.95).add_(score)
        m.actor.weight.add_(self.actor_lr * (delta[:, None, None] * self.actor_trace).mean(0)).clamp_(-3, 3)
        m.actor.bias.add_(self.actor_lr * (delta[:, None] * self.bias_trace).mean(0)).clamp_(-3, 3)
        m.critic.weight.add_(self.value_lr * (delta[:, None] * features).mean(0, keepdim=True)).clamp_(-3, 3)
        m.critic.bias.add_(self.value_lr * delta.mean()).clamp_(-3, 3)
        self.reset(done)
        return delta
