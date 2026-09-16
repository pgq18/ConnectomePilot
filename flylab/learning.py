"""Differentiable, connectome-constrained rate dynamics for imitation + PPO.

This is an engineered continuous-state controller, not the previous LIF model.
All measured edges stay fixed; encoder, neuron dynamics and decoder can learn.
"""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy import sparse
from .population import input_map
from .population_ppo import WorldBatch

DATA = Path(__file__).resolve().parents[1] / 'data/male-cns'


class FixedSparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, transpose):
        ctx.save_for_backward(transpose)
        return torch.sparse.mm(weight, x)

    @staticmethod
    def backward(ctx, gradient):
        transpose, = ctx.saved_tensors
        return torch.sparse.mm(transpose, gradient.contiguous()), None, None


def tensor_csr(matrix):
    return torch.sparse_csr_tensor(torch.as_tensor(matrix.indptr, dtype=torch.int32),
                                   torch.as_tensor(matrix.indices, dtype=torch.int32),
                                   torch.as_tensor(matrix.data, dtype=torch.float32), size=matrix.shape)


def matrix_digest(matrix):
    h = hashlib.sha256()
    for a in (matrix.indptr, matrix.indices, matrix.data): h.update(memoryview(a).cast('B'))
    return h.hexdigest()


class LearningPolicy(nn.Module):
    def __init__(self, kind, mean=None, std=None, data=DATA, graph_override=None):
        super().__init__()
        self.kind = kind
        self.register_buffer('obs_mean', torch.zeros(23) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('obs_std', torch.ones(23) if std is None else torch.as_tensor(std, dtype=torch.float32).clamp_min(.1))
        self.brain_info = None
        if kind in ('fly', 'shuffled'):
            if graph_override is None:
                matrix = sparse.load_npz(Path(data) / 'weights.npz').tocsr().astype(np.float32)
                meta = np.load(Path(data) / 'brain.npz', allow_pickle=False)
                inputs, _ = input_map(meta['cell_type'], meta['side'], meta['ids'])
                descending = np.flatnonzero(meta['superclass'] == 'descending_neuron')
            else:
                matrix, inputs, descending = graph_override
                matrix = matrix.tocsr().astype(np.float32)
            if kind == 'shuffled': matrix = matrix[np.random.default_rng(20260915).permutation(matrix.shape[0])].tocsr()
            self.memory_size = matrix.shape[0]
            self.brain_info = {'neurons': matrix.shape[0], 'edges': matrix.nnz, 'inputs': len(inputs),
                               'descending': len(descending), 'weight_digest': matrix_digest(matrix),
                               'wiring': kind, 'neural_steps_per_action': 5, 'edge_weights_trainable': False}
            self.register_buffer('weight', tensor_csr(matrix), persistent=False)
            self.register_buffer('transpose', tensor_csr(matrix.T.tocsr()), persistent=False)
            self.register_buffer('inputs', torch.as_tensor(inputs, dtype=torch.long), persistent=False)
            self.register_buffer('descending', torch.as_tensor(descending, dtype=torch.long), persistent=False)
            self.encoder = nn.Sequential(nn.Linear(18, 64), nn.Tanh(), nn.Linear(64, len(inputs)), nn.Tanh())
            self.gain_raw = nn.Parameter(torch.full((self.memory_size,), -math.log(2)))  # gain=1
            self.leak_raw = nn.Parameter(torch.zeros(self.memory_size))  # retention=.5
            self.bias = nn.Parameter(torch.zeros(self.memory_size))
            self.readout = nn.Sequential(nn.Linear(len(descending) + 5, 128), nn.Tanh(),
                                         nn.Linear(128, 128), nn.Tanh())
            width = 128
        else:
            width = 352 if kind == 'gru' else 864
            self.memory_size = width if kind == 'gru' else 1
            self.encoder = nn.Sequential(nn.Linear(23, width), nn.Tanh())
            self.core = nn.GRUCell(width, width) if kind == 'gru' else nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.actor = nn.Linear(width, 2)
        # All variants share the same raw-observation value estimator; actor
        # gradients alone train the connectome. The actor has no range bypass.
        self.critic = nn.Sequential(nn.Linear(23, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((2,), -1.5))
        nn.init.orthogonal_(self.actor.weight, .01); nn.init.zeros_(self.actor.bias)

    def initial(self, batch):
        return self.obs_mean.new_zeros((batch, self.memory_size))

    def forward(self, observation, hidden=None, starts=None, condition='clean'):
        x = ((observation - self.obs_mean) / self.obs_std).clamp(-5, 5)
        if hidden is None: hidden = self.initial(len(x))
        if starts is not None: hidden = hidden * (~starts.bool()).float()[:, None]
        if self.brain_info:
            stimulus = self.encoder(x[:, :18])
            if condition == 'disconnected': stimulus = stimulus * 0
            drive = torch.zeros_like(hidden).index_copy(1, self.inputs, stimulus)
            gain = 3 * torch.sigmoid(self.gain_raw)
            retention = .05 + .9 * torch.sigmoid(self.leak_raw)
            for _ in range(5):
                incoming = FixedSparse.apply(hidden.T.contiguous(), self.weight, self.transpose).T
                hidden = retention * hidden + (1 - retention) * torch.tanh(gain * incoming + self.bias + drive)
            signals = hidden[:, self.descending]
            signals = nn.functional.layer_norm(signals, (signals.shape[-1],))
            if condition == 'silent': signals = signals * 0
            y = self.readout(torch.cat([signals, x[:, 18:]], dim=-1))
        else:
            y = self.encoder(x)
            if self.kind == 'gru': hidden = self.core(y, hidden); y = hidden
            else: y = self.core(y)
        return self.actor(y), self.critic(x).squeeze(-1), hidden

    def sequence(self, observations, hidden, starts):
        means, values = [], []
        for observation, start in zip(observations, starts):
            mean, value, hidden = self(observation, hidden, start)
            means.append(mean); values.append(value)
        return torch.stack(means), torch.stack(values), hidden

    def distribution(self, mean):
        return torch.distributions.Normal(mean, self.log_std.clamp(-4, 0).exp())

    def log_prob(self, mean, latent):
        correction = 2 * (math.log(2) - latent - nn.functional.softplus(-2 * latent))
        return (self.distribution(mean).log_prob(latent) - correction).sum(-1)

    def edge_digest(self):
        h = hashlib.sha256()
        for tensor in (self.weight.crow_indices(), self.weight.col_indices(), self.weight.values()):
            h.update(memoryview(tensor.cpu().numpy()).cast('B'))
        return h.hexdigest()


class LearningWorlds(WorldBatch):
    """Fresh training map range, preserving the original observation/action API."""
    def __init__(self, batch, seed, scene_seeds=None, condition='clean'):
        self.train_seed = seed
        if scene_seeds is None:
            scene_seeds = [int(np.random.default_rng(np.random.SeedSequence([seed, i, 900])).integers(10000000, 11000000))
                           for i in range(batch)]
        super().__init__(batch, seed, 'clutter', scene_seeds, condition)

    def step(self, actions, autoreset=True):
        raw, rewards, done, episodes = super().step(actions, autoreset=False)
        if autoreset:
            for i in np.flatnonzero(done):
                self.worlds[i].reset(int(self.rngs[i].integers(10000000, 11000000)), 'clutter')
                self.previous[i] = None; self.delayed[i] = None; self.jitter[i] = 0; self.steps[i] = 0
                # Only observe reset slots, avoiding a second observation of live slots.
                from .population import observation_vector
                observation = self.worlds[i].observe()
                raw[i] = observation_vector(observation)
                self.previous[i] = observation['rays']
        return raw, rewards, done, episodes
