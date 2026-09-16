"""Frozen MaleCNS population reservoir; engineered directional input, not a retina.

Sparse matrices are [post, pre], dense state is [neuron, environment].
No gradient graph or dense N-by-N tensor is constructed.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

DATA = Path(__file__).resolve().parents[1]/'data/male-cns'


def observation_vector(raw, previous=None):
    rays = np.asarray(raw['rays'], dtype=np.float32)
    approach = np.zeros(9, np.float32) if previous is None else np.clip(
        (np.asarray(previous, np.float32)-rays)/.3, -1, 1)
    context = [math.sin(raw['goal_angle']), math.cos(raw['goal_angle']),
               min(1, raw['goal_distance']/15), raw['linear']/1.1, raw['angular']/2.2]
    return np.asarray([*(rays/3), *approach, *context], np.float32)


def input_map(types, sides, ids):
    """Disjoint visual-cell groups for 9 distance and 9 closing-speed channels.

    Anatomical side constrains the hemisphere. Body-ID ranks only distribute
    cells reproducibly; they do NOT imply physiological receptive fields.
    Each selected cell receives one channel, never a sum of all directions.
    """
    chosen = np.isin(types, ['LC4','LPLC2','LC11','LC9','LC15','LC16','LC17','LC21'])
    neurons, channels = [], []
    for side, bearings in [('R', range(0,5)), ('L', range(4,9))]:
        candidates = np.flatnonzero(chosen & (sides == side))
        candidates = candidates[np.argsort(ids[candidates])]
        labels = [j for j in bearings] + [9+j for j in bearings]
        if len(candidates) < len(labels):
            raise ValueError(f'Insufficient annotated visual inputs on {side}')
        for group, channel in zip(np.array_split(candidates, len(labels)), labels):
            neurons.extend(group.tolist()); channels.extend([channel]*len(group))
    return np.asarray(neurons, np.int64), np.asarray(channels, np.int64)


class PopulationBrain:
    dt = .02
    neural_steps = 5

    def __init__(self, batch, device='cpu', wiring='real', seed=42, data=DATA,
                 noise_hz=1.2, gain=3., input_gain=.9):
        self.device = torch.device(device)
        self.batch, self.wiring = batch, wiring
        self.noise_hz, self.gain, self.input_gain = noise_hz, gain, input_gain
        matrix = sparse.load_npz(Path(data)/'weights.npz').tocsr().astype(np.float32)
        meta = np.load(Path(data)/'brain.npz', allow_pickle=False)
        self.n = matrix.shape[0]
        self.dn_numpy = np.flatnonzero(meta['superclass']=='descending_neuron')
        inputs, channels = input_map(meta['cell_type'], meta['side'], meta['ids'])
        self.input_numpy, self.channel_numpy = inputs, channels
        if not len(self.dn_numpy):
            raise ValueError('No descending-neuron annotations')
        if wiring == 'shuffled':
            # Reassign destinations by one fixed permutation. Preserves all edge
            # values, outgoing degrees and incoming-degree distribution, but not
            # each named neuron's incoming degree. Not a degree-preserving swap.
            matrix = matrix[np.random.default_rng(20260915).permutation(self.n)].tocsr()
        elif wiring != 'real':
            raise ValueError(wiring)
        digest = hashlib.sha256()
        for a in (matrix.indptr, matrix.indices, matrix.data):
            digest.update(memoryview(a).cast('B'))
        self.digest, self.edges = digest.hexdigest(), matrix.nnz
        self.weights = torch.sparse_csr_tensor(
            torch.as_tensor(matrix.indptr, dtype=torch.int32, device=self.device),
            torch.as_tensor(matrix.indices, dtype=torch.int32, device=self.device),
            torch.as_tensor(matrix.data, device=self.device),
            size=matrix.shape, device=self.device)
        self.dn = torch.as_tensor(self.dn_numpy, device=self.device)
        self.inputs = torch.as_tensor(inputs, device=self.device)
        self.channels = torch.as_tensor(channels, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.voltage = torch.zeros((self.n,batch), device=self.device)
        self.spikes = torch.zeros_like(self.voltage)
        self.fast = torch.zeros((len(self.dn),batch), device=self.device)
        self.slow = torch.zeros_like(self.fast)

    def reset(self, mask=None):
        if mask is None:
            for a in (self.voltage,self.spikes,self.fast,self.slow): a.zero_()
        else:
            for a in (self.voltage,self.spikes,self.fast,self.slow): a[:,mask] = 0

    @torch.no_grad()
    def advance(self, observations, disconnected=False):
        distance = (1-observations[:,:9]).clamp(0,1)
        approach = observations[:,9:18].clamp(-1,1)
        drive = torch.cat((distance,approach),1)[:,self.channels].T*self.input_gain
        if disconnected: drive.zero_()
        for _ in range(self.neural_steps):
            current = torch.sparse.mm(self.weights,self.spikes)
            self.voltage.mul_(math.exp(-self.dt/.1)).add_(current, alpha=self.gain).add_(.14)
            if self.noise_hz:
                noise = torch.rand(self.voltage.shape, generator=self.generator, device=self.device)
                self.voltage.add_((noise < self.noise_hz*self.dt).float(), alpha=.22)
            self.voltage[self.inputs] += drive
            self.spikes.copy_(self.voltage >= 1)
            self.voltage.masked_fill_(self.spikes.bool(),0)
            outputs = self.spikes[self.dn]
            self.fast.mul_(math.exp(-self.dt/.05)).add_(outputs, alpha=1-math.exp(-self.dt/.05))
            self.slow.mul_(math.exp(-self.dt/.2)).add_(outputs, alpha=1-math.exp(-self.dt/.2))
        # Goal and robot state are a shared context input, explicitly bypassing
        # the brain. Distance and approach have no direct policy bypass.
        return torch.cat((self.fast.T,self.slow.T,observations[:,18:]),1)

    def info(self):
        return {'neurons':self.n,'edges':self.edges,'descending_neurons':len(self.dn),
                'input_neurons':len(self.inputs),'input_channels':18,
                'policy_features':2*len(self.dn)+5,'wiring':self.wiring,
                'weight_digest':self.digest,'trainable_connectome':False,
                'noise_hz':self.noise_hz,'gain':self.gain,'input_gain':self.input_gain,
                'trace_tau_seconds':[.05,.2], 'neural_dt_seconds':self.dt,
                'neural_steps_per_action':self.neural_steps,
                'context_bypass':['goal_sin','goal_cos','goal_distance','linear','angular']}

    def current_digest(self):
        digest=hashlib.sha256()
        for tensor in (self.weights.crow_indices(),self.weights.col_indices(),self.weights.values()):
            digest.update(memoryview(tensor.cpu().numpy()).cast('B'))
        return digest.hexdigest()
