"""Serializable rollout state for continuing the plasticity experiment."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from .world import Action


def environment_state(env):
    worlds = []
    for world in env.worlds:
        state = deepcopy(vars(world))
        state['last_action'] = [world.last_action.linear, world.last_action.angular]
        worlds.append(state)
    return {'worlds': worlds, 'rngs': [deepcopy(r.bit_generator.state) for r in env.rngs],
            'previous': deepcopy(env.previous), 'delayed': deepcopy(env.delayed),
            'jitter': env.jitter.tolist(), 'steps': env.steps.tolist(),
            'scenario': env.scenario, 'condition': env.condition}


def restore_environment(env, state):
    if len(env.worlds) != len(state['worlds']): raise ValueError('Environment batch mismatch')
    for world, saved, rng, rng_state in zip(env.worlds, state['worlds'], env.rngs, state['rngs']):
        saved = deepcopy(saved); saved['last_action'] = Action(*saved['last_action'])
        vars(world).clear(); vars(world).update(saved)
        rng.bit_generator.state = deepcopy(rng_state)
    env.previous = deepcopy(state['previous']); env.delayed = deepcopy(state['delayed'])
    env.jitter = np.asarray(state['jitter'], dtype=np.float64)
    env.steps = np.asarray(state['steps'], dtype=np.int64)
    env.scenario, env.condition = state['scenario'], state['condition']


def load_learning_state(checkpoint, model, optimizer=None, local=None, restore_traces=False):
    model.load_state_dict(checkpoint['model'], strict=True)
    if optimizer is not None:
        if checkpoint.get('optimizer') is None: raise ValueError('Missing optimizer state')
        optimizer.load_state_dict(checkpoint['optimizer'])
    if local is not None and restore_traces:
        for name, target in [('edge', local.edge_trace), ('actor', local.actor_trace), ('bias', local.bias_trace)]:
            target.copy_(checkpoint['local_traces'][name])


def save_checkpoint(path, model, optimizer, local, config, total_steps, env, raw, hidden, starts):
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict() if optimizer else None,
        'local_traces': {'edge': local.edge_trace, 'actor': local.actor_trace, 'bias': local.bias_trace} if local else None,
        'config': config, 'total_steps': total_steps,
        'rollout_state': {'environment': environment_state(env), 'raw': torch.as_tensor(raw).cpu(),
            'hidden': hidden.detach(), 'starts': starts,
            'torch_cpu_rng': torch.get_rng_state(),
            'torch_device_rng': torch.cuda.get_rng_state(hidden.device) if hidden.is_cuda else None}}
    path = Path(path); temporary = path.with_suffix('.pending')
    torch.save(payload, temporary); temporary.replace(path)
