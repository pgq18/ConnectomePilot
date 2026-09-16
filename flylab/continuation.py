"""Restore a saved rollout without resetting episodes or neural state."""
import torch
from .resume import restore_environment


def restore_rollout_state(checkpoint, env, device):
    state = checkpoint['rollout_state']
    restore_environment(env, state['environment'])
    raw = state['raw'].cpu().numpy().copy()
    hidden = state['hidden'].to(device).detach().clone()
    starts = state['starts'].to(device).clone()
    if len(raw) != len(env.worlds) or len(hidden) != len(raw) or len(starts) != len(raw):
        raise ValueError('Checkpoint rollout batch mismatch')
    torch.set_rng_state(state['torch_cpu_rng'].cpu())
    if torch.device(device).type == 'cuda':
        if state['torch_device_rng'] is None:
            raise ValueError('Missing GPU random state')
        torch.cuda.set_rng_state(state['torch_device_rng'].cpu(), device)
    return raw, hidden, starts
