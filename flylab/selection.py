"""Choose fixed-budget checkpoints by validation success count alone."""


def select_checkpoints(candidates):
    """Equal map counts; ties prefer earlier budget, then lower training seed."""
    if not candidates:
        raise ValueError('No validation candidates')
    count = candidates[0]['episodes']
    if count <= 0 or any(c['episodes'] != count for c in candidates):
        raise ValueError('Candidates must use the same positive validation map count')
    identities = [(c['seed'], c['total_steps']) for c in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate checkpoint candidate')
    if any(not 0 <= c['successes'] <= count for c in candidates):
        raise ValueError('Invalid validation success count')
    ordered = sorted(candidates, key=lambda c: (-c['successes'], c['total_steps'], c['seed']))
    per_seed = {}
    for candidate in ordered:
        per_seed.setdefault(str(candidate['seed']), dict(candidate))
    return {
        'rule': 'maximum validation successes; ties: earlier steps, then lower seed',
        'candidates': [dict(c) for c in sorted(candidates, key=lambda c: (c['seed'], c['total_steps']))],
        'per_seed': per_seed,
        'single_policy': dict(ordered[0]),
    }
