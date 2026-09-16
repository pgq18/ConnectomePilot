"""Freeze a learned connectome while retaining the readout's Adam history."""


def freeze_learned_edges(model, optimizer):
    """Call after loading the joint model and its original optimizer layout.

    Loading first preserves PyTorch's parameter-order mapping. Only then remove
    the edge parameter and its Adam state; actor/critic states remain untouched.
    """
    if model.mode != 'ppo_edges':
        raise ValueError('Expected a loaded joint edge/readout model')
    edge = model.edge_weight
    if sum(p is edge for group in optimizer.param_groups for p in group['params']) != 1:
        raise ValueError('The joint optimizer must contain the edge parameter once')
    expected = {'actor.weight', 'actor.bias', 'critic.weight', 'critic.bias'}
    if {name for name, p in model.named_parameters() if p is not edge and p.requires_grad} != expected:
        raise ValueError('Unexpected trainable parameters')
    edge.requires_grad_(False)
    edge.grad = None
    for group in optimizer.param_groups:
        group['params'] = [p for p in group['params'] if p is not edge]
    optimizer.state.pop(edge, None)
    model.mode = 'readout'
