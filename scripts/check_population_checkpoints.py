"""Reload every final checkpoint and verify finite inference and learned weights."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from flylab.population_ppo import Policy, RunningNorm
from train_population import write_json


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rows = []
    for kind in ('mlp', 'gru', 'fly', 'shuffled'):
        for seed in (11, 22, 33):
            folder = ROOT / 'results/population/main' / f'{kind}-{seed}'
            final = torch.load(folder / 'policy.pt', map_location='cpu', weights_only=False)
            initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=False)
            config = final['config']
            assert config['status'] == 'complete'
            policy = Policy(kind, config['observation_dim']).eval()
            policy.load_state_dict(final['policy'], strict=True)
            norm = RunningNorm(config['observation_dim']).eval()
            norm.load_state_dict(final['normalizer'], strict=True)
            difference = sum(float((tensor - initial['policy'][name]).square().sum())
                             for name, tensor in final['policy'].items()) ** .5
            assert difference > 0
            with torch.no_grad():
                features = torch.randn(4, config['observation_dim'])
                mean, value, _ = policy(norm(features), starts=torch.ones(4, dtype=torch.bool))
                assert torch.isfinite(mean).all() and torch.isfinite(value).all()
                assert mean.shape == (4, 2) and value.shape == (4,)
            rows.append({'run': f'{kind}-{seed}', 'strict_reload': True,
                         'finite_probe_inference': True, 'policy_weight_change_l2': difference,
                         'brain_weights_unchanged': config.get('brain_weights_unchanged'),
                         'normalizer_count': float(norm.count)})
    write_json(ROOT / 'results/population/checkpoint-validation.json', rows)
    print(f'All {len(rows)} final checkpoints reload, produce finite probe outputs, and differ from initialization.')


if __name__ == '__main__':
    main()
