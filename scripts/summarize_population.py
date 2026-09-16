"""Validate all predeclared runs, then aggregate without checkpoint selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('mlp', 'gru', 'fly', 'shuffled')
SEEDS = (11, 22, 33)
CONDITIONS = ('clean', 'noise', 'delay')


def load(path):
    return json.loads(path.read_text())


def stats(values):
    return {'values': values, 'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT / 'results/population')
    args = parser.parse_args()
    suite = load(args.root / 'main/status.json')
    assert suite['status'] == 'complete', 'Main suite has not completed'
    expected = {f'{kind}-{seed}' for kind in KINDS for seed in SEEDS}
    assert set(suite['completed']) == expected
    runs = {}
    source_hashes = None
    maps = set(range(3000000, 3000100))
    for kind in KINDS:
        for seed in SEEDS:
            name = f'{kind}-{seed}'
            folder = args.root / 'main' / name
            config = load(folder / 'run.json')
            evaluations = load(folder / 'evaluation.json')
            assert config['status'] == 'complete' and config['steps_actual'] == 100352
            assert config['kind'] == kind and config['seed'] == seed
            assert config['batch'] == 32 and config['device'] == 'cuda:1'
            assert config['train_scene_range'] == [0, 999999]
            if source_hashes is None:
                source_hashes = config['source_hashes']
            assert source_hashes == config['source_hashes'], 'Training implementation changed between runs'
            if kind in ('fly', 'shuffled'):
                assert config['brain_weights_unchanged']
                assert config['brain_hash_after'] == config['brain']['weight_digest']
            by_condition = {x['condition']: x for x in evaluations}
            required = set(CONDITIONS)
            if seed == 11 and kind in ('fly', 'shuffled'):
                required |= {'disconnected', 'silent'}
            assert set(by_condition) == required and len(evaluations) == len(required)
            for evaluation in evaluations:
                episodes = evaluation['episodes']
                summary = evaluation['summary']
                assert len(episodes) == 100 and {r['scene_seed'] for r in episodes} == maps
                for status, key in [('success', 'successes'), ('collision', 'collisions'), ('timeout', 'timeouts')]:
                    assert sum(r['status'] == status for r in episodes) == summary[key]
                assert summary['successes'] + summary['collisions'] + summary['timeouts'] == 100
                assert summary['success_rate'] == summary['successes'] / 100
            runs[name] = {'config': config, 'evaluation': {k: v['summary'] for k, v in by_condition.items()}}
    for name, digest in source_hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f'Local source differs: {name}'
    metrics = ('success_rate', 'successes', 'collisions', 'timeouts', 'mean_task_return', 'mean_angular_action_change')
    groups = {}
    for kind in KINDS:
        groups[kind] = {
            'conditions': {condition: {metric: stats([runs[f'{kind}-{seed}']['evaluation'][condition][metric]
                                                    for seed in SEEDS]) for metric in metrics}
                           for condition in CONDITIONS},
            'training_seconds': stats([runs[f'{kind}-{seed}']['config']['training_seconds'] for seed in SEEDS]),
            'gpu_peak_mib': stats([runs[f'{kind}-{seed}']['config']['gpu_peak_mib'] for seed in SEEDS]),
            'trainable_parameters': runs[f'{kind}-11']['config']['trainable_parameters'],
        }
    output = {'created_at': datetime.now(timezone.utc).isoformat(), 'suite': suite,
              'seeds': list(SEEDS), 'steps_per_run': 100352, 'total_training_steps': 100352 * 12,
              'test_scene_seeds': [3000000, 3000099], 'uncertainty': 'Sample SD across three training seeds; same 100 maps reused.',
              'source_hashes': source_hashes, 'groups': groups,
              'ablations_seed_11': {kind: runs[f'{kind}-11']['evaluation'] for kind in ('fly', 'shuffled')},
              'runs': runs}
    reference_path = args.root / 'references.json'
    if reference_path.exists():
        output['engineering_references'] = [{k: v for k, v in x.items() if k != 'episodes'} for x in load(reference_path)]
    (args.root / 'comparison.json').write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print('| 策略 | 干净场景 | 距离噪声 | 100ms 距离延迟 | 训练秒/次 | 峰值显存 MiB |')
    print('|---|---:|---:|---:|---:|---:|')
    for kind, group in groups.items():
        cells = []
        for condition in CONDITIONS:
            result = group['conditions'][condition]['success_rate']
            cells.append(f"{100*result['mean']:.1f} ± {100*result['sample_sd']:.1f}%")
        print(f"| {kind} | {' | '.join(cells)} | {group['training_seconds']['mean']:.2f} | {group['gpu_peak_mib']['mean']:.1f} |")
    print('All 12 runs, source hashes, held-out map sets, outcome counts, and frozen brain weights verified.')


if __name__ == '__main__':
    main()
