"""Audit completed runs and summarize measured values without selecting models."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results/plasticity/main'
MODES = ['readout','ppo_edges','three_factor']
SEEDS = [71,72,73]


def read(path): return json.loads(path.read_text())
def stats(values):
    return {'values': values, 'mean': float(np.mean(values)), 'sample_sd': float(np.std(values,ddof=1))}


def main():
    result = {'seeds': SEEDS, 'test_maps': [5300000,5300099], 'steps_per_run':65536, 'groups': {}}
    expected = list(range(5300000,5300100))
    checks = []
    source_hashes = read(RUNS/'source-manifest.json')
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in source_hashes.items())
    for mode in MODES:
        group = {'initial': [], 'runs': [], 'conditions': {}}
        for seed in SEEDS:
            path = RUNS/f'{mode}-{seed}'
            config = read(path/'config.json'); complete = read(path/'complete.json')
            assert config['steps']==65536 and config['seed']==seed and config['mode']==mode
            initial = read(path/'initial-evaluation.json')
            assert sorted(r['scene_seed'] for r in initial['episodes'])==expected
            group['initial'].append(initial['summary']['success_rate'])
            evaluations = read(path/'evaluation.json')
            for evaluation in evaluations:
                assert sorted(r['scene_seed'] for r in evaluation['episodes'])==expected
                assert evaluation['summary']['episodes']==100
            assert complete['sign_violations']==0
            before = torch.load(path/'initial.pt',map_location='cpu',weights_only=True)['model']
            after = torch.load(path/'final.pt',map_location='cpu',weights_only=True)['model']
            fixed = ['source','target','sign','inputs','channels','descending','dn_mean','dn_std','initial_weight']
            assert all(torch.equal(before[name],after[name]) for name in fixed)
            if mode=='readout': assert torch.equal(before['edge_weight'],after['edge_weight'])
            else: assert not torch.equal(before['edge_weight'],after['edge_weight'])
            for parameter in ['actor.weight','critic.weight']:
                assert not torch.equal(before[parameter],after[parameter])
            delta = (after['edge_weight']-before['edge_weight']).abs()
            complete['relative_edge_l1_change'] = float(delta.sum()/before['edge_weight'].abs().sum())
            complete['edge_change_quantiles'] = torch.quantile(delta,torch.tensor([.5,.9,.99])).tolist()
            group['runs'].append(complete)
            checks.append({'mode':mode,'seed':seed,'fixed_graph_and_calibration':True,'test_map_count':100})
        group['initial'] = stats(group['initial'])
        for condition in ['clean','noise','delay']:
            group['conditions'][condition] = {metric: stats([run['evaluations'][condition][metric] for run in group['runs']])
                for metric in ['success_rate','collisions','timeouts','mean_task_return','mean_angular_action_change']}
        if mode!='readout':
            group['restored_initial_edges_seed71'] = read(RUNS/f'{mode}-71/restored-edges-evaluation.json')['summary']
        result['groups'][mode] = group
    for seed in SEEDS:
        digests = [read(RUNS/f'{mode}-{seed}/config.json')['initial_state_sha256'] for mode in MODES]
        assert len(set(digests))==1
    result['audit'] = {'same_initial_states_within_seed':True,'source_files_match_training_hashes':True,'runs':checks}
    (RUNS.parent/'comparison.json').write_text(json.dumps(result,indent=2))
    lines = ['|方法|初始成功率|训练后成功率|噪声|延迟|', '|---|---:|---:|---:|---:|']
    names = {'readout':'PPO 只训练读出','ppo_edges':'PPO 训练脑内连接＋读出','three_factor':'奖励调制局部学习'}
    def fmt(x): return f"{100*x['mean']:.1f} ± {100*x['sample_sd']:.1f}%"
    for mode in MODES:
        g = result['groups'][mode]
        lines.append('|'+names[mode]+'|'+fmt(g['initial'])+'|'+'|'.join(fmt(g['conditions'][c]['success_rate']) for c in ['clean','noise','delay'])+'|')
    (RUNS.parent/'result-table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print('Audit passed: nine complete runs, identical initialization per seed, fixed graph/signs/calibration, identical test maps.')


if __name__=='__main__':main()
