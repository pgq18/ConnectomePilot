"""Audit the full-state frozen-brain continuation and four fixed test budgets."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_frozen_readout import verify_evaluation,stats
OUT=ROOT/'results/plasticity/staged-long'
MAIN=OUT/'main'
SEEDS=[71,72,73]
STEPS=[262144,524288,786432,1048576]
METRICS=['success_rate','successes','collisions','timeouts','mean_task_return','mean_angular_action_change']


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    suite=read(MAIN/'suite-complete.json')
    assert suite['final_evaluations_complete'] and suite['total_steps']==STEPS[-1]
    assert suite['seeds']==SEEDS and suite['milestones']==STEPS
    for file in ['source-manifest.json','input-manifest.json']:
        for name,digest in read(MAIN/file).items():assert sha(ROOT/name)==digest,name
    def core(s):return s.split('        records = []\n',1)[1].split('        if not all(torch.isfinite',1)[0]
    original=core((ROOT/'scripts/train_frozen_readout.py').read_text())
    assert original==core((ROOT/'scripts/extend_staged.py').read_text())
    result={'seeds':SEEDS,'steps':STEPS,'fresh_maps':[5700000,5700199],
            'validation_maps':[5400000,5400099],'new_training_steps':3*(1048576-262144),
            'fresh':{},'validation':{},'previous_test':{},'runs':[],'initial_validation_remeasurement':[]}
    fresh={step:[] for step in STEPS};validation={step:[] for step in STEPS}
    previous={'before':[],'after':[]}
    for seed in SEEDS:
        folder=MAIN/f'staged-{seed}'
        old=ROOT/f'results/plasticity/staged/main/staged-{seed}'
        source=torch.load(old/'final.pt',map_location='cpu',weights_only=True)
        graph_source=torch.load(ROOT/f'results/plasticity/main/ppo_edges-{seed}/final.pt',map_location='cpu',weights_only=True)
        assert source['model']['edge_weight'].numpy().tobytes()==graph_source['model']['edge_weight'].numpy().tobytes()
        config=read(folder/'config.json');resume=read(folder/'resume-audit.json');complete=read(folder/'complete.json')
        assert config['source_checkpoint_sha256']==sha(old/'final.pt')==resume['source_checkpoint_sha256']
        assert config['initial_state_sha256']==read(old/'measurements.json')['final_state_sha256']==resume['model_state_sha256']
        assert config['trainable_parameters']==3960 and config['initial_optimizer_steps']==[16384]
        assert config['continuation']=='full_rollout_state' and config['frozen_since_steps']==65536
        assert all(resume[k] for k in ['full_environment_restored','observations_and_neural_state_restored','cpu_and_gpu_rng_restored'])
        assert resume['optimizer_steps']==[16384]
        assert complete['base_steps']==262144 and complete['total_steps']==1048576
        assert complete['frozen_edges_equal_source'] and complete['edge_change_since_freeze_max']==0 and complete['sign_violations']==0
        assert complete['final_optimizer_steps']==[65536]
        for step in STEPS:
            path=old/'final.pt' if step==262144 else folder/('final.pt' if step==1048576 else f'checkpoint-{step}.pt')
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            assert checkpoint['total_steps']==step and 'rollout_state' in checkpoint
            assert checkpoint['model']['edge_weight'].numpy().tobytes()==source['model']['edge_weight'].numpy().tobytes()
            for name,value in source['model'].items():
                if not name.startswith(('actor.','critic.')):assert torch.equal(value,checkpoint['model'][name]),name
            assert len(checkpoint['optimizer']['state'])==4
            assert {int(v['step']) for v in checkpoint['optimizer']['state'].values()}=={step//16}
            if step>262144:
                for name in ['actor.weight','critic.weight']:assert not torch.equal(source['model'][name],checkpoint['model'][name])
            evaluation=read(MAIN/f'evaluation/step-{step}-{seed}.json')
            assert evaluation['checkpoint_sha256']==sha(path) and evaluation['total_steps']==step and evaluation['seed']==seed
            fresh[step].append(verify_evaluation(evaluation,5700000,200))
            validation[step].append(verify_evaluation(read(folder/f'validation-{step}.json'),5400000,100))
        old_initial=read(old/'validation-262144.json');new_initial=read(folder/'validation-262144.json')
        older={row['scene_seed']:row for row in old_initial['episodes']}
        result['initial_validation_remeasurement'].append({'seed':seed,
            'status_changes':sum(row['status']!=older[row['scene_seed']]['status'] for row in new_initial['episodes']),
            'mean_return_difference':new_initial['summary']['mean_task_return']-old_initial['summary']['mean_task_return']})
        previous['before'].append(verify_evaluation(read(ROOT/f'results/plasticity/staged/main/evaluation/staged_262k-{seed}.json'),5600000,200))
        previous['after'].append(verify_evaluation(read(folder/'previous-test.json'),5600000,200))
        logs=read(folder/'training.json')['updates']
        assert logs[0]['steps']==263168 and logs[-1]['steps']==1048576 and len(logs)==768
        complete['mean_last_quarter_kl']=float(np.mean([r['approx_kl'] for r in logs[-192:]]))
        result['runs'].append(complete)
    for step in STEPS:
        result['fresh'][step]={m:stats([r[m] for r in fresh[step]]) for m in METRICS}
        result['validation'][step]={m:stats([r[m] for r in validation[step]]) for m in METRICS}
    result['paired_gains']={step:stats([b['success_rate']-a['success_rate'] for a,b in zip(fresh[262144],fresh[step])]) for step in STEPS[1:]}
    result['previous_test']={stage:{m:stats([r[m] for r in rows]) for m in METRICS} for stage,rows in previous.items()}
    result['training_seconds']=sum(r['train_seconds'] for r in result['runs'])
    result['pipeline_seconds']=(MAIN/'suite-complete.json').stat().st_mtime-(MAIN/'source-manifest.json').stat().st_mtime
    result['gpu_peak_mib']=max(r['gpu_peak_mib'] for r in result['runs'])
    result['audit']={'full_rollout_restored':True,'frozen_edges_bitwise_match_65536_and_262144':True,
        'optimizer_steps':[16384,65536],'sampling_and_learning_unchanged':True,
        'core_sha256':hashlib.sha256(original.encode()).hexdigest(),'source_and_input_hashes_match':True,'matched_test_maps':True}
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    lines=['|总步数|种子 71|种子 72|种子 73|均值 ± 样本标准差|相对 262,144 步（百分点）|',
           '|---|---:|---:|---:|---:|---:|']
    for step in STEPS:
        score=result['fresh'][step]['success_rate']
        values='|'.join(f'{x*100:.1f}%' for x in score['values'])
        gain=(score['mean']-result['fresh'][262144]['success_rate']['mean'])*100
        lines.append(f"|{step:,}|{values}|{score['mean']*100:.1f} ± {score['sample_sd']*100:.1f}%|{gain:+.1f}|")
    (OUT/'result-table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print('Audit passed: full rollout continuation, frozen graph, exact optimizer counts, unchanged learning and paired evaluation.')


if __name__=='__main__':main()
