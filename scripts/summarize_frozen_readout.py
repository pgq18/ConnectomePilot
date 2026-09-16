"""Audit staged training and compare checkpoints on identically held-out maps."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/plasticity/staged'
MAIN = OUT/'main'
SEEDS = [71,72,73]
GROUPS = ['joint_65k','joint_262k','readout_262k','staged_262k']
LABELS = {'joint_65k':'联合训练 65k（起点）','joint_262k':'持续联合训练 262k',
          'readout_262k':'全程只训练读出 262k','staged_262k':'先联合再冻结 262k'}


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values): return {'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}


def verify_evaluation(result,start,count):
    rows = result['episodes']; s = result['summary']
    assert sorted(r['scene_seed'] for r in rows) == list(range(start,start+count))
    counts = Counter(r['status'] for r in rows)
    assert s['episodes'] == count == sum(counts.values())
    for metric,kind in [('successes','success'),('collisions','collision'),('timeouts','timeout')]:
        assert s[metric] == counts[kind]
    assert s['successes']+s['collisions']+s['timeouts'] == count
    assert abs(s['success_rate']-s['successes']/count) < 1e-12
    return s


def main():
    torch.set_num_threads(4)
    suite = read(MAIN/'suite-complete.json')
    assert suite['final_evaluations_complete'] and suite['total_steps']==262144 and suite['seeds']==SEEDS
    for manifest in ['source-manifest.json','input-manifest.json']:
        for name,digest in read(MAIN/manifest).items(): assert sha(ROOT/name)==digest,name
    def core(text): return text.split('        records = []\n',1)[1].split('        if not all(torch.isfinite',1)[0]
    assert core((ROOT/'scripts/train_frozen_readout.py').read_text()) == core((ROOT/'scripts/extend_plasticity.py').read_text())
    metrics = ['success_rate','successes','collisions','timeouts','mean_task_return','mean_angular_action_change']
    result = {'seeds':SEEDS,'test_maps':[5600000,5600199],'steps':[65536,131072,262144],
              'new_training_steps':3*(262144-65536),'groups':{},'runs':[],'validation':{},'previous_test':{},
              'initial_validation_remeasurement':[]}
    for label in GROUPS:
        rows = []
        for seed in SEEDS:
            score = read(MAIN/f'evaluation/{label}-{seed}.json')
            assert score['seed']==seed and score['label']==label
            if label=='staged_262k': path=MAIN/f'staged-{seed}/final.pt'
            elif label=='joint_65k': path=ROOT/f'results/plasticity/main/ppo_edges-{seed}/final.pt'
            else: path=ROOT/f'results/plasticity/long/main/{"ppo_edges" if label=="joint_262k" else "readout"}-{seed}/final.pt'
            assert score['checkpoint_sha256']==sha(path)
            rows.append(verify_evaluation(score,5600000,200))
        result['groups'][label] = {metric:stats([r[metric] for r in rows]) for metric in metrics}
    for seed in SEEDS:
        folder=MAIN/f'staged-{seed}'
        config=read(folder/'config.json');complete=read(folder/'complete.json')
        oldpath=ROOT/f'results/plasticity/main/ppo_edges-{seed}/final.pt'
        before=torch.load(oldpath,map_location='cpu',weights_only=True)
        after=torch.load(folder/'final.pt',map_location='cpu',weights_only=True)
        assert config['source_checkpoint_sha256']==sha(oldpath)
        assert config['initial_state_sha256']==read(oldpath.parent/'measurements.json')['final_state_sha256']
        assert config['source_mode']=='ppo_edges' and config['mode']=='readout'
        assert config['trainable_parameters']==3960 and config['initial_optimizer_steps']==[4096]
        assert complete['total_steps']==262144 and complete['base_steps']==65536
        assert complete['frozen_edges_equal_source'] and complete['edge_change_since_freeze_max']==0
        assert complete['sign_violations']==0 and complete['final_optimizer_steps']==[16384]
        assert after['total_steps']==262144 and 'rollout_state' in after
        assert before['model']['edge_weight'].numpy().tobytes()==after['model']['edge_weight'].numpy().tobytes()
        assert len(after['optimizer']['state'])==4
        assert {int(v['step']) for v in after['optimizer']['state'].values()}=={16384}
        for name,tensor in before['model'].items():
            if name.startswith(('actor.','critic.')): continue
            assert torch.equal(tensor,after['model'][name]),name
        for name in ['actor.weight','critic.weight']:
            assert not torch.equal(before['model'][name],after['model'][name]),name
        baseline=read(ROOT/f'results/plasticity/long/main/ppo_edges-{seed}/validation-65536.json')
        initial=read(folder/'validation-65536.json')
        verify_evaluation(initial,5400000,100);verify_evaluation(baseline,5400000,100)
        previous_rows={r['scene_seed']:r for r in baseline['episodes']}
        # CUDA index_add may change floating-point trajectories between executions.
        # Audit categorical outcomes exactly and retain numerical differences.
        status_changes=sum(r['status']!=previous_rows[r['scene_seed']]['status'] for r in initial['episodes'])
        assert status_changes==0
        result['initial_validation_remeasurement'].append({'seed':seed,'status_changes':status_changes,
            'mean_task_return_difference':initial['summary']['mean_task_return']-baseline['summary']['mean_task_return'],
            'max_episode_task_return_difference':max(abs(r['task_return']-previous_rows[r['scene_seed']]['task_return']) for r in initial['episodes'])})
        result['runs'].append(complete)
    staged=result['groups']['staged_262k']['success_rate']['values']
    result['paired_gains']={label:stats([a-b for a,b in zip(staged,g['success_rate']['values'])])
                            for label,g in result['groups'].items() if label!='staged_262k'}
    for label,subdir in [('joint_262k','ppo_edges'),('readout_262k','readout'),('staged_262k','staged')]:
        result['validation'][label]={}
        for step in result['steps']:
            rows=[]
            for seed in SEEDS:
                folder=MAIN/f'staged-{seed}' if label=='staged_262k' else ROOT/f'results/plasticity/long/main/{subdir}-{seed}'
                rows.append(verify_evaluation(read(folder/f'validation-{step}.json'),5400000,100))
            result['validation'][label][step]=stats([r['success_rate'] for r in rows])
        rows=[]
        for seed in SEEDS:
            path=MAIN/f'staged-{seed}/previous-test.json' if label=='staged_262k' else ROOT/f'results/plasticity/long/main/{subdir}-{seed}/test-after.json'
            rows.append(verify_evaluation(read(path),5500000,200))
        result['previous_test'][label]={m:stats([r[m] for r in rows]) for m in metrics}
    result['training_seconds']=sum(r['train_seconds'] for r in result['runs'])
    result['pipeline_seconds']=(MAIN/'suite-complete.json').stat().st_mtime-(MAIN/'source-manifest.json').stat().st_mtime
    result['gpu_peak_mib']=max(r['gpu_peak_mib'] for r in result['runs'])
    result['audit']={'frozen_edges_bitwise_unchanged':True,'initial_model_preserved':True,
                     'readout_adam_steps':[4096,16384],'source_and_input_hashes_match':True,
                     'ppo_update_code_unchanged':True,'matched_test_maps':True}
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    lines=['|方法|种子 71|种子 72|种子 73|均值 ± 样本标准差|','|---|---:|---:|---:|---:|']
    for label in GROUPS:
        s=result['groups'][label]['success_rate']
        cells='|'.join(f'{x*100:.1f}%' for x in s['values'])
        lines.append(f"|{LABELS[label]}|{cells}|{s['mean']*100:.1f} ± {s['sample_sd']*100:.1f}%|")
    (OUT/'result-table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print('Audit passed: frozen learned graph, preserved initial model, continued readout optimizer and matched evaluation.')


if __name__=='__main__':main()
