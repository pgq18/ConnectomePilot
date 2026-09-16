"""Audit extended frozen-brain training, validation selection and fresh evaluation."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.summarize_frozen_readout import verify_evaluation,stats
from flylab.selection import select_checkpoints
OUT=ROOT/'results/plasticity/staged-4m'
MAIN=OUT/'main'
SEEDS=[71,72,73]
STEPS=[1048576,1572864,2097152,2621440,3145728,3670016,4194304]
METRICS=['success_rate','successes','collisions','timeouts','mean_task_return','mean_angular_action_change']


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes,count):
    z=1.959963984540054;p=successes/count
    denominator=1+z*z/count
    center=(p+z*z/(2*count))/denominator
    half=z*((p*(1-p)/count+z*z/(4*count*count))**.5)/denominator
    return [center-half,center+half]


def main():
    torch.set_num_threads(4)
    suite=read(MAIN/'suite-complete.json')
    assert suite['final_evaluations_complete'] and suite['total_steps']==STEPS[-1]
    assert suite['seeds']==SEEDS and suite['milestones']==STEPS
    for file in ['source-manifest.json','input-manifest.json']:
        for name,digest in read(MAIN/file).items():assert sha(ROOT/name)==digest,name
    def core(s):return s.split('        records = []\n',1)[1].split('        if not all(torch.isfinite',1)[0]
    original=core((ROOT/'scripts/train_frozen_readout.py').read_text())
    assert original==core((ROOT/'scripts/extend_staged_4m.py').read_text())
    result={'seeds':SEEDS,'steps':STEPS,'fresh_maps':[5800000,5800499],
            'validation_maps':[5410000,5410199],'new_training_steps':3*(4194304-1048576),
            'fresh':{},'validation':{},'previous_test':{},'runs':[]}
    fresh={step:[] for step in STEPS};validation={step:[] for step in STEPS}
    previous={'before':[],'after':[]}
    for seed in SEEDS:
        folder=MAIN/f'staged-{seed}'
        old=ROOT/f'results/plasticity/staged-long/main/staged-{seed}'
        source=torch.load(old/'final.pt',map_location='cpu',weights_only=True)
        graph_source=torch.load(ROOT/f'results/plasticity/main/ppo_edges-{seed}/final.pt',map_location='cpu',weights_only=True)
        assert source['model']['edge_weight'].numpy().tobytes()==graph_source['model']['edge_weight'].numpy().tobytes()
        config=read(folder/'config.json');resume=read(folder/'resume-audit.json');complete=read(folder/'complete.json')
        assert config['source_checkpoint_sha256']==sha(old/'final.pt')==resume['source_checkpoint_sha256']
        assert config['initial_state_sha256']==read(old/'measurements.json')['final_state_sha256']==resume['model_state_sha256']
        assert config['trainable_parameters']==3960 and config['initial_optimizer_steps']==[65536]
        assert config['continuation']=='full_rollout_state' and config['frozen_since_steps']==65536
        assert all(resume[k] for k in ['full_environment_restored','observations_and_neural_state_restored','cpu_and_gpu_rng_restored'])
        assert resume['optimizer_steps']==[65536]
        assert complete['base_steps']==1048576 and complete['total_steps']==4194304
        assert complete['frozen_edges_equal_source'] and complete['edge_change_since_freeze_max']==0 and complete['sign_violations']==0
        assert complete['final_optimizer_steps']==[262144]
        for step in STEPS:
            path=old/'final.pt' if step==1048576 else folder/('final.pt' if step==4194304 else f'checkpoint-{step}.pt')
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            assert checkpoint['total_steps']==step and 'rollout_state' in checkpoint
            assert checkpoint['model']['edge_weight'].numpy().tobytes()==source['model']['edge_weight'].numpy().tobytes()
            for name,value in source['model'].items():
                if not name.startswith(('actor.','critic.')):assert torch.equal(value,checkpoint['model'][name]),name
            assert len(checkpoint['optimizer']['state'])==4
            assert {int(v['step']) for v in checkpoint['optimizer']['state'].values()}=={step//16}
            if step>1048576:
                for name in ['actor.weight','critic.weight']:assert not torch.equal(source['model'][name],checkpoint['model'][name])
            evaluation=read(MAIN/f'evaluation/step-{step}-{seed}.json')
            assert evaluation['checkpoint_sha256']==sha(path) and evaluation['total_steps']==step and evaluation['seed']==seed
            fresh[step].append(verify_evaluation(evaluation,5800000,500))
            validation[step].append(verify_evaluation(read(folder/f'validation-{step}.json'),5410000,200))
        previous['before'].append(verify_evaluation(read(folder/'previous-test-1048576.json'),5700000,200))
        previous['after'].append(verify_evaluation(read(folder/'previous-test-4194304.json'),5700000,200))
        logs=read(folder/'training.json')['updates']
        assert logs[0]['steps']==1049600 and logs[-1]['steps']==4194304 and len(logs)==3072
        complete['mean_last_quarter_kl']=float(np.mean([r['approx_kl'] for r in logs[-768:]]))
        result['runs'].append(complete)
    for step in STEPS:
        result['fresh'][step]={m:stats([r[m] for r in fresh[step]]) for m in METRICS}
        result['validation'][step]={m:stats([r[m] for r in validation[step]]) for m in METRICS}
    result['paired_gains']={step:stats([b['success_rate']-a['success_rate'] for a,b in zip(fresh[1048576],fresh[step])]) for step in STEPS[1:]}
    result['previous_test']={stage:{m:stats([r[m] for r in rows]) for m in METRICS} for stage,rows in previous.items()}
    result['sum_process_training_seconds']=sum(r['train_seconds'] for r in result['runs'])
    result['max_process_training_seconds']=max(r['train_seconds'] for r in result['runs'])
    result['pipeline_seconds']=(MAIN/'suite-complete.json').stat().st_mtime-(MAIN/'source-manifest.json').stat().st_mtime
    result['gpu_peak_mib']=max(r['gpu_peak_mib'] for r in result['runs'])
    result['audit']={'full_rollout_restored':True,'frozen_edges_bitwise_match_65536_and_1048576':True,
        'optimizer_steps':[65536,262144],'sampling_and_learning_unchanged':True,
        'core_sha256':hashlib.sha256(original.encode()).hexdigest(),'source_and_input_hashes_match':True,'matched_test_maps':True}
    selection=read(MAIN/'selection.json')
    rebuilt=select_checkpoints(selection['candidates'])
    for key in ['rule','candidates','per_seed','single_policy']:
        assert rebuilt[key]==selection[key],key
    assert len(selection['candidates'])==len(SEEDS)*len(STEPS)
    test_start=read(MAIN/'evaluation/test-start.json')
    assert test_start['selection_sha256']==sha(MAIN/'selection.json')
    assert selection['selected_at_unix']<=test_start['started_at_unix']
    assert selection['validation_start']==5410000 and selection['validation_episodes']==200
    assert selection['fresh_test_start']==5800000 and selection['fresh_test_episodes']==500
    assert selection['chosen_before_test']
    for candidate in selection['candidates']:
        seed=candidate['seed'];step=candidate['total_steps']
        path=MAIN/f'staged-{seed}/validation-{step}.json'
        assert sha(path)==candidate['validation_sha256']
        assert candidate['successes']==read(path)['summary']['successes']
        assert candidate['episodes']==200
        assert candidate['checkpoint_sha256']==read(MAIN/f'evaluation/step-{step}-{seed}.json')['checkpoint_sha256']
    chosen=[selection['per_seed'][str(seed)] for seed in SEEDS]
    chosen_scores=[verify_evaluation(read(MAIN/f"evaluation/step-{c['total_steps']}-{c['seed']}.json"),5800000,500) for c in chosen]
    result['selected']={m:stats([row[m] for row in chosen_scores]) for m in METRICS}
    result['selected_checkpoints']=chosen
    policy=selection['single_policy']
    policy_score=verify_evaluation(read(MAIN/f"evaluation/step-{policy['total_steps']}-{policy['seed']}.json"),5800000,500)
    result['single_policy']={**policy,'test':policy_score,'wilson_95':wilson(policy_score['successes'],500)}
    result['highest_test_point_descriptive']=max(
        ({'seed':seed,'total_steps':step,'success_rate':fresh[step][i]['success_rate']} for step in STEPS for i,seed in enumerate(SEEDS)),
        key=lambda row:row['success_rate'])
    result['previous_selected']={m:stats([verify_evaluation(read(MAIN/f"staged-{c['seed']}/previous-test-{c['total_steps']}.json"),5700000,200)[m] for c in chosen]) for m in METRICS}
    result['previous_recorded_baseline']=[read(ROOT/f'results/plasticity/staged-long/main/evaluation/step-1048576-{seed}.json')['summary']['success_rate'] for seed in SEEDS]
    result['audit']['validation_selection_committed_before_fresh_test']=True
    # Export byte-identical, easy-to-find copies after all audits have passed.
    for choice in chosen:
        seed=choice['seed'];step=choice['total_steps']
        original=(ROOT/f'results/plasticity/staged-long/main/staged-{seed}/final.pt' if step==1048576
                  else MAIN/f'staged-{seed}'/('final.pt' if step==4194304 else f'checkpoint-{step}.pt'))
        exported=MAIN/f'staged-{seed}/best-validation.pt'
        if exported.exists():assert sha(exported)==sha(original)
        else:shutil.copy2(original,exported)
        assert sha(exported)==choice['checkpoint_sha256']
    single=MAIN/'best-validation.pt'
    original=MAIN/f"staged-{policy['seed']}/best-validation.pt"
    if single.exists():assert sha(single)==sha(original)
    else:shutil.copy2(original,single)
    result['single_policy']['export']='results/plasticity/staged-4m/main/best-validation.pt'
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    lines=['|总步数|种子 71|种子 72|种子 73|均值 ± 样本标准差|相对 1,048,576 步（百分点）|',
           '|---|---:|---:|---:|---:|---:|']
    for step in STEPS:
        score=result['fresh'][step]['success_rate']
        values='|'.join(f'{x*100:.1f}%' for x in score['values'])
        gain=(score['mean']-result['fresh'][1048576]['success_rate']['mean'])*100
        lines.append(f"|{step:,}|{values}|{score['mean']*100:.1f} ± {score['sample_sd']*100:.1f}%|{gain:+.1f}|")
    (OUT/'result-table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print('Audit passed: full rollout continuation, frozen graph, exact optimizer counts, unchanged learning and paired evaluation.')


if __name__=='__main__':main()
