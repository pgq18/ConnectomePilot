"""Audit the fixed-budget continuation and paired, newly held-out test results."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/plasticity/long'
MODES=['readout','ppo_edges','three_factor']
SEEDS=[71,72,73]
STEPS=[65536,131072,262144]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):return {'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}


def verify_evaluation(result,start,count):
    rows=result['episodes']
    assert sorted(r['scene_seed'] for r in rows)==list(range(start,start+count))
    assert result['summary']['episodes']==count
    assert result['summary']['successes']==sum(r['status']=='success' for r in rows)


def main():
    torch.set_num_threads(4)
    suite=read(OUT/'main/suite-complete.json')
    assert suite['total_steps']==262144 and suite['final_evaluations_complete']
    assert suite['seeds']==SEEDS and suite['modes']==MODES
    assert all(sha(ROOT/name)==digest for name,digest in read(OUT/'main/source-manifest.json').items())
    source=ROOT/'scripts/train_plasticity.py';extended=ROOT/'scripts/extend_plasticity.py'
    def core(text):return text.split('        records = []\n',1)[1].split('        if not all(torch.isfinite',1)[0]
    assert core(source.read_text())==core(extended.read_text())
    result={'steps':STEPS,'seeds':SEEDS,'fresh_test_maps':[5500000,5500199],
            'validation_maps':[5400000,5400099],'groups':{},'audit':{'learning_update_unchanged':True,'source_hashes_match':True}}
    for mode in MODES:
        group={'runs':[],'validation':{},'fresh_test':{},'legacy':{}}
        validation={step:[] for step in STEPS};fresh={'before':[],'after':[]}
        legacy={'before':{c:[] for c in ['clean','noise','delay']},'after':{c:[] for c in ['clean','noise','delay']}}
        for seed in SEEDS:
            path=OUT/f'main/{mode}-{seed}';old=ROOT/f'results/plasticity/main/{mode}-{seed}'
            config=read(path/'config.json');complete=read(path/'complete.json')
            assert config['source_checkpoint_sha256']==sha(old/'final.pt')
            assert config['initial_state_sha256']==read(old/'measurements.json')['final_state_sha256']
            assert complete['base_steps']==65536 and complete['total_steps']==262144 and complete['sign_violations']==0
            before=torch.load(old/'final.pt',map_location='cpu',weights_only=True)
            after=torch.load(path/'final.pt',map_location='cpu',weights_only=True)
            assert after['total_steps']==262144 and 'rollout_state' in after
            for name in ['source','target','sign','inputs','channels','descending','dn_mean','dn_std','initial_weight']:
                assert torch.equal(before['model'][name],after['model'][name]),name
            if mode=='readout':assert torch.equal(before['model']['edge_weight'],after['model']['edge_weight'])
            if mode!='three_factor':
                old_counts={int(v['step']) for v in before['optimizer']['state'].values()}
                new_counts={int(v['step']) for v in after['optimizer']['state'].values()}
                assert old_counts=={4096} and new_counts=={16384},(old_counts,new_counts)
                complete['optimizer_update_counts']=[4096,16384]
                old_updates=read(old/'training.json')['updates']
                new_updates=read(path/'training.json')['updates']
                complete['mean_last_quarter_kl_before']=float(np.mean([u['approx_kl'] for u in old_updates[-len(old_updates)//4:]]))
                complete['mean_last_quarter_kl_after']=float(np.mean([u['approx_kl'] for u in new_updates[-len(new_updates)//4:]]))
            complete['edge_change_since_65536_mean']=float((after['model']['edge_weight']-before['model']['edge_weight']).abs().mean())
            complete['total_absolute_edge_strength_before']=float(before['model']['edge_weight'].abs().sum())
            complete['total_absolute_edge_strength_after']=float(after['model']['edge_weight'].abs().sum())
            for step in STEPS:
                score=read(path/f'validation-{step}.json');verify_evaluation(score,5400000,100)
                validation[step].append(score['summary'])
            for stage in ['before','after']:
                score=read(path/f'test-{stage}.json');verify_evaluation(score,5500000,200)
                fresh[stage].append(score['summary'])
            for stage,file in [('before',old/'evaluation.json'),('after',path/'legacy-evaluation.json')]:
                for score in read(file):
                    verify_evaluation(score,5300000,100)
                    legacy[stage][score['condition']].append(score['summary'])
            group['runs'].append(complete)
        metrics=['success_rate','successes','collisions','timeouts','mean_task_return','mean_angular_action_change']
        for step,rows in validation.items():group['validation'][step]={m:stats([r[m] for r in rows]) for m in metrics}
        for stage,rows in fresh.items():group['fresh_test'][stage]={m:stats([r[m] for r in rows]) for m in metrics}
        group['fresh_test']['paired_success_gain']=stats([b['success_rate']-a['success_rate'] for a,b in zip(fresh['before'],fresh['after'])])
        for stage,conditions in legacy.items():
            group['legacy'][stage]={c:{m:stats([r[m] for r in rows]) for m in metrics} for c,rows in conditions.items()}
        result['groups'][mode]=group
    result['new_training_steps']=9*(262144-65536)
    result['total_training_seconds']=sum(r['train_seconds'] for g in result['groups'].values() for r in g['runs'])
    result['pipeline_wall_seconds']=(OUT/'main/suite-complete.json').stat().st_mtime-(OUT/'main/source-manifest.json').stat().st_mtime
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    def fmt(x):return f"{100*x['mean']:.1f} ± {100*x['sample_sd']:.1f}%"
    labels={'readout':'PPO 只训练读出','ppo_edges':'PPO 脑内连接＋读出','three_factor':'奖励调制局部学习'}
    lines=['|方法|65,536 步|262,144 步|平均变化（百分点）|','|---|---:|---:|---:|']
    for mode in MODES:
        g=result['groups'][mode]['fresh_test']
        lines.append(f"|{labels[mode]}|{fmt(g['before']['success_rate'])}|{fmt(g['after']['success_rate'])}|{g['paired_success_gain']['mean']*100:+.1f}|")
    (OUT/'result-table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print('Audit passed: model/optimizer continuation, fixed graph/calibration, matched evaluation maps and source hashes.')


if __name__=='__main__':main()
