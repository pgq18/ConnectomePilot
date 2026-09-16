"""Check the full comparison before aggregating reported results."""
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
KINDS=('mlp','gru','fly','shuffled');SEEDS=(61,62,63);CONDITIONS=('clean','noise','delay')


def load(path):return json.loads(path.read_text())


def stats(values):
    return {'values':values,'mean':statistics.mean(values),'sample_sd':statistics.stdev(values)}


def main():
    root=ROOT/'results/learning';suite=load(root/'main/status.json')
    assert suite['status']=='complete'
    assert set(suite['completed'])=={f'{k}-{s}' for k in KINDS for s in SEEDS}
    runs={};hashes=None;dataset=None
    for kind in KINDS:
        for seed in SEEDS:
            name=f'{kind}-{seed}';folder=root/'main'/name
            config=load(folder/'run.json');evaluations=load(folder/'evaluation.json')
            assert config['status']=='complete' and config['ppo_steps']==32768
            assert config['kind']==kind and config['seed']==seed and config['device']=='cuda:1'
            assert config['bc_updates']==256 and config['dagger_updates']==128 and config['dagger_episodes']==32
            assert config['batch']==8 and config['sequence']==16 and config['rollout']==32
            if hashes is None:hashes=config['source_hashes'];dataset=config['dataset_sha256']
            assert config['source_hashes']==hashes and config['dataset_sha256']==dataset
            assert set((e['stage'],e['condition']) for e in evaluations)=={(s,c) for s in ('imitation','ppo') for c in CONDITIONS}
            for e in evaluations:
                rows=e['episodes'];summary=e['summary']
                assert len(rows)==100 and {r['scene_seed'] for r in rows}==set(range(4300000,4300100))
                for status,key in [('success','successes'),('collision','collisions'),('timeout','timeouts')]:
                    assert sum(r['status']==status for r in rows)==summary[key]
                assert summary['successes']+summary['collisions']+summary['timeouts']==100
            if config['brain']:
                assert config['edge_weights_unchanged']
                assert all(config['parameter_changes'][n]>0 for n in ('gain_raw','bias','leak_raw','encoder.0.weight'))
            dagger=load(folder/'dagger-collection.json')
            assert len(dagger['episodes'])==32
            runs[name]={'config':config,'dagger_labeled_frames':sum(r['steps'] for r in dagger['episodes']),
                        'evaluation':{f"{e['stage']}/{e['condition']}":e['summary'] for e in evaluations}}
    for path,digest in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    assert hashlib.sha256((root/'train.npz').read_bytes()).hexdigest()==dataset
    groups={}
    for kind in KINDS:
        groups[kind]={}
        for stage in ('imitation','ppo'):
            groups[kind][stage]={}
            for condition in CONDITIONS:
                metrics=('success_rate','collisions','timeouts','mean_task_return','mean_angular_action_change')
                groups[kind][stage][condition]={m:stats([runs[f'{kind}-{seed}']['evaluation'][f'{stage}/{condition}'][m]
                                                        for seed in SEEDS]) for m in metrics}
    result={'created_at':datetime.now(timezone.utc).isoformat(),'suite':suite,'groups':groups,'runs':runs,
            'source_hashes':hashes,'dataset_sha256':dataset,'test_maps':[4300000,4300099],
            'uncertainty':'Sample SD over three training seeds; identical test maps reused.'}
    for name in ('sensory-probe','teacher-test','mechanism','old-fly-test',
                 'learned-representation-probe','structural-paths','reload-check'):
        path=root/f'{name}.json'
        if path.exists():result[name]=load(path)
    (root/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    for stage in ('imitation','ppo'):
        print(stage)
        for kind in KINDS:
            fields=[]
            for condition in CONDITIONS:
                s=groups[kind][stage][condition]['success_rate']
                fields.append(f"{condition}: {100*s['mean']:.1f} ± {100*s['sample_sd']:.1f}% {s['values']}")
            print(kind,'; '.join(fields))
    print('All 12 runs, map partitions, source hashes, outcomes, learned dynamics and frozen edges verified.')


if __name__=='__main__':main()
