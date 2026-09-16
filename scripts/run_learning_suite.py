"""Serial execution of the predeclared trainable-dynamics comparison."""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_population import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/learning/main');args=p.parse_args()
    if args.output.exists():p.error('Use a new suite directory.')
    args.output.mkdir(parents=True)
    status={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'completed':[]}
    for seed in (61,62,63):
        for kind in ('mlp','gru','fly','shuffled'):
            name=f'{kind}-{seed}';status['current']=name;write_json(args.output/'status.json',status)
            with (args.output/f'{name}.log').open('w') as log:
                result=subprocess.run([sys.executable,str(ROOT/'scripts/train_learning.py'),'--kind',kind,
                    '--seed',str(seed),'--device','cuda:1','--output',str(args.output/name)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(status='failed',returncode=result.returncode);write_json(args.output/'status.json',status)
                raise SystemExit(result.returncode)
            status['completed'].append(name);print('Completed',name,flush=True)
    status.update(status='complete',current=None,finished_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output/'status.json',status)


if __name__=='__main__':main()
