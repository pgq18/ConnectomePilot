"""Run the predeclared 3-seed comparison serially on the selected GPU."""
import argparse
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_population import write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--output',type=Path,default=ROOT/'results/population/main')
    args=p.parse_args()
    output=args.output.resolve()
    if output.exists(): p.error('Use a new suite output directory; existing results are never overwritten.')
    output.mkdir(parents=True)
    status={'started_at':datetime.now(timezone.utc).isoformat(),'status':'running','completed':[]}
    for seed in (11,22,33):
        for kind in ('mlp','gru','fly','shuffled'):
            destination=output/f'{kind}-{seed}'
            conditions=['clean','noise','delay']
            if seed==11 and kind in ('fly','shuffled'): conditions+=['disconnected','silent']
            status['current']=f'{kind}-{seed}'; write_json(output/'status.json',status)
            command=[sys.executable,str(ROOT/'scripts/train_population.py'),
                '--kind',kind,'--seed',str(seed),'--steps','100352','--batch','32',
                '--device',args.device,'--eval-episodes','100','--eval-start','3000000',
                '--conditions',*conditions,'--output',str(destination)]
            with (output/f'{kind}-{seed}.log').open('w') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(status='failed',returncode=result.returncode)
                write_json(output/'status.json',status); raise SystemExit(result.returncode)
            status['completed'].append(f'{kind}-{seed}')
            print(f'Completed {kind}-{seed}',flush=True)
    status.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),current=None)
    write_json(output/'status.json',status)

if __name__=='__main__': main()
