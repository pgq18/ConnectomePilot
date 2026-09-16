"""Preserve full-state long-continuation sources, inputs and measured evidence."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/plasticity/staged-4m'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    assert json.loads((OUT/'main/suite-complete.json').read_text())['final_evaluations_complete']
    sources=set(json.loads((OUT/'main/source-manifest.json').read_text()))
    for directory in ['flylab','scripts','tests']:
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob('*.py'))
    sources.update(['AGENTS.md','README.md','environment.yml','requirements/base.txt','requirements/train.txt',
        'requirements/snapshots/linux-cu128-rl.txt',
        'data/plasticity/graph.npz','data/plasticity/manifest.json',
        'docs/PLASTICITY_PROTOCOL.md','docs/PLASTICITY_TRIAL.md',
        'docs/FROZEN_READOUT_PROTOCOL.md','docs/FROZEN_READOUT_TRIAL.md',
        'docs/STAGED_LONG_PROTOCOL.md','docs/STAGED_LONG_TRIAL.md',
        'docs/STAGED_4M_PROTOCOL.md','docs/STAGED_4M_TRIAL.md'])
    artifacts=set((OUT/'main').rglob('*.json'))|set((OUT/'main').rglob('*.pt'))|set((OUT/'main').rglob('*.log'))
    artifacts.update(ROOT/name for name in json.loads((OUT/'main/input-manifest.json').read_text()))
    for seed in [71,72,73]:
        artifacts.add(ROOT/f'results/plasticity/main/ppo_edges-{seed}/final.pt')
        old=ROOT/f'results/plasticity/staged-long/main/staged-{seed}'
        artifacts.update([old/'measurements.json',old/'validation-1048576.json'])
        artifacts.add(ROOT/f'results/plasticity/staged-long/main/evaluation/step-1048576-{seed}.json')
    artifacts.update(OUT/name for name in ['comparison.json','result-table.md','comparison.png','comparison.pdf','failure-types.png','main.log','tests.log','smoke.log'])
    manifest={name:sha(ROOT/name) for name in sorted(sources)}
    for path in sorted(artifacts):manifest[str(path.relative_to(ROOT))]=sha(path)
    (OUT/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(OUT/'source-snapshot.tar.gz','w:gz') as archive:
        for name in sorted(sources):archive.add(ROOT/name,arcname=name)
    print(f'Archived {len(sources)} source/input files; checksummed {len(manifest)} files.')


if __name__=='__main__':main()
