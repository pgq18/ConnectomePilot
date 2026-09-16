"""Preserve the exact continuation inputs, sources and all reported evidence."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/plasticity/long'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    suite=json.loads((OUT/'main/suite-complete.json').read_text())
    assert suite['total_steps']==262144 and suite['final_evaluations_complete']
    sources=set(json.loads((OUT/'main/source-manifest.json').read_text()))
    for directory in ['flylab','scripts','tests']:
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob('*.py'))
    sources.update(['AGENTS.md','README.md','requirements-rl.linux.lock.txt',
        'docs/PLASTICITY_PROTOCOL.md','docs/PLASTICITY_TRIAL.md',
        'docs/PLASTICITY_LONG_PROTOCOL.md','docs/PLASTICITY_LONG_TRIAL.md',
        'data/plasticity/graph.npz','data/plasticity/manifest.json'])
    manifest={name:sha(ROOT/name) for name in sorted(sources)}
    artifacts=list((OUT/'main').rglob('*.pt'))+list((OUT/'main').rglob('*.json'))
    artifacts += [OUT/'comparison.json',OUT/'result-table.md',OUT/'comparison.png',OUT/'comparison.pdf',OUT/'failure-types.png',OUT/'tests.log',OUT/'main.log']
    artifacts += list((ROOT/'results/plasticity/main').glob('*/final.pt'))
    for path in artifacts:manifest[str(path.relative_to(ROOT))]=sha(path)
    (OUT/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(OUT/'source-snapshot.tar.gz','w:gz') as archive:
        for name in sorted(sources):archive.add(ROOT/name,arcname=name)
    print(f'Archived {len(sources)} source/input files; checksummed {len(manifest)} files including checkpoints and evaluation evidence.')


if __name__=='__main__':main()
