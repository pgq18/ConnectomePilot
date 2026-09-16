"""Archive exact trial inputs, code and checkpoint checksums after final audit."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    result=ROOT/'results/plasticity'
    assert len(list((result/'main').glob('*/complete.json')))==9
    files=set(json.loads((result/'main/source-manifest.json').read_text()))
    for folder in ['flylab','scripts']:
        files.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).glob('*.py'))
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_*.py'))
    files.update(['AGENTS.md','README.md','environment.yml','requirements/base.txt','requirements/train.txt',
        'requirements/snapshots/linux-cu128-rl.txt',
                  'docs/PLASTICITY_PROTOCOL.md','docs/PLASTICITY_TRIAL.md','data/plasticity/graph.npz',
                  'data/plasticity/manifest.json','scripts/summarize_plasticity.py','scripts/plot_plasticity.py',
                  'scripts/diagnose_plasticity.py','scripts/archive_plasticity.py',
                  'scripts/train_population.py','flylab/__init__.py','flylab/brain.py'])
    manifest={name:sha(ROOT/name) for name in sorted(files)}
    manifest.update({str(p.relative_to(ROOT)):sha(p) for p in sorted(result.glob('main/**/*.pt'))})
    manifest['data/male-cns/weights.npz']=sha(ROOT/'data/male-cns/weights.npz')
    manifest['data/male-cns/brain.npz']=sha(ROOT/'data/male-cns/brain.npz')
    (result/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(result/'source-snapshot.tar.gz','w:gz') as archive:
        for name in sorted(files): archive.add(ROOT/name,arcname=name)
    reference=result/'reference'
    (reference/'provenance.json').write_text(json.dumps({
        'upstream':'https://github.com/eganeganegan/flydoom',
        'revision':'b047fabffb62e10268de2fc1f12c765b8b019c0b',
        'purpose':'Read-only implementation reference, not installed or executed as the experiment.',
        'source_hashes':{str(p.relative_to(reference)):sha(p) for p in reference.rglob('*') if p.is_file() and p.name!='provenance.json'}
    },indent=2))
    print(f'Archived {len(files)} source/input files and checksummed {len(manifest)} artifacts.')


if __name__=='__main__':main()
