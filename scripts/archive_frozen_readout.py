"""Archive source/input snapshots and checksum staged-training evidence."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/plasticity/staged'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    assert json.loads((OUT/'main/suite-complete.json').read_text())['final_evaluations_complete']
    sources=set(json.loads((OUT/'main/source-manifest.json').read_text()))
    for directory in ['flylab','scripts','tests']:
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob('*.py'))
    sources.update(['AGENTS.md','README.md','requirements-rl.linux.lock.txt','flylab/staged.py',
        'tests/test_staged.py','tests/test_resume.py','tests/test_plasticity.py',
        'scripts/summarize_frozen_readout.py','scripts/plot_frozen_readout.py','scripts/archive_frozen_readout.py',
        'docs/FROZEN_READOUT_PROTOCOL.md','docs/FROZEN_READOUT_TRIAL.md',
        'docs/PLASTICITY_PROTOCOL.md','docs/PLASTICITY_TRIAL.md','docs/PLASTICITY_LONG_PROTOCOL.md','docs/PLASTICITY_LONG_TRIAL.md',
        'data/plasticity/graph.npz','data/plasticity/manifest.json'])
    artifacts=set((OUT/'main').rglob('*.pt'))|set((OUT/'main').rglob('*.json'))
    artifacts.update(ROOT/name for name in json.loads((OUT/'main/input-manifest.json').read_text()))
    for mode in ['readout','ppo_edges']:
        for seed in [71,72,73]:
            folder=ROOT/f'results/plasticity/long/main/{mode}-{seed}'
            artifacts.update(folder.glob('validation-*.json'));artifacts.add(folder/'test-after.json')
    for seed in [71,72,73]:artifacts.add(ROOT/f'results/plasticity/main/ppo_edges-{seed}/measurements.json')
    artifacts.update(OUT/name for name in ['comparison.json','result-table.md','comparison.png','comparison.pdf','failure-types.png','main.log','tests.log','freeze-forward-check.json'])
    manifest={name:sha(ROOT/name) for name in sorted(sources)}
    for path in sorted(artifacts):manifest[str(path.relative_to(ROOT))]=sha(path)
    (OUT/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2))
    with tarfile.open(OUT/'source-snapshot.tar.gz','w:gz') as archive:
        for name in sorted(sources):archive.add(ROOT/name,arcname=name)
    print(f'Archived {len(sources)} source/input files; verified hashes for {len(manifest)} files.')


if __name__=='__main__':main()
