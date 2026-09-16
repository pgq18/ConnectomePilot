# Contributing

## Environment

Use a dedicated project Conda prefix. Run Python through its absolute path:

```bash
PROJECT_DIR="$(pwd)"
conda env create --prefix "$PROJECT_DIR/.conda" --file environment.yml
"$PROJECT_DIR/.conda/bin/python" -m pip install -r requirements/train.txt
"$PROJECT_DIR/.conda/bin/python" -m unittest discover -s tests -v
```

Do not install project dependencies into system Python. See [GPU setup](docs/WORKSTATION.md) for CUDA experiments.

## Changes and experiments

- Keep measured connectivity, inferred transmitter signs, chosen neural dynamics, sensor encoders and action readouts distinguishable.
- Label artificial test fixtures and shuffled/random baselines explicitly.
- Compare controllers on identical held-out maps; choose checkpoints using validation maps before examining test results.
- Preserve immutable source/input snapshots and exact evaluation records when extending an experiment.
- Keep environments, credentials, raw data, model checkpoints and generated results out of Git.
- Add hardware adapters only as a separately reviewed change; current environments simulate robots.
- Run checks relevant to the change. Small artificial graphs test implementation, not biological performance.

Use Conventional Commits, for example `feat(control): add a simulated robot adapter`.
Describe the behavior change and relevant validation in pull requests. Preserve third-party notices.
