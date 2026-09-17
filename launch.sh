#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Prefer the activated Conda environment; retain existing local installations.
PROJECT_PYTHON="${CONDA_PREFIX:-$PROJECT_DIR/.conda}/bin/python"
if [[ ! -x "$PROJECT_PYTHON" ]]; then
  echo "Activate the project environment first: conda activate fruit-fly-lab" >&2
  exit 1
fi
cd "$PROJECT_DIR"
exec "$PROJECT_PYTHON" -m flylab.server "$@"
