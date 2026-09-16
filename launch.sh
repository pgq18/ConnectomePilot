#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PYTHON="$PROJECT_DIR/.conda/bin/python"
if [[ ! -x "$PROJECT_PYTHON" ]]; then
  echo "Project Conda environment is missing: $PROJECT_PYTHON" >&2
  exit 1
fi
cd "$PROJECT_DIR"
exec "$PROJECT_PYTHON" -m flylab.server "$@"
