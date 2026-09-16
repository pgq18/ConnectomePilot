#!/bin/bash
set -euo pipefail
FLY_LAB_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$FLY_LAB_DIR"
exec "$FLY_LAB_DIR/.conda/bin/python" -m flylab.server --open
