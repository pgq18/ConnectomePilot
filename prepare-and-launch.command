#!/bin/bash
set -euo pipefail
FLY_LAB_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$FLY_LAB_DIR"
PROJECT_PYTHON="${CONDA_PREFIX:-$FLY_LAB_DIR/.conda}/bin/python"
if [[ -t 0 ]]; then
    trap 'FLY_LAB_EXIT=$?; if [[ $FLY_LAB_EXIT -ne 0 && $FLY_LAB_EXIT -ne 130 ]]; then printf "\n启动未完成。请查看上方错误；窗口会保留，按回车退出。\n"; read -r FLY_LAB_REPLY || true; fi' EXIT
fi
if [[ ! -x "$PROJECT_PYTHON" ]]; then
    echo '请先激活项目环境：conda activate fruit-fly-lab' >&2
    exit 1
fi
if [[ ! -f "$FLY_LAB_DIR/data/male-cns/manifest.json" ]]; then
    printf '%s\n' '正在下载并准备真实 MaleCNS 连接数据，约 1.2 GB。所有操作使用项目 Conda 环境。'
    "$PROJECT_PYTHON" -u "$FLY_LAB_DIR/scripts/prepare_data.py" "$@"
fi
exec "$PROJECT_PYTHON" -m flylab.server --open
