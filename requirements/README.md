# Dependencies

All profiles use the same project Conda environment. Run commands from the repository root after `conda activate fruit-fly-lab`. Conda manages the environment location.

| File | Purpose | When to install |
|---|---|---|
| [`../environment.yml`](../environment.yml) | Creates the Python 3.10 Conda environment and installs `base.txt` | First setup |
| [`base.txt`](base.txt) | NumPy, SciPy, and PyArrow for simulation and connectome preparation | Included in Conda setup |
| [`train.txt`](train.txt) | Base dependencies plus Gymnasium, Stable-Baselines3, and Numba | Training and the full test suite |
| [`viz.txt`](viz.txt) | Base dependencies plus Matplotlib and Pillow | Plots and video frame rendering |

```bash
conda env create -f environment.yml
conda activate fruit-fly-lab

# Add the capabilities you need to the same environment.
python -m pip install -r requirements/train.txt
python -m pip install -r requirements/viz.txt
```

The optional profiles include `base.txt`; shared versions are defined once. For NVIDIA training, install the appropriate PyTorch build **before** `train.txt`, as shown in the [workstation guide](../docs/WORKSTATION.md). FFmpeg/FFprobe are separate video-encoding tools; see the [video guide](../docs/VIDEOS.md).

## Historical environment snapshots

[`snapshots/`](snapshots/) preserves the original recorded package versions without changing their contents:

- [`macos-rl.txt`](snapshots/macos-rl.txt): local macOS PPO environment.
- [`linux-cu128-rl.txt`](snapshots/linux-cu128-rl.txt): Linux workstation environment with PyTorch's CUDA 12.8 build.
- [`visualization.txt`](snapshots/visualization.txt): recorded visualization packages.

These are reference snapshots, not portable lock files. They do not specify a complete Conda environment, platform metadata, package hashes, or every package source. Do not install all three together or copy an environment between macOS and Linux. Use the profiles above for setup; consult the matching snapshot when investigating or reproducing an older experiment. Existing experiment archives retain their original paths.

## 为什么原来有这么多文件？

原来根目录混放了四份功能安装清单（基础、数据、训练、可视化）和三份历史版本快照。现在基础与数据合并为 `base.txt`，Conda 创建环境时自动安装；需要训练或导出画面时，再在同一环境补装 `train.txt` 或 `viz.txt`。历史版本移入 `snapshots/`，只用于核对旧实验环境，不代表还要创建三个环境。
