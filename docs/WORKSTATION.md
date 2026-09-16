# NVIDIA 工作站运行

## 独立环境

在工作站克隆仓库，从根目录执行：

```bash
PROJECT_DIR="$(pwd)"
conda env create --prefix "$PROJECT_DIR/.conda" --file environment.yml
"$PROJECT_DIR/.conda/bin/python" -m pip install \
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"$PROJECT_DIR/.conda/bin/python" -m pip install -r requirements/train.txt
"$PROJECT_DIR/.conda/bin/python" -c \
  'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

该 PyTorch/CUDA 组合来自本项目 RTX 5090 的既有环境。主机需有兼容 NVIDIA 驱动；其他硬件按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择构建。完整历史依赖见 `requirements/snapshots/linux-cu128-rl.txt`；不要把 Mac Conda 目录复制到 Linux。

在启动任务前检查空闲显存和现有工作。用 `CUDA_VISIBLE_DEVICES` 指定允许使用的物理卡，训练器里传对应逻辑设备编号：

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=1 "$PROJECT_DIR/.conda/bin/python" -m unittest discover -s tests -v
```

既有 4M 阶段每进程 PyTorch 张量峰值约 405.7 MiB，三进程启动时整卡占用约 4 GB；这两个指标口径不同。整脑实验的内存需求另见各轮报告。

## 远程网页

在本机建立 SSH 隧道。将 `your-workstation` 和 `/absolute/path/ConnectomePilot` 替换成自己的主机别名及远程项目路径：

```bash
ssh -L 127.0.0.1:8766:127.0.0.1:8766 -o ExitOnForwardFailure=yes your-workstation \
  'cd /absolute/path/ConnectomePilot && ./launch.sh --port 8766'
```

然后打开 <http://127.0.0.1:8766>。服务绑定回环地址，终端保持连接即可。训练不依赖网页是否打开。

## 文件管理

代码通过 Git 同步；数据从官方源准备，模型和结果另行传输。跨机器继续旧模型训练时，一起复制它需要的图、来源检查点与状态文件，保留目录关系和哈希。数据、环境、代理配置和训练产物不纳入 Git。

完整训练顺序见[训练指南](TRAINING.md)。
