# 训练与评估复现

所有命令从仓库根目录执行。先按[首页](../README.md)创建项目 Conda 环境；每次新开终端设置：

```bash
PROJECT_DIR="$(pwd)"
```

## 1. 轻量 CPU PPO

```bash
"$PROJECT_DIR/.conda/bin/python" -m pip install -r requirements-rl.txt
"$PROJECT_DIR/.conda/bin/python" scripts/train_ppo.py \
  --features rays --steps 8192 --rollout-steps 512 \
  --eval-seed-start 2000100 --output results/ppo/rays
```

`--features fly` 使用固定整脑的早期双输出特征，需要先准备 MaleCNS 数据。该入口不更新脑内连接。输出放在默认的 `results/ppo/rays` / `results/ppo/fly` 时，网页可发现对应检查点；详见 `flylab/ppo_controller.py`。

## 2. 准备真实连接子图

GPU 环境安装见[工作站指南](WORKSTATION.md)。完成官方数据准备后：

```bash
"$PROJECT_DIR/.conda/bin/python" scripts/prepare_data.py
"$PROJECT_DIR/.conda/bin/python" scripts/prepare_plasticity.py
```

生成 `data/plasticity/graph.npz` 与 `manifest.json`。既有数据版本生成 4,096 神经元、459,342 条允许连接、1,314 个下行神经元。脚本拒绝覆盖已有图；恢复旧模型时必须使用与检查点图哈希一致的文件。

## 3. 从零复现历史训练链

以下完整流程包含后续报告需要的基线。它会产生大量检查点并执行正式评估；不要当成几秒钟的安装检查。已有实验请使用新的输出目录并显式指定上一步来源，不覆盖原结果。

仅暴露一张已确认可用的物理卡；例如 GPU 1 在进程内显示为 `cuda:0`：

```bash
export CUDA_VISIBLE_DEVICES=1

# A. 三种学习方法 × 三个种子，每组 65,536 步。
"$PROJECT_DIR/.conda/bin/python" scripts/train_plasticity.py \
  --device cuda:0 --output results/plasticity/main

# B. 持续读出/联合/局部学习基线，每组延长至 262,144 步。
# 下一阶段会读取其中 readout 和 ppo_edges 的模型作同图比较。
"$PROJECT_DIR/.conda/bin/python" scripts/extend_plasticity.py \
  --device cuda:0 --output results/plasticity/long/main

# C. 回到 A 的 65k 联合模型，冻结连接，训练读出至 262,144 步。
"$PROJECT_DIR/.conda/bin/python" scripts/train_frozen_readout.py \
  --device cuda:0 --output results/plasticity/staged/main

# D. 保留完整环境、神经状态和优化器，继续至 1,048,576 步。
"$PROJECT_DIR/.conda/bin/python" scripts/extend_staged.py \
  --device cuda:0 --output results/plasticity/staged-long/main

# E. 三个种子并行，继续至 4,194,304 步；自动验证选模和独立测试。
"$PROJECT_DIR/.conda/bin/python" scripts/extend_staged_4m.py \
  --device cuda:0 --parallel-runs --output results/plasticity/staged-4m/main
```

最后一步每组新增 3,145,728 步。已有实验在一张 RTX 5090 上三进程并行约 77.7 分钟，包含训练和评估；这是实测记录，其他机器可能不同。

训练器为对应实验设计，来源步数、图哈希、优化器计数、预算与地图区间都有检查。它们不是任意检查点的通用续训工具。`--skip-final-evaluations` 仅跳过该入口末尾评估，不会取消输入检查或保存节点验证。

## 4. 汇总和审计

完整训练链结束后：

```bash
"$PROJECT_DIR/.conda/bin/python" -m pip install -r requirements-viz.txt
"$PROJECT_DIR/.conda/bin/python" scripts/summarize_staged_4m.py
"$PROJECT_DIR/.conda/bin/python" scripts/plot_staged_4m.py
"$PROJECT_DIR/.conda/bin/python" scripts/report_staged_4m.py
```

汇总器还会读取上轮汇总、归档与输入记录；若从零重建完整历史报告，按各轮 `summarize_*`、`plot_*`、`archive_*` 的输入要求逐轮执行。详见[实验索引](README.md)。仅训练和回放不需要重建历史汇总报告。

`report_staged_4m.py` 是既有实验报告生成器，包含历史运行条件说明。做新实验应新建报告并重新记录机器、日期、计时和实际结果，不能沿用历史条件作为新测量。

## 5. 评估边界

最新训练链的地图区间：

|用途|场景种子|
|---|---|
|训练采样|20,000,000–20,999,999|
|4M 阶段验证|5,410,000–5,410,199|
|4M 阶段独立测试|5,800,000–5,800,499|
|两次展示视频|5,900,000 / 5,900,001|

先按验证成功次数选模，再统一打开新测试集；并列时优先较早训练步数、较小训练种子。冻结后仍每步运行神经网络，只是不再更新脑内连接。

恢复全部状态不等于 CUDA 后续轨迹逐位确定。稀疏加法的浮点顺序可能导致复测差异；报告实际结果，不通过反复测试挑选更高成绩。

## 6. 其他路线

- 整脑群体读出：`scripts/run_population_suite.py` / `train_population.py`，见 [POPULATION_PROTOCOL](POPULATION_PROTOCOL.md)。
- 可训练节点动态、模仿学习与 DAgger：先 `prepare_learning.py`，再 `run_learning_suite.py`，见 [LEARNING_PROTOCOL](LEARNING_PROTOCOL.md)。
- 旧网页控制器同图评估：`scripts/evaluate.py --modes base reflex fly --episodes 10 --output results/evaluation.json`，需由项目 Conda Python 执行。

各脚本 `--help` 提供完整参数；预处理与历史汇总脚本部分采用固定路径，先阅读对应模块说明。
