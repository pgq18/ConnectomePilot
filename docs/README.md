# 文档索引

## 使用与实现

- [项目首页](../README.md)
- [功能实现与代码地图](ARCHITECTURE.md)
- [训练与评估复现](TRAINING.md)
- [工作站安装和运行](WORKSTATION.md)
- [视频导出](VIDEOS.md)
- [机器人接口和后续方向](ROBOTICS.md)
- [连接数据说明](../data/README.md)
- [第三方来源和许可](THIRD_PARTY.md)

## 实验记录

各轮使用不同的测试地图和训练设置；跨轮数字不能直接当成同图对照。

|阶段|协议|结果|
|---|---|---|
|早期传感器/整脑 SB3 PPO|报告内设置|[PPO_TRIAL](PPO_TRIAL.md)|
|整脑群体读出与 MLP/GRU 对照|[POPULATION_PROTOCOL](POPULATION_PROTOCOL.md)|[POPULATION_TRIAL](POPULATION_TRIAL.md)|
|可训练节点动态与模仿学习|[LEARNING_PROTOCOL](LEARNING_PROTOCOL.md)|[LEARNING_TRIAL](LEARNING_TRIAL.md)|
|子图连接学习，65k|[PLASTICITY_PROTOCOL](PLASTICITY_PROTOCOL.md)|[PLASTICITY_TRIAL](PLASTICITY_TRIAL.md)|
|三种方法延长至 262k|[PLASTICITY_LONG_PROTOCOL](PLASTICITY_LONG_PROTOCOL.md)|[PLASTICITY_LONG_TRIAL](PLASTICITY_LONG_TRIAL.md)|
|联合后冻结，262k|[FROZEN_READOUT_PROTOCOL](FROZEN_READOUT_PROTOCOL.md)|[FROZEN_READOUT_TRIAL](FROZEN_READOUT_TRIAL.md)|
|冻结续训至 1M|[STAGED_LONG_PROTOCOL](STAGED_LONG_PROTOCOL.md)|[STAGED_LONG_TRIAL](STAGED_LONG_TRIAL.md)|
|三个种子并行续训至 4M|[STAGED_4M_PROTOCOL](STAGED_4M_PROTOCOL.md)|[STAGED_4M_TRIAL](STAGED_4M_TRIAL.md)|

历史报告保留实际成绩与失败结果。公开整理将机器专属目录替换为 `$PROJECT_DIR`，并将未随 Git 分发的产物标为本地路径。原始源码快照、检查点、日志和数据留在各轮 `results/` 归档中。
