# 功能实现与代码地图

## 1. 三条模型路线

|路线|核心实现|神经输出|学习内容|当前入口|
|---|---|---|---|---|
|早期整脑演示|`flylab/brain.py`|左右 DNp01 活动|整脑固定；可外接 SB3 PPO|网页、`train_ppo.py`|
|整脑群体读出|`flylab/population.py`、`population_ppo.py`|1,314 个下行神经元及活动特征|固定脑＋PPO；另有可训练编码/节点动态路线|`train_population.py`、`train_learning.py`|
|真实子图连接学习|`flylab/plasticity.py`|1,314 个下行神经元状态|读出、脑内连接＋读出、局部塑性、联合后冻结|`train_plasticity.py` 和分阶段训练器|

网页中的“果蝇脑”和最新 4M 策略不同。前者使用全图 LIF，最新策略使用 4,096 节点的简化连续状态网络。

## 2. 环境与控制

|文件|职责|
|---|---|
|`flylab/world.py`|12 × 8 m 二维世界、圆形障碍、9 路测距、运动积分、碰撞、奖励、终止和轨迹|
|`flylab/controllers.py`|朝目标的基础动作、规则避障对照、残差合成与动作限幅|
|`flylab/gym_env.py`|Gymnasium 观测/动作/奖励接口，供 SB3 PPO 使用|
|`flylab/population_ppo.py`|批量环境、观测归一化、MLP/GRU 对照、循环策略、GAE|
|`flylab/server.py`|本地 HTTP 服务、仿真运行线程、状态接口和模式切换|
|`web/`|地图、传感射线、动作分解、神经活动和记录导出|

每个控制步为 0.1 秒，动作是线速度与角速度。基础控制只朝向目标；规则避障是显式对照。学习策略的动作经缩放后成为残差，与基础动作合成：

```text
final_action = bound(base_action + bounded_residual)
linear residual ∈ [-0.75, 0.25] m/s
angular residual ∈ [-1.8, 1.8] rad/s
final linear ∈ [0, 1.1] m/s
final angular ∈ [-2.2, 2.2] rad/s
```

## 3. 连接数据与神经计算

- `scripts/prepare_data.py`：下载/导入注释、神经递质预测与连接数据；构造稀疏矩阵并保存来源记录。
- `flylab/download.py`：代理、TLS、错误说明、重试、续传、文件版本和校验。
- `flylab/brain.py` / `propagation.py`：全图 LIF；SciPy 与 Numba 活跃神经元传播后端。
- `flylab/population.py`：GPU 全图稀疏传播、方向输入分组、群体读出；真实与扰乱连接对照。
- `scripts/prepare_plasticity.py`：保留输入到全部下行输出的路径，补足 4,096 节点，诱导子图并归一化；校验输出可达性。
- `flylab/plasticity.py`：固定边位置上的可微消息传播；每步内部执行 5 次 `tanh` 更新。允许的边强度可调整，投影保留原始符号并限幅。

稀疏矩阵采用 `[目标神经元, 来源神经元]`。突触数量经过符号赋值和归一化成为初始化权重；它不是测量得到的突触功能强度。

最新策略的 23 维观测包括 9 路距离、9 路接近速度、5 维目标/机器人上下文。前 18 维经过固定编码刺激神经网络；1,314 个下行状态标准化后与 5 维上下文一起进入线性 actor/critic。没有距离观测直接绕过脑网络进入该读出。

## 4. 学习方法

|文件|职责|
|---|---|
|`scripts/train_ppo.py`、`flylab/ppo_controller.py`|早期 SB3 PPO 训练、加载和网页推理|
|`scripts/train_population.py`|固定整脑群体读出与 MLP/GRU/扰乱连接 PPO 对照|
|`flylab/learning.py`、`teacher.py`|可训练感觉编码/节点动态；几何老师、模仿学习和 DAgger|
|`scripts/train_learning.py`|模仿学习 → DAgger → PPO 实验|
|`scripts/train_plasticity.py`|相同子图上的 `readout`、`ppo_edges`、`three_factor` 三组学习|
|`scripts/extend_plasticity.py`|三组方法续训至 262,144 步|
|`scripts/train_frozen_readout.py`|从 65,536 步联合模型冻结连接，保留读出 Adam 状态|
|`scripts/extend_staged.py`|完整状态续训至 1,048,576 步|
|`scripts/extend_staged_4m.py`|续训至 4,194,304 步，支持三个种子并行|

`three_factor` 使用每环境独立资格迹、局部前后神经元活动及 TD 奖励调制；不通过 PPO 梯度更新脑内连接。这是工程化局部学习规则，未模拟完整多巴胺解剖回路；现有实验未获得有效导航成绩。

## 5. 冻结、续训与评估

- `flylab/staged.py`：冻结已学连接，从优化器参数组移除连接，保留 actor/critic 的优化状态。
- `flylab/resume.py`：保存/恢复模型、优化器、环境状态、神经状态、观测和随机数状态。
- `flylab/continuation.py`：校验和恢复完整 rollout 状态。
- `flylab/selection.py`：仅按验证成功次数选模；并列优先较早步数，再优先较小种子。
- `scripts/summarize_*.py`：读取固定实验目录，汇总逐种子/逐地图结果和一致性检查。
- `scripts/plot_*.py`：从已有结果生成图表。
- `scripts/archive_*.py`：保存源文件、输入、模型和产物哈希；不能替代训练前预定协议。

同一次比较共用 held-out 地图。训练、验证、测试种子分开；最新实验先写入选模记录，再打开测试集。CUDA 稀疏加法可能产生浮点轨迹差异，冻结连接仍要求逐字节一致。

## 6. 视频管线

`record_navigation_demo.py` 保存一次预选地图的真实状态、控制动作和下行活动。`render_navigation_demo.py` 将 10 Hz 控制记录渲染为 30 fps 画面；位置/朝向只作显示插值。`encode_navigation_demo.py` 检查完整帧数、编码 H.264，并完整解码校验。

录像入口目前锁定演示所用的种子 72、4,194,304 步检查点及其图哈希，防止误标模型。推广到其他实验需要同步修改断言和画面中的实验标签。

## 7. 扩展边界

机器人适配沿用 observation → base action → residual → bounded action → reward 接口。尚未实现 ROS 2、MuJoCo、Isaac Sim 或硬件适配器，也未实现机械臂残差训练。下一步应先换真实机器人动力学仿真，匹配控制周期和动作空间，再重新训练与评估。
