"""Write the measured long-training report after the full audit succeeds."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/plasticity/staged-4m'
LOCAL='..'


def main():
    d=json.loads((OUT/'comparison.json').read_text())
    assert d['audit']['validation_selection_committed_before_fresh_test']
    steps=d['steps'];first=d['fresh'][str(steps[0])];last=d['fresh'][str(steps[-1])]
    selected=d['selected'];policy=d['single_policy'];score=policy['test']
    def fmt(s):return f"{s['mean']*100:.1f} ± {s['sample_sd']*100:.1f}%"
    def pct(x):return f'{x*100:.1f}%'
    def link(name):return f'{LOCAL}/results/plasticity/staged-4m/{name}'
    gain=100*(last['success_rate']['mean']-first['success_rate']['mean'])
    rows=[]
    for c in d['selected_checkpoints']:
        i=d['seeds'].index(c['seed'])
        rows.append(f"|{c['seed']}|{c['total_steps']:,}|{pct(c['successes']/c['episodes'])}|{pct(selected['success_rate']['values'][i])}|")
    outcome=[]
    for step in steps:
        m=d['fresh'][str(step)]
        counts=[sum(m[key]['values']) for key in ['successes','collisions','timeouts']]
        outcome.append(f"|{step:,}|{counts[0]}|{counts[1]}|{counts[2]}|{m['mean_task_return']['mean']:.2f}|{m['mean_angular_action_change']['mean']:.3f}|")
    validation=[]
    for step in steps:
        s=d['validation'][str(step)]['success_rate']
        values='|'.join(pct(x) for x in s['values'])
        validation.append(f"|{step:,}|{values}|{pct(s['mean'])}|")
    nonmonotone=any(d['fresh'][str(b)]['success_rate']['mean']<d['fresh'][str(a)]['success_rate']['mean'] for a,b in zip(steps,steps[1:]))
    h=d['highest_test_point_descriptive']
    seconds=d['pipeline_seconds'];minutes=seconds/60
    jitter=100*(last['mean_angular_action_change']['mean']/first['mean_angular_action_change']['mean']-1)
    text=f'''# 分阶段模型续训至 419 万步：本轮最高测试成功率 {pct(h['success_rate'])}

日期：2026-09-16。三个训练种子均已从 1,048,576 步继续到 **4,194,304 总环境步**，三进程并行执行。脑内连接从 65,536 步起冻结，后续只学习读出。

**本轮观察到的最高测试成功率为 {pct(h['success_rate'])}，来自种子 {h['seed']}、{h['total_steps']:,} 步模型。** 这是 21 个固定测试点中的事后最高值，不是理论上限；每种子预先按验证集选出的结果见下表，跨种子的单模型选择另按预定并列规则处理。

**在同一批 500 张新测试地图上，起点平均成功率为 {fmt(first['success_rate'])}，最终 419 万步为 {fmt(last['success_rate'])}，变化 {gain:+.1f} 个百分点。** 各种子按验证集选择检查点后，新测试平均成功率为 **{fmt(selected['success_rate'])}**。

严格按跨种子预定规则选出的单模型是种子 **{policy['seed']}**、**{policy['total_steps']:,} 步**：验证成功率 **{pct(policy['successes']/policy['episodes'])}**，独立测试 **{score['successes']}/500 = {pct(score['success_rate'])}**，Wilson 95% 区间为 **{pct(policy['wilson_95'][0])}–{pct(policy['wilson_95'][1])}**。验证集并列时优先较早步数，所以它与事后测试最高模型可能不同。

## 同图比较：继续训练的效果

全部训练完成后，在新地图 5,800,000–5,800,499 上统一评估七个预定节点。表中为三个训练种子的均值 ± 样本标准差，不是置信区间。

{(OUT/'result-table.md').read_text().strip()}

{'平均成绩存在中途回落，更多训练步数没有带来单调提升。' if nonmonotone else '这七个节点的平均成绩未出现回落；节点间仍可能存在未观测到的波动。'} 事后查看完整测试曲线，最高单点为种子 {h['seed']}、{h['total_steps']:,} 步的 **{pct(h['success_rate'])}**；它只作为曲线描述，测试结果没有用于选模。

![固定训练节点成功率]({link('comparison.png')})

曲线图纵轴为 70%–100%。细线为各训练种子，粗线和阴影为均值及样本标准差；上方数字是各节点的均值，星号标出每个种子事先按验证集选出的检查点。左图为新测试集，右图为验证集。

## 验证集选择与独立检验

验证地图为 5,410,000–5,410,199，共 200 张，所有模型使用相同地图。选择规则为成功次数最多，并列时优先较早步数；跨种子再并列时优先较小种子号。选择结果及检查点哈希在新测试开始前写入记录，并通过时间顺序和文件哈希审计。

|种子|按验证集选择的步数|验证成功率|500 张新地图成功率|
|---|---:|---:|---:|
{chr(10).join(rows)}

每种子选模后的测试均值为 **{fmt(selected['success_rate'])}**。三个模型各跑相同 500 张地图，不是 1,500 张独立地图；单模型的 Wilson 区间仅描述该固定策略在本地图分布下的抽样不确定性。

|总步数|种子 71|种子 72|种子 73|均值|
|---|---:|---:|---:|---:|
{chr(10).join(validation)}

## 失败类型与动作平滑性

以下计数汇总每节点的 3 × 500 个评估回合；回报和平滑性先逐回合统计，再在三个模型间平均。

|总步数|成功|碰撞|超时|平均任务回报|平均角速度指令变化|
|---|---:|---:|---:|---:|---:|
{chr(10).join(outcome)}

角速度指令变化为相邻控制步角速度指令差的绝对值，单位 rad/s。起点到终点变化 **{jitter:+.1f}%**。该指标来自各模型实际走过的不同轨迹和回合长度，本轮没有单独优化平滑性。

![成功和失败比例]({link('failure-types.png')})

## 与上一轮 85.8% 的衔接

在上一轮地图 5,700,000–5,700,199 上，起点重测为 **{fmt(d['previous_test']['before']['success_rate'])}**，最终模型为 **{fmt(d['previous_test']['after']['success_rate'])}**，验证选出的模型为 **{fmt(d['previous_selected']['success_rate'])}**。这批地图仅做次要复测，没有用于选择新模型。

上一轮保存的起点三个成绩为 {', '.join(pct(x) for x in d['previous_recorded_baseline'])}。本轮重测三个成绩为 {', '.join(pct(x) for x in d['previous_test']['before']['success_rate']['values'])}。CUDA 稀疏加法沿用非全局确定性设置，重复执行可能出现浮点轨迹差异；记录保留实际结果，未通过重复测试挑选高分。

## 训练设置与核验

- 真实 MaleCNS 子图：4,096 个神经元、459,342 条允许连接、1,314 个下降神经元读出。感觉编码、简化神经动态与机器人动作残差映射沿用既有工程建模。
- 每组新增 3,145,728 环境步，共新增 9,437,184 步；模型、Adam、环境、观测、神经状态、回合标记和 CPU/GPU RNG 完整恢复。
- 仅 actor/critic 的 3,960 个参数参与训练。脑内连接在全部保存节点均与 65,536 步来源逐字节相同；其他固定模型缓冲区也未变。Adam 更新计数由 65,536 到 262,144。
- 学习率 0.0003、8 环境、rollout 128、minibatch 64、每次 4 轮优化、裁剪 0.2、折扣 0.99、GAE 0.95、奖励缩放 0.1、固定标准差 0.4、梯度上限 0.5，全程不变。采样/PPO 核心与上一轮逐字一致。
- 两端项目 Conda 的 **46 项测试通过**。额外真实模型短流程通过，调试地图未计入正式结果。原始模型、训练源码、输入和结果哈希已保存。

## 资源和产物

运行于 RTX 5090 工作站 的物理 GPU 1，三个独立进程并行、每进程 4 个 CPU 线程。正式训练及统一评估的墙钟时间约 **{minutes:.1f} 分钟**。各训练进程记录的训练时间分别为 {', '.join(f"{r['train_seconds']/60:.1f}" for r in d['runs'])} 分钟，这些时间互相重叠，不能相加当作总等待时间。单进程 PyTorch 最大已分配显存为 **{d['gpu_peak_mib']:.1f} MiB**；该数字不包括其他进程和 CUDA 上下文/缓存。启动时三进程整卡占用约 4 GB。

- [种子 72 按验证集选出的模型，本轮测试最高 92.6%]({link('main/staged-72/best-validation.pt')})。
- [按跨种子并列规则预先选出的单模型]({link('main/best-validation.pt')})。
- [完整数值与审计结果]({link('comparison.json')})、[选模记录]({link('main/selection.json')})。
- [训练及评估协议]({LOCAL}/docs/STAGED_4M_PROTOCOL.md)。
- 实验产物目录：`results/plasticity/staged-4m/`（本地生成，不随 Git 分发）。

本轮是当前二维仿真避障分布下的预算扩展实验。更长时间、其他训练种子或其他地图分布仍可能改变结果，不能据此确定生物结构的学习上限或真实机器人性能。
'''
    text=re.sub(r'!?\[([^\]]+)\]\(\.\./(results/[^)]+)\)',
                lambda m:f'{m[1]}（本地产物）：`{m[2]}`',text)
    text=text.replace('../docs/STAGED_4M_PROTOCOL.md','STAGED_4M_PROTOCOL.md')
    (ROOT/'docs/STAGED_4M_TRIAL.md').write_text(text)
    print(ROOT/'docs/STAGED_4M_TRIAL.md')


if __name__=='__main__':main()
