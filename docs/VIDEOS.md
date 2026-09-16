# 避障视频

## 已有演示

两段演示使用同一个种子 72、4,194,304 步模型。每张新地图预先选定，只运行一个回合。

|地图|结果|仿真时长|路径长度|最小车体间隙|
|---|---|---:|---:|---:|
|5900000|到达|9.4 s|10.14 m|9.9 cm|
|5900001|到达|11.2 s|11.07 m|10.2 cm|

视频在 [v0.1.0 发布页](https://github.com/pgq18/ConnectomePilot/releases/tag/v0.1.0)，文件名分别为 `navigation-map-5900000.mp4` 和 `navigation-map-5900001.mp4`。它们是单回合展示，不能取代独立测试成功率。

## 记录一次回合

先按[训练指南](TRAINING.md)准备数据和检查点。默认路径是 `results/plasticity/staged-4m/main/staged-72/final.pt`。模型权重不随 Git 源码分发。

```bash
PROJECT_DIR="$(pwd)"
CUDA_VISIBLE_DEVICES=1 "$PROJECT_DIR/.conda/bin/python" scripts/record_navigation_demo.py \
  --device cuda:0 --scene 5900002 --output results/videos/map-5900002
```

输出真实策略轨迹 `rollout.json`、逐控制步下行神经状态 `neural-activity.npz`，同时记录模型/图/脚本哈希。已有输出目录会被拒绝，避免覆盖旧记录。

录像入口当前专用于种子 72、4M 检查点；即使传 `--checkpoint`，仍会检查训练种子、步数和图哈希。其他模型需要同步调整检查与渲染标签。

## 渲染与编码

```bash
"$PROJECT_DIR/.conda/bin/python" -m pip install -r requirements-viz.txt
"$PROJECT_DIR/.conda/bin/python" scripts/render_navigation_demo.py \
  results/videos/map-5900002 --preview
"$PROJECT_DIR/.conda/bin/python" scripts/render_navigation_demo.py \
  results/videos/map-5900002
"$PROJECT_DIR/.conda/bin/python" scripts/encode_navigation_demo.py \
  results/videos/map-5900002
```

渲染支持 Mac 系统中文字体与常见 Linux Noto CJK / 文泉驿字体。若自动检测不到，传 `--font /absolute/path/chinese-font.ttc --mono-font /absolute/path/mono.ttf`。

编码需要 FFmpeg/FFprobe。已有工具可直接使用；也可通过 `conda install --prefix "$PROJECT_DIR/.conda" -c conda-forge ffmpeg` 安装到项目环境，再显式传 `--ffmpeg "$PROJECT_DIR/.conda/bin/ffmpeg" --ffprobe "$PROJECT_DIR/.conda/bin/ffprobe"`。

画面为 1600 × 900、30 fps。策略按 10 Hz 运行；画面仅插值位置与朝向，1 倍仿真速度播放，首尾附静帧。显示轨迹、距离射线、基础/残差/最终动作、1,314 个下行神经元的简化模型状态。

编码前确认所有 PNG 帧已完成，特别是在跨机器传输时。编码器检查帧数、尺寸、时长和 H.264 格式，完整解码通过后才生成 `avoidance.mp4`。保留 `render.json`、`video-info.json` 与 `encoding-audit.json` 便于核验。
