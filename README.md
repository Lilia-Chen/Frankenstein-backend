# Frankenstein-backend

FastAPI WebSocket 后端，实时文本转动作生成。支持三种模型：`mock`、`dart`、`motionstreamer`。

WebSocket 端点：`ws://<host>:8000/ws/motion`
协议详情见 [BACKEND_OUTPUT_SPEC.md](BACKEND_OUTPUT_SPEC.md)

---

## 快速启动

```bash
# 编辑 .env，设置 MOTION_MODEL=mock / dart / motionstreamer
./start.sh
```

---

## 模型配置

### mock（无需任何依赖）

```env
MOTION_MODEL=mock
```

直接运行 `./start.sh`，用于前端调试。

---

### DART

**环境：** conda `DART`（Python 3.10，见 `DART-main/environment.yml`）
RTX 5070 Ti 用户见 `DART-main/SETUP_SUMMARY.md`

```env
MOTION_MODEL=dart
DART_ROOT=DART-main
DART_DENOISER_CKPT=DART-main/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt
DART_DEVICE=cuda
```

---

### MotionStreamer

**环境：** conda `MotionStreamer`（Python 3.10，见 `MotionStreamer/environment_rtx5070ti.yml`）

**第一次使用需下载模型权重：**

```bash
# 1. Causal TAE
cd MotionStreamer && python humanml3d_272/prepare/download_Causal_TAE_t2m_272_ckpt.py

# 2. Transformer
cd MotionStreamer && python humanml3d_272/prepare/download_t2m_model_ckpt.py

# 3. sentence-T5-XXL (~10 GB)
huggingface-cli download sentence-transformers/sentence-t5-xxl \
  --local-dir MotionStreamer/sentencet5-xxl/
```

```env
MOTION_MODEL=motionstreamer
MS_ROOT=MotionStreamer
MS_DEVICE=cuda
MS_TAE_CKPT=MotionStreamer/Causal_TAE/net_last.pth
MS_TRANS_CKPT=MotionStreamer/Experiments/t2m_model/latest.pth
MS_MEAN_STD_ROOT=MotionStreamer/humanml3d_272/mean_std
MS_TEXT_ENCODER=MotionStreamer/sentencet5-xxl/
MS_DIFF_STEPS=10
```

> `MS_DIFF_STEPS`：扩散采样步数（默认 10，原始 50）。值越小越快，值越大质量越高。

---

## 项目结构

```
app/
  main.py          # FastAPI 应用，startup warmup
  ws.py            # WebSocket handler
  schemas.py       # Pydantic 消息类型
  services/
    generator.py   # stream_generation / SessionManager
    registry.py    # 按 MOTION_MODEL 选择生成器
  models/
    base.py        # MotionGenerator 抽象基类
    mock.py        # Mock 生成器
    dart.py        # DART 生成器
    motionstreamer.py  # MotionStreamer 生成器
DART-main/         # DART 源码（子模块）
MotionStreamer/    # MotionStreamer 源码（子模块）
BACKEND_OUTPUT_SPEC.md  # WebSocket 协议规范
```

---

## .env 完整参考

```env
# 选择模型: mock | dart | motionstreamer
MOTION_MODEL=motionstreamer

# DART
DART_ROOT=DART-main
DART_DENOISER_CKPT=DART-main/mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt
DART_DEVICE=cuda

# MotionStreamer
MS_ROOT=MotionStreamer
MS_DEVICE=cuda
MS_TAE_CKPT=MotionStreamer/Causal_TAE/net_last.pth
MS_TRANS_CKPT=MotionStreamer/Experiments/t2m_model/latest.pth
MS_MEAN_STD_ROOT=MotionStreamer/humanml3d_272/mean_std
MS_TEXT_ENCODER=MotionStreamer/sentencet5-xxl/
MS_UNIT_LENGTH=4
MS_BLOCK_SIZE=78
MS_DIFF_STEPS=10

# Server
HOST=0.0.0.0
PORT=8000
```
