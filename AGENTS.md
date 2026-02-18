# Agent Guide — Frankenstein-backend

FastAPI WebSocket 后端，文本转动作生成。支持 `mock` / `dart` / `motionstreamer` 三种模型，通过 `MOTION_MODEL` 环境变量切换。

---

## 架构

```
请求路径：main.py → ws.py → services/generator.py → services/registry.py → models/
```

| 文件 | 职责 |
|------|------|
| `app/main.py` | FastAPI app，startup warmup（预加载模型） |
| `app/ws.py` | WebSocket handler，解析消息类型，调用 SessionManager |
| `app/schemas.py` | 所有 Pydantic 消息类型（GenerateRequest、MotionFrame 等） |
| `app/services/generator.py` | `stream_generation` 协程，`SessionManager` 管理并发请求 |
| `app/services/registry.py` | 根据 `MOTION_MODEL` 返回对应生成器实例 |
| `app/models/base.py` | `MotionGenerator` 抽象基类，`generate()` 为 async generator |
| `app/models/mock.py` | Mock 生成器，不依赖任何 ML 库 |
| `app/models/dart.py` | DART 生成器 |
| `app/models/motionstreamer.py` | MotionStreamer 生成器（含 Runtime 单例） |

---

## 运行

```bash
./start.sh   # 读取 .env，启动 uvicorn 0.0.0.0:8000
```

开发时切换模型只需改 `.env` 里的 `MOTION_MODEL`，重启即可。

---

## 添加新模型

1. 在 `app/models/` 下新建文件，继承 `MotionGenerator`，实现 `generate()` async generator
2. 在 `app/services/registry.py` 注册新模型名
3. `.env` 设 `MOTION_MODEL=<新名称>`

---

## MotionStreamer 关键细节

- **单例**：`MotionStreamerRuntime` 用 `threading.Lock` 保护，首次调用 `get()` 时初始化
- **冷启动**：T5-XXL 10GB，加载约 30-60 秒；startup warmup 在服务启动时触发，避免首次请求超时
- **扩散步数**：`MS_DIFF_STEPS`（默认 10，原始硬编码 50），通过替换 `trans_encoder.diff_loss.gen_diffusion` 实现，加载 checkpoint 后替换
- **原生帧率**：HumanML3D 20fps，生成时固定用 `duration × 20` 计算帧数，与前端传入的 `fps` 无关
- **sys.argv 冲突**：MotionStreamer 的 `option_transformer.parse_args()` 会读 `sys.argv`，初始化时需临时替换为 `sys.argv[:1]`（见 `_init_runtime`）

## DART 关键细节

- numpy 锁定 **1.21.5**（代码使用已废弃的 `np.float`）
- RTX 5070 Ti 需要从源码编译 pytorch3d、手动 clone CLIP

## RTX 5070 Ti 通用坑

- setuptools 需 `<70`
- PyTorch 用 `--index-url https://download.pytorch.org/whl/cu128`
- 详见 `DART-main/SETUP_SUMMARY.md`

---

## WebSocket 协议

完整规范见 [BACKEND_OUTPUT_SPEC.md](BACKEND_OUTPUT_SPEC.md)。

简要：
- 连接后服务端发 `handshake`
- 客户端发 `generate`（含 text、duration_seconds、fps）
- 服务端逐帧返回 `frame`，最后发 `done`
- 客户端可随时发 `cancel`

四元数格式：`[x, y, z, w]`，坐标系：Y-up 右手系。

---

## 注意事项

- `.env` 不提交 git（含密钥/路径）
- MotionStreamer 权重、T5 模型、DART 权重均在 `.gitignore` 中排除
- `generate_frames_token` 是逐 token 解码的实验性方法，当前 `generate()` 不使用它（用 `_infer_motion_272` 批量生成）
