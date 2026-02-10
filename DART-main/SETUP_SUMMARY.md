# RTX 5070 Ti Setup Summary

## 🎯 Quick Reference

### One-Line Setup
```bash
bash setup_rtx5070ti_complete.sh
```

### Run Demo
```bash
conda activate DART
source ./demos/run_demo.sh
```

## 📋 Critical Version Requirements

| Package | Version | Why |
|---------|---------|-----|
| Python | 3.10+ | PyTorch 2.10+ requirement |
| numpy | **1.21.5** | Code uses deprecated `np.float` |
| PyTorch | 2.10.0+cu128 | sm_120 support |
| matplotlib | 3.3.4 | Compatible with numpy 1.21.5 |
| setuptools | <70 | CLIP needs pkg_resources |

## ⚠️ Common Pitfalls

### 1. numpy Version
❌ **Don't**: Use numpy 1.23+ or 2.x
✅ **Do**: Lock to numpy 1.21.5
```bash
pip install numpy==1.21.5 --force-reinstall --no-deps
```

### 2. CLIP Installation
❌ **Don't**: `pip install git+https://github.com/openai/CLIP.git`
✅ **Do**: Clone and install with --no-build-isolation
```bash
git clone https://github.com/openai/CLIP.git clip_repo
cd clip_repo
pip install -e . --no-build-isolation
```

### 3. pytorch3d Build
❌ **Don't**: `pip install -e .`
✅ **Do**: Use --no-build-isolation
```bash
pip install -e . --no-build-isolation
```

## 🚀 Performance Expectations

- **GPU Memory**: ~4.5 GB / 12 GB
- **GPU Utilization**: 5-10% (inference is fast!)
- **CPU Utilization**: 70-100% (pyrender visualization)
- **Inference Speed**: <100ms per motion primitive

**Note**: High CPU usage is NORMAL for real-time visualization.

## 📁 Important Files

1. **environment_py310_verified.yml** - Complete dependency specification
2. **setup_rtx5070ti_complete.sh** - Automated installation
3. **INSTALL_RTX5070Ti.md** - Full installation guide
4. **verify_installation.py** - Verification script
5. **quick_check.sh** - Fast health check

## 🔍 Verification Commands

```bash
# Check environment
conda activate DART
python verify_installation.py

# Quick check
bash quick_check.sh

# Check GPU
nvidia-smi

# Check versions
python -c "import torch, numpy, pytorch3d, clip; print(f'PyTorch: {torch.__version__}\nnumpy: {numpy.__version__}\npytorch3d: {pytorch3d.__version__}')"
```

## 🎮 Demo Commands

### Interactive Demo
```bash
source ./demos/run_demo.sh
```
Prompts: "walk forward", "jump", "turn left", "sit down", etc.

### Headless Generation
```bash
source ./demos/rollout.sh
```
Lower CPU usage, outputs .pkl/.npz for Blender rendering.

## 🐛 Troubleshooting Quick Fixes

### numpy AttributeError
```bash
pip install numpy==1.21.5 --force-reinstall --no-deps
```

### matplotlib ImportError
```bash
pip install matplotlib==3.3.4
```

### CLIP Installation Failed
```bash
pip install 'setuptools<70'
cd /path/to/clip_repo
pip install -e . --no-build-isolation
```

### pytorch3d Build Error
```bash
pip install -e . --no-build-isolation
```

## ✅ Success Indicators

When everything works:
- ✅ `verify_installation.py` shows all green checkmarks
- ✅ Demo window opens with 3D human model
- ✅ Text prompts generate smooth animations
- ✅ GPU memory usage ~4.5 GB
- ✅ No CUDA errors or warnings

## 📊 System Requirements

- **GPU**: RTX 50-series (sm_120) or compatible
- **VRAM**: 6 GB minimum, 12 GB recommended
- **RAM**: 16 GB minimum
- **Storage**: ~10 GB for environment + models
- **OS**: Ubuntu 20.04+, WSL2, or compatible Linux

## 🔗 Key Differences from Original

| Aspect | Original (RTX 4090) | RTX 5070 Ti |
|--------|---------------------|-------------|
| Python | 3.8 | 3.10 |
| PyTorch | 2.1.0 | 2.10.0 |
| CUDA | 11.8 | 12.8 |
| Arch | sm_90 | sm_120 |

## 💡 Pro Tips

1. **Save numpy version**: Always reinstall with `--no-deps` to prevent auto-upgrade
2. **CPU usage is normal**: pyrender is CPU-intensive for real-time viz
3. **Use headless mode**: For batch generation, use `rollout.sh`
4. **Check before building**: Run `quick_check.sh` before long builds
5. **CLIP needs special care**: setuptools<70 + --no-build-isolation

## 📝 Installation Log

Track your installation with:
```bash
conda activate DART
python -c "import torch, numpy, pytorch3d, clip; print('All imports successful')"
python verify_installation.py > install_verification_$(date +%Y%m%d).log
```

---

**Created**: 2026-02-10
**Tested**: RTX 5070 Ti Laptop GPU
**Status**: ✅ Fully Working
**Demo**: Interactive text-to-motion verified
