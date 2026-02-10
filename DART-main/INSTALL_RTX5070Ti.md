# DART Installation Guide for RTX 5070 Ti (sm_120)

Complete installation guide for running DART on NVIDIA RTX 50-series GPUs with Blackwell architecture (sm_120 compute capability).

## Tested Configuration

- **GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120)
- **Driver**: 591.74 (CUDA 13.1)
- **OS**: Ubuntu 22.04.4 LTS / WSL2
- **Python**: 3.10.19
- **PyTorch**: 2.10.0+cu128
- **pytorch3d**: 0.7.9 (compiled from source)
- **CLIP**: 1.0

## Quick Start

### Option 1: Automated Installation (Recommended)

```bash
cd /home/lilia_chen/workspace/Frankenstein-backend/DART-main
bash setup_rtx5070ti_complete.sh
```

This script will:
1. Create conda environment from `environment_py310_verified.yml`
2. Install PyTorch 2.10 with CUDA 12.8
3. Install CLIP from source
4. Build pytorch3d from source
5. Verify the installation

### Option 2: Manual Installation

#### 1. Create Conda Environment

```bash
conda env create -f environment_py310_verified.yml
conda activate DART
```

#### 2. Install PyTorch with sm_120 Support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Verify PyTorch:**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

#### 3. Install CLIP

```bash
cd /home/lilia_chen/workspace
git clone https://github.com/openai/CLIP.git clip_repo
cd clip_repo
pip install -e . --no-build-isolation
```

#### 4. Build pytorch3d

```bash
cd /home/lilia_chen/workspace/pytorch3d
pip install -e . --no-build-isolation
```

**Note**: This takes 10-20 minutes. No progress bar is shown, which is normal.

#### 5. Verify Installation

```bash
cd /home/lilia_chen/workspace/Frankenstein-backend/DART-main
python verify_installation.py
```

## Running Demos

### Interactive Text-to-Motion Demo

```bash
conda activate DART
cd /home/lilia_chen/workspace/Frankenstein-backend/DART-main
source ./demos/run_demo.sh
```

**Usage:**
- A pyrender window will open showing a 3D human model
- Input text prompts in the terminal (e.g., "walk forward", "jump", "turn left")
- The model will generate and visualize the motion in real-time

**Example prompts:**
- walk forward
- run in circles
- jump
- sit down
- wave hand
- turn 180 degrees
- lie on the ground

### Headless Motion Generation

```bash
source ./demos/rollout.sh
```

This generates motion sequences without real-time visualization. Output files can be rendered in Blender for better performance.

## Critical Dependencies

### ⚠️ Version-Locked Requirements

These versions MUST match exactly to avoid errors:

| Package | Version | Reason |
|---------|---------|--------|
| Python | 3.10+ | Required for PyTorch 2.10+ |
| numpy | 1.21.5 | Code uses deprecated `np.float` (removed in 2.0+) |
| PyTorch | 2.10.0+cu128 | First version with sm_120 support |
| matplotlib | 3.3.4 | Newer versions require numpy>=1.23 |
| scipy | 1.10.1 | Compatible with numpy 1.21.5 |
| pandas | 2.0.0 | Compatible with numpy 1.21.5 |
| setuptools | <70 | CLIP requires pkg_resources (removed in 70+) |
| scikit-image | 0.21.0 | Compatible with numpy 1.21.5 |

### Why These Specific Versions?

1. **numpy 1.21.5**: DART code uses `np.float` which was deprecated in NumPy 1.20 and removed in 2.0. Using numpy>=1.23 causes:
   ```
   AttributeError: module 'numpy' has no attribute 'float'
   ```

2. **matplotlib 3.3.4**: Versions 3.4+ require numpy>=1.23, which conflicts with our numpy 1.21.5 requirement.

3. **setuptools <70**: CLIP's setup.py imports `pkg_resources`, which was removed in setuptools 70+.

4. **PyTorch 2.10.0+cu128**: First stable release with CUDA 12.8 and sm_120 (Blackwell) architecture support.

## Common Issues and Solutions

### Issue 1: numpy AttributeError

**Error:**
```
AttributeError: module 'numpy' has no attribute 'float'
```

**Solution:**
```bash
pip install numpy==1.21.5 --force-reinstall --no-deps
```

### Issue 2: CLIP Installation Fails

**Error:**
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**Solution:**
```bash
pip install 'setuptools<70'
cd /path/to/clip_repo
pip install -e . --no-build-isolation
```

### Issue 3: pytorch3d Build Fails

**Error:**
```
No module named 'torch'
```

**Solution:**
Use `--no-build-isolation` flag:
```bash
pip install -e . --no-build-isolation
```

### Issue 4: High CPU Usage During Demo

**Expected Behavior:**
- GPU: 5-10% utilization (fast inference)
- CPU: 70-100% utilization (pyrender real-time visualization)

This is NORMAL. PyTorch inference on GPU is very fast (milliseconds), but pyrender's 3D rendering is CPU-intensive. To reduce CPU usage:

1. Use headless mode: `source ./demos/rollout.sh`
2. Export to Blender for offline rendering (more efficient)

### Issue 5: matplotlib Requires numpy>=1.23

**Error:**
```
ImportError: Matplotlib requires numpy>=1.23
```

**Solution:**
```bash
pip install matplotlib==3.3.4
pip install numpy==1.21.5 --force-reinstall --no-deps
```

## Performance Notes

### GPU Memory Usage
- Model loading: ~4.5 GB VRAM
- Inference: 5-10% GPU utilization (very fast)
- Total: ~4.5-5 GB / 12 GB

### CPU Usage
- Real-time visualization (pyrender): 70-100% CPU (normal)
- Headless mode: Much lower CPU usage

### Inference Speed
- Text-to-motion generation: <100ms per motion primitive
- Real-time interaction is smooth

## Differences from Original Environment

The original `environment.yml` was designed for:
- RTX 4090 (sm_90)
- Python 3.8
- PyTorch 2.1.0 + CUDA 11.8

Key changes for RTX 5070 Ti:

| Component | Original | RTX 5070 Ti |
|-----------|----------|-------------|
| Python | 3.8 | 3.10 |
| PyTorch | 2.1.0+cu118 | 2.10.0+cu128 |
| CUDA | 11.8 | 12.8 |
| Compute Capability | sm_90 | sm_120 |
| numpy | 1.21.5 | 1.21.5 (same) |

## Verification Tests

After installation, run:

```bash
python verify_installation.py
```

Expected output:
```
✓ PyTorch installed: version 2.10.0+cu128
✓ CUDA available: version 12.8
✓ Device name: NVIDIA GeForce RTX 5070 Ti Laptop GPU
✓ Compute capability: sm_120
✓ Supported architectures: sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
✓ Architecture list includes sm_90/sm_120 support
✓ Matrix multiplication successful
✓ pytorch3d installed: version 0.7.9
✓ Created mesh with 3 vertices on GPU
✓ All checks passed! Environment is ready.
```

## Files Included

- `environment_py310_verified.yml` - Complete conda environment specification
- `setup_rtx5070ti_complete.sh` - Automated installation script
- `verify_installation.py` - Installation verification script
- `quick_check.sh` - Fast health check
- `INSTALL_RTX5070Ti.md` - This document

## Troubleshooting

### Check PyTorch CUDA Support
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Check GPU Compute Capability
```bash
python -c "import torch; print(torch.cuda.get_device_capability(0))"
```

### Check numpy Version
```bash
python -c "import numpy; print(numpy.__version__)"
```

### Check All Dependencies
```bash
bash quick_check.sh
```

### View GPU Usage
```bash
nvidia-smi
```

## Contact

If you encounter issues not covered in this guide:
1. Check the [DART repository issues](https://github.com/zkf1997/DART/issues)
2. Verify your NVIDIA driver supports CUDA 12.8+
3. Ensure you're using the exact versions specified in `environment_py310_verified.yml`

## License

This installation guide is provided as-is for the DART project. DART itself follows its own license terms.

---

**Last Updated**: 2026-02-10
**Verified By**: RTX 5070 Ti sm_120 testing
**Status**: ✅ All demos working
