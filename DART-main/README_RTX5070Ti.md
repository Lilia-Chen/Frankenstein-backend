# RTX 5070 Ti Setup Guide for DART

This guide explains how to set up the DART environment for **NVIDIA RTX 5070 Ti** (and other RTX 50-series GPUs with sm_120 compute capability).

## Problem

RTX 5070 Ti has compute capability **sm_120** (Blackwell architecture), which is **not supported by stable PyTorch releases** as of early 2026. You'll see this error:

```
NVIDIA GeForce RTX 5070 Ti Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
```

## Solution

Use **PyTorch nightly builds** with **CUDA 12.9** and build **pytorch3d from source**.

## Quick Setup (Automated)

```bash
cd /home/lilia_chen/workspace/Frankenstein-backend/DART-main
./setup_rtx5070ti.sh
```

This script will:
1. Create the DART conda environment (without PyTorch)
2. Install PyTorch nightly with CUDA 12.9
3. Build pytorch3d from source
4. Verify the installation

## Manual Setup

If you prefer to do it manually:

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate DART
```

### 2. Install PyTorch Nightly

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
```

### 3. Build pytorch3d from source

```bash
cd /home/lilia_chen/workspace/pytorch3d
pip install -e .
```

### 4. Verify installation

**Quick check:**
```bash
./quick_check.sh
```

**Detailed verification:**
```bash
python verify_installation.py
```

**Manual verification:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import pytorch3d; print('pytorch3d version:', pytorch3d.__version__)"
```

## Verification Tools

Three scripts are provided to verify your installation:

### 1. **quick_check.sh** - Fast environment check
```bash
./quick_check.sh
```
Performs a quick sanity check (5 seconds). Use this regularly to ensure everything is working.

### 2. **verify_installation.py** - Comprehensive verification
```bash
python verify_installation.py
```
Performs detailed tests including:
- PyTorch version and nightly build detection
- CUDA availability and version
- GPU detection and compute capability (sm_120)
- Basic GPU operations (matrix multiplication)
- pytorch3d installation and GPU mesh operations
- Architecture support analysis

### 3. **Manual checks**
```bash
conda activate DART
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Important Notes

⚠️ **Stability Warning**: PyTorch nightly builds are development versions and may have bugs or instabilities.

⚠️ **Kernel Availability**: Even with nightly builds, you might still encounter "no kernel image is available for execution" errors if the nightly build doesn't have full sm_120 kernels compiled yet.

⚠️ **System CUDA**: Make sure your system has **CUDA 12.8 or later** installed. Check with:
```bash
nvcc --version
nvidia-smi
```

## Changes Made to environment.yml

The following changes were made to support RTX 5070 Ti:

1. **Commented out conda PyTorch packages**:
   - `pytorch`, `pytorch-cuda`, `pytorch3d`, `torchvision`, `torchaudio`, `torchtriton`

2. **Commented out CUDA 11.8 libraries**:
   - `cuda-cudart`, `cuda-libraries`, `libcublas`, `libnvjpeg`, etc.

3. **Removed duplicate pip sections** that were trying to install cu128

4. **Added post-install instructions** at the end of environment.yml

## Tracking Official Support

Monitor these GitHub issues for updates on stable PyTorch support:
- [Official support for sm_120 in stable builds #164342](https://github.com/pytorch/pytorch/issues/164342)
- [Feature Request: Add support for CUDA sm_120 #159847](https://github.com/pytorch/pytorch/issues/159847)

## Reverting to RTX 40-series Setup

If you want to use this on an RTX 4090 or other older GPU, simply:
1. Uncomment the conda pytorch packages in environment.yml
2. Comment out the nightly pip installation
3. Don't build pytorch3d from source (use the conda package)

## References

- [PyTorch Forums: NVIDIA GeForce RTX 5070 Ti support](https://discuss.pytorch.org/t/nvidia-geforce-rtx-5070-ti-with-cuda-capability-sm-120/221509)
- [PyTorch Nightly Builds](https://pytorch.org/get-started/locally/)
- [pytorch3d Installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
