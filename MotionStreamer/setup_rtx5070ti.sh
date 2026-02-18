#!/usr/bin/env bash
# =============================================================================
# MotionStreamer — RTX 5070 Ti (sm_120 / CUDA 12.8) Setup Script
# =============================================================================
# Mirrors the approach used for DART (see DART-main/SETUP_SUMMARY.md).
# Key pitfalls carried over from DART:
#   1. setuptools<70  →  already pinned in environment_rtx5070ti.yml
#   2. CLIP must be cloned + installed with --no-build-isolation
#   3. pytorch3d must be built from source with --no-build-isolation
#   4. PyTorch needs the cu128 index for sm_120 support
#
# Run from the repo root (Frankenstein-backend/):
#   bash MotionStreamer/setup_rtx5070ti.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MS_DIR="${SCRIPT_DIR}"

echo "========================================================"
echo " MotionStreamer — RTX 5070 Ti Setup"
echo " MS_DIR  : ${MS_DIR}"
echo " REPO_ROOT: ${REPO_ROOT}"
echo "========================================================"

# ---------------------------------------------------------------------------
# 1. Create conda environment
# ---------------------------------------------------------------------------
echo ""
echo "[1/7] Creating conda environment 'MotionStreamer' ..."
if conda env list | grep -q "^MotionStreamer "; then
    echo "  -> Environment 'MotionStreamer' already exists, skipping creation."
    echo "     To recreate: conda env remove -n MotionStreamer && rerun this script."
else
    conda env create -f "${MS_DIR}/environment_rtx5070ti.yml"
    echo "  -> Environment created."
fi

# All subsequent pip commands run inside the new env
PIP="$(conda run -n MotionStreamer which pip)"
PYTHON="$(conda run -n MotionStreamer which python)"

# ---------------------------------------------------------------------------
# 2. Install PyTorch 2.10+ with CUDA 12.8 (sm_120 support)
# ---------------------------------------------------------------------------
echo ""
echo "[2/7] Installing PyTorch 2.10+ cu128 ..."
conda run -n MotionStreamer \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
echo "  -> PyTorch installed."

# Verify GPU visibility
echo "  -> Verifying CUDA availability:"
conda run -n MotionStreamer python -c \
    "import torch; print(f'     torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ---------------------------------------------------------------------------
# 3. Install CLIP — MUST use clone + --no-build-isolation
#    (pip install git+https://... fails; same pitfall as DART)
# ---------------------------------------------------------------------------
echo ""
echo "[3/7] Installing CLIP ..."
CLIP_DIR="/tmp/clip_repo"
if [ ! -d "${CLIP_DIR}" ]; then
    git clone https://github.com/openai/CLIP.git "${CLIP_DIR}"
fi
conda run -n MotionStreamer \
    pip install -e "${CLIP_DIR}" --no-build-isolation
echo "  -> CLIP installed."

# ---------------------------------------------------------------------------
# 4. Build pytorch3d from source — MUST use --no-build-isolation
#    (same pitfall as DART; use latest pytorch3d compatible with PyTorch 2.x)
# ---------------------------------------------------------------------------
echo ""
echo "[4/7] Building pytorch3d from source ..."
PYTORCH3D_DIR="/tmp/pytorch3d"
if [ ! -d "${PYTORCH3D_DIR}" ]; then
    git clone https://github.com/facebookresearch/pytorch3d.git "${PYTORCH3D_DIR}"
fi
# Export CUDA arch for RTX 5070 Ti (Blackwell, sm_120)
export TORCH_CUDA_ARCH_LIST="12.0"
conda run -n MotionStreamer \
    bash -c "cd ${PYTORCH3D_DIR} && pip install -e . --no-build-isolation"
echo "  -> pytorch3d installed."

# ---------------------------------------------------------------------------
# 5. Download model checkpoints from HuggingFace
# ---------------------------------------------------------------------------
echo ""
echo "[5/7] Downloading model checkpoints ..."
cd "${MS_DIR}"

echo "  -> Causal TAE checkpoint ..."
conda run -n MotionStreamer \
    python humanml3d_272/prepare/download_Causal_TAE_t2m_272_ckpt.py

echo "  -> t2m_model (Transformer) checkpoint ..."
conda run -n MotionStreamer \
    python humanml3d_272/prepare/download_t2m_model_ckpt.py

echo "  -> Checkpoints downloaded."

# ---------------------------------------------------------------------------
# 6. Download sentence-T5-XXL text encoder
# ---------------------------------------------------------------------------
echo ""
echo "[6/7] Downloading sentence-T5-XXL text encoder (~10 GB) ..."
cd "${MS_DIR}"
conda run -n MotionStreamer \
    huggingface-cli download \
        --resume-download \
        sentence-transformers/sentence-t5-xxl \
        --local-dir sentencet5-xxl/
echo "  -> sentence-T5-XXL downloaded to MotionStreamer/sentencet5-xxl/"

# ---------------------------------------------------------------------------
# 7. Verify installation
# ---------------------------------------------------------------------------
echo ""
echo "[7/7] Verifying installation ..."
conda run -n MotionStreamer python -c "
import torch, numpy, clip, pytorch3d, sentence_transformers
print(f'  torch     : {torch.__version__}')
print(f'  numpy     : {numpy.__version__}')
print(f'  pytorch3d : {pytorch3d.__version__}')
print(f'  CLIP      : {clip.__version__}')
print(f'  sentence-transformers: {sentence_transformers.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "========================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. conda activate MotionStreamer"
echo "   2. Update .env in repo root (uncomment MS_* variables)"
echo "   3. cd <repo-root> && uvicorn app.main:app --reload"
echo "========================================================"
