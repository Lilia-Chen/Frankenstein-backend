#!/bin/bash
# Setup script for DART environment with RTX 5070 Ti (sm_120) support
# This script handles the special requirements for Blackwell architecture GPUs

set -e  # Exit on error

echo "=========================================="
echo "DART Environment Setup for RTX 5070 Ti"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Creating DART conda environment...${NC}"
conda env create -f environment.yml

echo ""
echo -e "${GREEN}Step 2: Activating DART environment...${NC}"
# Note: We need to use conda run instead of activate in script
eval "$(conda shell.bash hook)"
conda activate DART

echo ""
echo -e "${YELLOW}Step 3: Installing PyTorch Nightly with CUDA 12.9...${NC}"
echo "This is required for RTX 5070 Ti (sm_120) support"
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

echo ""
echo -e "${YELLOW}Step 4: Building pytorch3d from source...${NC}"
PYTORCH3D_DIR="/home/lilia_chen/workspace/pytorch3d"

if [ ! -d "$PYTORCH3D_DIR" ]; then
    echo -e "${RED}Error: pytorch3d source not found at $PYTORCH3D_DIR${NC}"
    exit 1
fi

cd "$PYTORCH3D_DIR"
echo "Building pytorch3d (this may take several minutes)..."
pip install -e .

echo ""
echo -e "${GREEN}Step 5: Verifying installation...${NC}"
cd /home/lilia_chen/workspace/Frankenstein-backend/DART-main
python verify_installation.py

echo ""
echo -e "${GREEN}=========================================="
echo "Setup complete!"
echo "==========================================${NC}"
echo ""
echo "To activate the environment, run:"
echo "  conda activate DART"
echo ""
echo -e "${YELLOW}IMPORTANT NOTES:${NC}"
echo "1. PyTorch nightly builds may have stability issues"
echo "2. If you encounter 'no kernel image available' errors, the nightly build"
echo "   may not have full sm_120 support compiled yet"
echo "3. Monitor https://github.com/pytorch/pytorch/issues/164342 for updates"
echo "   on stable PyTorch support for RTX 50-series GPUs"
echo ""
