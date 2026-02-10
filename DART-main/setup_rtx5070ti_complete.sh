#!/bin/bash
# Complete setup script for DART on RTX 5070 Ti (sm_120)
# This script automates the entire installation process
# Verified working on Ubuntu 22.04 / WSL2

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "DART Setup for RTX 5070 Ti (sm_120)"
echo "==========================================${NC}"
echo ""

# ============================================================================
# Step 1: Create conda environment
# ============================================================================
echo -e "${GREEN}Step 1: Creating conda environment...${NC}"
if conda env list | grep -q "^DART "; then
    echo -e "${YELLOW}DART environment already exists. Removing...${NC}"
    conda env remove -n DART -y
fi

conda env create -f environment_py310_verified.yml
echo -e "${GREEN}✓ Conda environment created${NC}"
echo ""

# ============================================================================
# Step 2: Activate environment and install PyTorch
# ============================================================================
echo -e "${GREEN}Step 2: Installing PyTorch 2.10+ with CUDA 12.8...${NC}"
conda run -n DART pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
conda run -n DART python << 'EOF'
import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"✓ Compute capability: sm_{cap[0]}{cap[1]}")
EOF

echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# ============================================================================
# Step 3: Install CLIP
# ============================================================================
echo -e "${GREEN}Step 3: Installing CLIP...${NC}"

# Clone CLIP if not exists
if [ ! -d "/home/lilia_chen/workspace/clip_repo" ]; then
    cd /home/lilia_chen/workspace
    git clone https://github.com/openai/CLIP.git clip_repo
fi

cd /home/lilia_chen/workspace/clip_repo
conda run -n DART pip install -e . --no-build-isolation

# Verify CLIP installation
conda run -n DART python -c "import clip; print('✓ CLIP installed')"
echo -e "${GREEN}✓ CLIP installed${NC}"
echo ""

# ============================================================================
# Step 4: Build pytorch3d from source
# ============================================================================
echo -e "${GREEN}Step 4: Building pytorch3d from source...${NC}"
echo -e "${YELLOW}This will take 10-20 minutes. Please be patient...${NC}"

cd /home/lilia_chen/workspace/pytorch3d
conda run -n DART pip install -e . --no-build-isolation

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ pytorch3d build failed${NC}"
    exit 1
fi

# Verify pytorch3d installation
conda run -n DART python << 'EOF'
import pytorch3d
print(f"✓ pytorch3d version: {pytorch3d.__version__}")
EOF

echo -e "${GREEN}✓ pytorch3d built successfully${NC}"
echo ""

# ============================================================================
# Step 5: Final verification
# ============================================================================
echo -e "${GREEN}Step 5: Running final verification...${NC}"
cd /home/lilia_chen/workspace/Frankenstein-backend/DART-main
conda run -n DART python verify_installation.py

echo ""
echo -e "${GREEN}=========================================="
echo "Installation Complete!"
echo "==========================================${NC}"
echo ""
echo "To activate the environment:"
echo "  conda activate DART"
echo ""
echo "To run the interactive demo:"
echo "  cd /home/lilia_chen/workspace/Frankenstein-backend/DART-main"
echo "  source ./demos/run_demo.sh"
echo ""
echo "To run headless demo:"
echo "  source ./demos/rollout.sh"
echo ""
echo -e "${GREEN}Verified working on RTX 5070 Ti (sm_120)${NC}"
echo ""
