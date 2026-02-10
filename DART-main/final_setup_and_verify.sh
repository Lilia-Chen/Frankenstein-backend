#!/bin/bash
# Final setup and verification for DART with RTX 5070 Ti

echo "=========================================="
echo "Final Setup & Verification for RTX 5070 Ti"
echo "=========================================="
echo ""

# Check if DART environment is active
if [[ "$CONDA_DEFAULT_ENV" != "DART" ]]; then
    echo "❌ Please activate DART environment first:"
    echo "   conda activate DART"
    echo "   bash $0"
    exit 1
fi

echo "✓ DART environment active"
echo ""

# Step 1: Build pytorch3d
echo "Step 1: Building pytorch3d from source..."
echo "This will take 10-20 minutes..."
cd /home/lilia_chen/workspace/pytorch3d
pip install -e . --no-build-isolation

if [ $? -ne 0 ]; then
    echo "❌ pytorch3d build failed"
    exit 1
fi

echo ""
echo "✓ pytorch3d built successfully"
echo ""

# Step 2: Quick verification
echo "Step 2: Quick verification..."
python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: sm_{cap[0]}{cap[1]}")
EOF

echo ""

# Step 3: Full verification
echo "Step 3: Running full verification..."
cd /home/lilia_chen/workspace/Frankenstein-backend/DART-main
python verify_installation.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
