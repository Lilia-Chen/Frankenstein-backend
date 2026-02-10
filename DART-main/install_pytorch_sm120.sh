#!/bin/bash
# Install PyTorch with sm_120 support for RTX 5070 Ti
# Tries multiple installation methods

set -e

echo "=========================================="
echo "Installing PyTorch for RTX 5070 Ti"
echo "=========================================="
echo ""

# Check if DART environment is active
if [[ "$CONDA_DEFAULT_ENV" != "DART" ]]; then
    echo "⚠ Please activate DART environment first:"
    echo "  conda activate DART"
    exit 1
fi

echo "Current PyTorch version (if installed):"
python -c "import torch; print(f'  torch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}')" 2>/dev/null || echo "  Not installed"
echo ""

echo "Uninstalling old PyTorch packages..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || echo "No existing PyTorch installation"

echo ""
echo "=========================================="
echo "Attempting installation methods..."
echo "=========================================="

# Method 1: Try stable PyTorch 2.9+ with CUDA 12.8
echo ""
echo "Method 1: PyTorch stable 2.9+ with CUDA 12.8"
echo "This may have sm_120 support in recent releases"
echo ""

if pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee /tmp/pytorch_install.log; then
    echo "✓ Installation successful!"

    echo ""
    echo "Checking PyTorch version and CUDA support..."
    python << 'EOF'
import torch
version = torch.__version__
cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'

print(f"✓ PyTorch version: {version}")
print(f"✓ CUDA version: {cuda_version}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"✓ Compute capability: sm_{cap[0]}{cap[1]}")

    # Test basic operation
    try:
        x = torch.randn(100, 100, device='cuda')
        y = x @ x
        print("✓ Basic GPU operations work!")
    except Exception as e:
        print(f"✗ GPU operation failed: {e}")
        print("⚠ May need to try nightly builds")
EOF

    INSTALL_SUCCESS=true
else
    echo "✗ Method 1 failed"
    INSTALL_SUCCESS=false
fi

# Method 2: Try nightly if stable failed
if [ "$INSTALL_SUCCESS" = false ]; then
    echo ""
    echo "=========================================="
    echo "Method 2: PyTorch nightly with CUDA 12.4"
    echo "=========================================="

    if pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124; then
        echo "✓ Nightly installation successful!"
        INSTALL_SUCCESS=true
    else
        echo "✗ Method 2 failed"
    fi
fi

# Method 3: Try cu121 nightly
if [ "$INSTALL_SUCCESS" = false ]; then
    echo ""
    echo "=========================================="
    echo "Method 3: PyTorch nightly with CUDA 12.1"
    echo "=========================================="

    if pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121; then
        echo "✓ Nightly installation successful!"
        INSTALL_SUCCESS=true
    else
        echo "✗ Method 3 failed"
    fi
fi

# Final check
echo ""
echo "=========================================="
echo "Final Verification"
echo "=========================================="

python << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: sm_{cap[0]}{cap[1]}")

    # Check architecture support
    if hasattr(torch.cuda, 'get_arch_list'):
        archs = torch.cuda.get_arch_list()
        print(f"Supported architectures: {archs}")

        # Check for sm_120 or sm_90 (fallback)
        has_support = any('120' in str(arch) or '90' in str(arch) for arch in archs)
        if has_support:
            print("✓ Architecture includes sm_90/sm_120")
        else:
            print("⚠ sm_120 not in architecture list - may have limited support")

    # Test operation
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = x @ x
        print("✓ GPU operations successful!")
    except Exception as e:
        print(f"✗ GPU operation failed: {e}")
        sys.exit(1)
else:
    print("✗ CUDA not available!")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ PyTorch installation complete!"
    echo "=========================================="
    echo ""
    echo "⚠ pytorch3d may need to be rebuilt:"
    echo "  cd /home/lilia_chen/workspace/pytorch3d"
    echo "  pip install -e . --force-reinstall --no-deps"
    echo ""
    echo "Then verify with: python verify_installation.py"
else
    echo ""
    echo "=========================================="
    echo "✗ Installation failed"
    echo "=========================================="
    echo ""
    echo "Please try manual installation or check PyTorch forums:"
    echo "  https://pytorch.org/get-started/locally/"
    exit 1
fi
