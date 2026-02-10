#!/bin/bash
# Upgrade PyTorch to nightly build for RTX 5070 Ti sm_120 support

set -e

echo "=========================================="
echo "Upgrading PyTorch to Nightly Build"
echo "=========================================="
echo ""

# Check if DART environment is active
if [[ "$CONDA_DEFAULT_ENV" != "DART" ]]; then
    echo "⚠ Please activate DART environment first:"
    echo "  conda activate DART"
    exit 1
fi

echo "Current PyTorch version:"
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA: {torch.version.cuda}')"
echo ""

echo "Uninstalling old PyTorch packages..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Installing PyTorch Nightly with CUDA 12.4..."
echo "Note: Using cu124 as cu129 is not yet available"
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

echo ""
echo "New PyTorch version:"
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA: {torch.version.cuda}')"
python -c "import torch; print(f'  Nightly: {\"dev\" in torch.__version__ or \"+\" in torch.__version__}')"

echo ""
echo "Verifying CUDA and sm_120 support..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'✓ Compute capability: sm_{cap[0]}{cap[1]}')
    archs = torch.cuda.get_arch_list()
    print(f'✓ Supported architectures: {archs}')

    # Test basic operation
    try:
        x = torch.randn(100, 100, device='cuda')
        y = x @ x
        print('✓ Basic GPU operations work!')
    except Exception as e:
        print(f'✗ GPU operation failed: {e}')
else:
    print('✗ CUDA not available!')
"

echo ""
echo "=========================================="
echo "PyTorch Nightly installation complete!"
echo "=========================================="
echo ""
echo "⚠ IMPORTANT: pytorch3d was built with old PyTorch 2.1.0"
echo "You may need to rebuild pytorch3d:"
echo ""
echo "  cd /home/lilia_chen/workspace/pytorch3d"
echo "  pip install -e . --force-reinstall --no-deps"
echo ""
echo "Then run: python verify_installation.py"
