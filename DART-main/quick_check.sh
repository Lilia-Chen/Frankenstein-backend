#!/bin/bash
# Quick verification script for DART environment
# Run this anytime to check if your environment is working

echo "Quick DART Environment Check"
echo "============================="
echo ""

# Check if DART environment is active
if [[ "$CONDA_DEFAULT_ENV" != "DART" ]]; then
    echo "⚠ DART conda environment is not active"
    echo "Run: conda activate DART"
    exit 1
fi

# Run Python checks
python << 'EOF'
import sys

def check(name, func):
    try:
        result = func()
        print(f"✓ {name}: {result}")
        return True
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

# Check PyTorch
check("PyTorch", lambda: __import__('torch').__version__)

# Check CUDA
check("CUDA available", lambda: __import__('torch').cuda.is_available())

# Check GPU
def get_gpu():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU"
check("GPU", get_gpu)

# Check compute capability
def get_capability():
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        return f"sm_{cap[0]}{cap[1]}"
    return "N/A"
check("Compute capability", get_capability)

# Check pytorch3d
check("pytorch3d", lambda: __import__('pytorch3d').__version__)

# Quick GPU test
def gpu_test():
    import torch
    x = torch.randn(100, 100, device='cuda')
    y = x @ x
    return "Matrix ops OK"
check("GPU operations", gpu_test)

EOF

echo ""
echo "For detailed verification, run: python verify_installation.py"
