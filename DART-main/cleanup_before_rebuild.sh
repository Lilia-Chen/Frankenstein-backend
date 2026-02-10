#!/bin/bash
# Cleanup script before rebuilding DART environment

echo "=========================================="
echo "Cleanup Before Rebuilding DART Environment"
echo "=========================================="
echo ""

# 1. Clean conda cache
echo "1. Cleaning conda cache..."
conda clean --all -y

# 2. Clean pip cache
echo ""
echo "2. Cleaning pip cache..."
pip cache purge 2>/dev/null || echo "  (pip cache already clean or not accessible)"

# 3. Clean pytorch3d build artifacts
echo ""
echo "3. Cleaning pytorch3d build artifacts..."
if [ -d "/home/lilia_chen/workspace/pytorch3d/build" ]; then
    rm -rf /home/lilia_chen/workspace/pytorch3d/build
    echo "  Removed pytorch3d/build"
fi

if [ -d "/home/lilia_chen/workspace/pytorch3d/pytorch3d.egg-info" ]; then
    rm -rf /home/lilia_chen/workspace/pytorch3d/pytorch3d.egg-info
    echo "  Removed pytorch3d.egg-info"
fi

if [ -d "/home/lilia_chen/workspace/pytorch3d/dist" ]; then
    rm -rf /home/lilia_chen/workspace/pytorch3d/dist
    echo "  Removed pytorch3d/dist"
fi

# 4. Remove any __pycache__ directories
echo ""
echo "4. Cleaning Python cache files..."
find /home/lilia_chen/workspace/pytorch3d -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /home/lilia_chen/workspace/pytorch3d -type f -name "*.pyc" -delete 2>/dev/null || true
find /home/lilia_chen/workspace/pytorch3d -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  Removed __pycache__ and .pyc files"

# 5. Check disk space
echo ""
echo "5. Current disk space:"
df -h /home | tail -1

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Ready to create new environment with:"
echo "  conda env create -f environment_py310.yml"
