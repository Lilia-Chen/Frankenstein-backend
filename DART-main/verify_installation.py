#!/usr/bin/env python
"""
Verification script for DART environment on RTX 5070 Ti
Tests PyTorch, CUDA, pytorch3d, and basic operations
"""

import sys
from typing import Tuple

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def check_pytorch() -> Tuple[bool, str]:
    """Check PyTorch installation"""
    try:
        import torch
        version = torch.__version__

        # Check if it's a nightly/dev version
        is_nightly = 'dev' in version or '+' in version

        return True, version, is_nightly
    except ImportError as e:
        return False, str(e), False

def check_cuda(torch) -> Tuple[bool, dict]:
    """Check CUDA availability and details"""
    if not torch.cuda.is_available():
        return False, {}

    info = {
        'cuda_version': torch.version.cuda,
        'device_count': torch.cuda.device_count(),
        'device_name': torch.cuda.get_device_name(0),
        'capability': torch.cuda.get_device_capability(0),
        'arch_list': torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else [],
    }

    return True, info

def check_pytorch3d() -> Tuple[bool, str]:
    """Check pytorch3d installation"""
    try:
        import pytorch3d
        version = pytorch3d.__version__
        return True, version
    except ImportError as e:
        return False, str(e)

def test_basic_operations(torch) -> Tuple[bool, str]:
    """Test basic PyTorch operations on GPU"""
    try:
        # Create a tensor on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')

        # Perform matrix multiplication
        z = torch.matmul(x, y)

        # Check result
        if z.shape == (1000, 1000) and z.device.type == 'cuda':
            return True, "Matrix multiplication successful"
        else:
            return False, "Unexpected result shape or device"

    except Exception as e:
        return False, str(e)

def test_pytorch3d_operations() -> Tuple[bool, str]:
    """Test basic pytorch3d operations"""
    try:
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex
        import torch

        # Create a simple triangle mesh
        verts = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], device='cuda')

        faces = torch.tensor([[0, 1, 2]], device='cuda')

        # Create mesh
        mesh = Meshes(verts=[verts], faces=[faces])

        if mesh.verts_packed().shape[0] == 3:
            return True, "Created mesh with 3 vertices on GPU"
        else:
            return False, "Unexpected mesh structure"

    except Exception as e:
        return False, str(e)

def main():
    print_header("DART Environment Verification for RTX 5070 Ti")

    all_passed = True

    # Check PyTorch
    print(f"{Colors.BOLD}1. PyTorch Installation{Colors.END}")
    pytorch_ok, pytorch_version, is_nightly = check_pytorch()

    if pytorch_ok:
        print_success(f"PyTorch installed: version {pytorch_version}")
        if is_nightly:
            print_success("Using nightly/development build (required for sm_120)")
        else:
            print_warning("Using stable build - may not support sm_120!")

        import torch

        # Check CUDA
        print(f"\n{Colors.BOLD}2. CUDA Support{Colors.END}")
        cuda_ok, cuda_info = check_cuda(torch)

        if cuda_ok:
            print_success(f"CUDA available: version {cuda_info['cuda_version']}")
            print_success(f"Device count: {cuda_info['device_count']}")
            print_success(f"Device name: {cuda_info['device_name']}")

            capability = cuda_info['capability']
            print_success(f"Compute capability: sm_{capability[0]}{capability[1]}")

            # Check if sm_120 is detected
            if capability == (12, 0):
                print_success("✓ RTX 5070 Ti (sm_120) detected!")
            else:
                print_warning(f"Note: Detected sm_{capability[0]}{capability[1]}, not sm_120")

            if cuda_info['arch_list']:
                print(f"  Supported architectures: {', '.join(cuda_info['arch_list'])}")

                # Check for sm_120 or sm_90 in arch list (sm_90 can sometimes run sm_120)
                has_sm120 = any('120' in arch or '90' in arch for arch in cuda_info['arch_list'])
                if has_sm120:
                    print_success("Architecture list includes sm_90/sm_120 support")
                else:
                    print_warning("sm_120 not found in architecture list - may encounter 'no kernel image' errors")
        else:
            print_error("CUDA not available!")
            all_passed = False

        # Test basic operations
        if cuda_ok:
            print(f"\n{Colors.BOLD}3. Basic GPU Operations{Colors.END}")
            ops_ok, ops_msg = test_basic_operations(torch)

            if ops_ok:
                print_success(ops_msg)
            else:
                print_error(f"GPU operations failed: {ops_msg}")
                all_passed = False
    else:
        print_error(f"PyTorch not installed: {pytorch_version}")
        all_passed = False
        print("\nSkipping further tests due to PyTorch import failure")
        sys.exit(1)

    # Check pytorch3d
    print(f"\n{Colors.BOLD}4. pytorch3d Installation{Colors.END}")
    p3d_ok, p3d_version = check_pytorch3d()

    if p3d_ok:
        print_success(f"pytorch3d installed: version {p3d_version}")

        # Test pytorch3d operations
        print(f"\n{Colors.BOLD}5. pytorch3d GPU Operations{Colors.END}")
        p3d_ops_ok, p3d_ops_msg = test_pytorch3d_operations()

        if p3d_ops_ok:
            print_success(p3d_ops_msg)
        else:
            print_error(f"pytorch3d operations failed: {p3d_ops_msg}")
            print_warning("This might be a kernel compilation issue with sm_120")
            all_passed = False
    else:
        print_error(f"pytorch3d not installed: {p3d_version}")
        all_passed = False

    # Summary
    print_header("Verification Summary")

    if all_passed:
        print_success(f"{Colors.BOLD}All checks passed! Environment is ready.{Colors.END}")
        return 0
    else:
        print_error(f"{Colors.BOLD}Some checks failed. See details above.{Colors.END}")
        print(f"\n{Colors.YELLOW}Common issues:{Colors.END}")
        print("  • 'no kernel image available' - PyTorch nightly may not have full sm_120 support yet")
        print("  • Missing CUDA - Install CUDA 12.8+ on your system")
        print("  • pytorch3d import fails - Rebuild from source with current PyTorch")
        return 1

if __name__ == "__main__":
    sys.exit(main())
