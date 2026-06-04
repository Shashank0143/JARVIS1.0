#!/usr/bin/env python
"""GPU diagnostics and setup verification script for JARVIS."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    print("\n" + "="*70)
    print("JARVIS GPU/CUDA Diagnostic Tool")
    print("="*70 + "\n")

    # Check PyTorch installation
    print("1. Checking PyTorch installation...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__} installed")
    except ImportError:
        print("   ✗ PyTorch not installed!")
        print("   Install with: pip install torch")
        return

    # Check CUDA availability
    print("\n2. Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"   ✓ CUDA is available")
    else:
        print(f"   ✗ CUDA is not available")
        print("   Your system will use CPU for training (slower)")

    # Check CUDA version
    if cuda_available:
        print("\n3. CUDA Details:")
        print(f"   - CUDA Version: {torch.version.cuda}")
        print(f"   - cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   - cuDNN Enabled: {torch.backends.cudnn.enabled}")
        print(f"   - Benchmark Mode: {torch.backends.cudnn.benchmark}")

        # Check GPU devices
        print("\n4. GPU Device(s):")
        device_count = torch.cuda.device_count()
        print(f"   - Number of GPUs: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = props.total_memory / 1e9

            print(f"\n   GPU {i}: {props.name}")
            print(f"   - Compute Capability: {props.major}.{props.minor}")
            print(f"   - Total Memory: {total:.2f}GB")
            print(f"   - Allocated: {allocated:.2f}GB")
            print(f"   - Reserved: {reserved:.2f}GB")
            print(f"   - Free: {total - reserved:.2f}GB")

        # Test GPU computation
        print("\n5. Testing GPU computation...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            result = y.sum().item()
            print(f"   ✓ GPU computation successful (result: {result:.2f})")
        except Exception as exc:
            print(f"   ✗ GPU computation failed: {exc}")

    else:
        print("\n3. CPU Information:")
        print("   Training will use CPU (slower than GPU)")

    # Check JARVIS GPU utilities
    print("\n6. Checking JARVIS GPU utilities...")
    try:
        from jarvis_ai import gpu
        from jarvis_ai.gpu import get_gpu_info, is_rtx_3050, get_optimal_batch_size
        
        info = get_gpu_info()
        print(f"   ✓ JARVIS GPU module loaded")
        print(f"   - GPU Available: {info.get('available')}")
        
        if is_rtx_3050():
            print(f"   ✓ RTX 3050 Detected!")
            print(f"   - Mixed Precision: FP16 (Enabled)")
            print(f"   - Recommended Batch Size: {get_optimal_batch_size()}")
            print(f"   - Expected Speedup: 2-3x with FP16")
        
        if info.get('available'):
            devices = info.get('devices', [])
            for device in devices:
                print(f"     • {device['name']} ({device['total_memory_gb']:.1f}GB)")
    except ImportError as exc:
        print(f"   ✗ Could not load JARVIS GPU module: {exc}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR YOUR SYSTEM:")
    print("="*70)

    if not cuda_available:
        print("\n⚠ GPU NOT DETECTED - To enable GPU acceleration:")
        print("  1. Install NVIDIA GPU driver (latest version)")
        print("  2. Reinstall PyTorch with CUDA support:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("  3. Verify with: nvidia-smi")
    else:
        from jarvis_ai import is_rtx_3050, get_optimal_batch_size
        
        is_3050 = is_rtx_3050()
        device_count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1e9
        
        if is_3050:
            print(f"\n✓ RTX 3050 Laptop GPU Detected!")
            print(f"  GPU Memory: {total_memory:.1f}GB")
            print(f"\nOptimizations Enabled:")
            print(f"  • FP16 Mixed Precision Training (2-3x faster)")
            print(f"  • Automatic Batch Size Optimization")
            print(f"  • cuDNN TF32 Acceleration")
            print(f"\nRecommended Training Settings:")
            print(f"  - Batch Size: {get_optimal_batch_size()}")
            print(f"  - Learning Rate: 3e-4 (default)")
            print(f"  - Command: python main.py train-transformer --epochs 5")
            print(f"\nExpected Performance:")
            print(f"  - Training Time: ~30-45s per epoch")
            print(f"  - Memory Usage: ~3-4GB (safe for RTX 3050)")
        elif total_memory < 4:
            print(f"\n⚠ Limited GPU memory ({total_memory:.1f}GB)")
            print("  Recommendations:")
            print("  - Use smaller batch sizes: --batch-size 4")
            print("  - Reduce model size")
        elif total_memory < 8:
            print(f"\n✓ GPU memory adequate ({total_memory:.1f}GB)")
            print("  Recommended:")
            print("  - Batch size: 8-12")
            print("  - Good for transformer training")
        else:
            print(f"\n✓ Excellent GPU memory ({total_memory:.1f}GB)")
            print("  Recommended:")
            print("  - Batch size: 16-32+")
            print("  - Can handle larger models")

        print("\n✓ Your system is ready for GPU-accelerated training!")
        print("  Run JARVIS training with:")
        print("  python main.py train-transformer --epochs 5")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
