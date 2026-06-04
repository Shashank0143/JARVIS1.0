"""GPU and CUDA configuration utilities for JARVIS."""

from __future__ import annotations

import os
from typing import Optional

try:
    import torch
except ImportError:
    torch = None


def initialize_cuda() -> bool:
    """
    Initialize CUDA and configure GPU settings.
    
    Returns:
        bool: True if CUDA is available and initialized, False otherwise.
    """
    if torch is None:
        print("Warning: PyTorch not installed. GPU support unavailable.")
        return False
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Info: CUDA is not available. Using CPU for training.")
        return False
    
    # Enable CUDA
    try:
        # Set environment variables for optimal GPU performance
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        # Get GPU information
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        
        print(f"✓ CUDA initialized successfully")
        print(f"  - CUDA Version: {cuda_version}")
        print(f"  - cuDNN Version: {cudnn_version}")
        print(f"  - GPU(s) Available: {device_count}")
        print(f"  - Primary GPU: {device_name}")
        
        # Configure cuDNN
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
        
        # Enable TF32 for better performance on Ampere GPUs (RTX 30-series, A100, etc.)
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        
        return True
    
    except Exception as exc:
        print(f"Warning: Failed to initialize CUDA: {exc}")
        return False


def get_device() -> torch.device:
    """
    Get the appropriate device (GPU or CPU) for PyTorch operations.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU device.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required but not installed.")
    
    if torch.cuda.is_available():
        # Force CUDA device (fixes privateuseone:0 issues)
        device = torch.device("cuda:0")
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
        return device
    return torch.device("cpu")


def get_device_string() -> str:
    """
    Get device string for logging and debugging.
    
    Returns:
        str: Device string ("cuda" or "cpu").
    """
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_gpu_memory(label: str = "GPU Memory") -> None:
    """
    Print current GPU memory usage.
    
    Args:
        label: Label for the memory report.
    """
    if torch is None or not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"{label}:")
    print(f"  - Allocated: {allocated:.2f}GB / {total:.2f}GB")
    print(f"  - Reserved: {reserved:.2f}GB / {total:.2f}GB")


def clear_gpu_cache() -> None:
    """Clear GPU cache to free up memory."""
    if torch is None or not torch.cuda.is_available():
        return
    
    torch.cuda.empty_cache()
    print("✓ GPU cache cleared")


def enable_mixed_precision() -> Optional[object]:
    """
    Enable automatic mixed precision (AMP) for faster training.
    
    Returns:
        torch.cuda.amp.autocast context manager if CUDA available, None otherwise.
    """
    if torch is None or not torch.cuda.is_available():
        return None
    
    try:
        # Enable TF32 on Ampere GPUs (RTX 30-series) for faster operations
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Return autocast for mixed precision training
        return torch.amp.autocast( "cuda",dtype=torch.float16)
    except Exception as exc:
        print(f"Warning: Could not enable mixed precision: {exc}")
        return None


def get_mixed_precision_scaler() -> Optional[object]:
    """
    Get GradScaler for mixed precision training on RTX 3050.
    
    Returns:
        torch.cuda.amp.GradScaler if CUDA available, None otherwise.
    """
    if torch is None or not torch.cuda.is_available():
        return None
    
    try:
        from torch.amp import GradScaler
        return GradScaler("cuda")
    except Exception as exc:
        print(f"Warning: Could not create GradScaler: {exc}")
        return None


def is_rtx_3050() -> bool:
    """
    Check if GPU is RTX 3050 (Ampere architecture).
    
    Returns:
        bool: True if RTX 3050 detected, False otherwise.
    """
    if torch is None or not torch.cuda.is_available():
        return False
    
    try:
        device_name = torch.cuda.get_device_name(0).lower()
        return "rtx 3050" in device_name or "3050" in device_name
    except Exception:
        return False


def get_optimal_batch_size() -> int:
    """
    Get optimal batch size for your GPU (RTX 3050 Laptop = 4-8).
    
    Returns:
        int: Recommended batch size for training.
    """
    if not torch.cuda.is_available():
        return 4
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # RTX 3050 Laptop has ~4GB VRAM
        if total_memory < 5:
            return 4  # RTX 3050 Laptop
        elif total_memory < 8:
            return 8
        elif total_memory < 12:
            return 16
        else:
            return 32
    except Exception:
        return 4


def get_gpu_info() -> dict:
    """
    Get detailed GPU information.
    
    Returns:
        dict: Dictionary containing GPU specifications and status.
    """
    if torch is None:
        return {"available": False, "reason": "PyTorch not installed"}
    
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA not available"}
    
    try:
        device_count = torch.cuda.device_count()
        devices = []
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "id": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1e9,
                "allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                "reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
            })
        
        return {
            "available": True,
            "device_count": device_count,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "devices": devices,
        }
    
    except Exception as exc:
        return {"available": False, "error": str(exc)}
