"""
CPU-specific optimizations for faster inference on CPU-only systems.
"""

import torch
import os
import multiprocessing

def optimize_for_cpu():
    """Apply CPU-specific optimizations."""
    
    # Set optimal number of threads
    num_cores = multiprocessing.cpu_count()
    torch.set_num_threads(num_cores)
    
    # Enable CPU optimizations
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
    
    # Enable Intel MKL optimizations if available
    try:
        import intel_extension_for_pytorch as ipex
        return True
    except ImportError:
        pass
    
    # Enable oneDNN optimizations
    try:
        torch.backends.mkldnn.enabled = True
    except:
        pass
    
    # Set memory allocation strategy
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    return False

def get_cpu_optimized_dtype():
    """Get the best dtype for CPU inference."""
    # CPU works better with float32, but we can try bfloat16 for newer CPUs
    if hasattr(torch, 'bfloat16'):
        return torch.bfloat16
    return torch.float32

# Auto-apply optimizations when imported
if not torch.cuda.is_available():
    optimize_for_cpu()