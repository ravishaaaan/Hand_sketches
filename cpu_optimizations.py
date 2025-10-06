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

def apply_scheduler_optimizations(scheduler, inference_steps=50):
    """Apply scheduler-specific optimizations for 50 steps."""
    try:
        # Configure scheduler for maximum speed
        if hasattr(scheduler, 'set_timesteps'):
            # Use custom timestep spacing for faster convergence
            scheduler.set_timesteps(inference_steps)
            
        # Scheduler-specific optimizations
        scheduler_name = scheduler.__class__.__name__
        
        if "DPMSolver" in scheduler_name:
            # Optimize DPM solver for 50 steps
            if hasattr(scheduler, 'solver_order'):
                scheduler.solver_order = 2  # Optimal for 50 steps
            if hasattr(scheduler, 'lower_order_final'):
                scheduler.lower_order_final = True
                
        elif "DDIM" in scheduler_name:
            # Optimize DDIM for 50 steps
            if hasattr(scheduler, 'clip_sample'):
                scheduler.clip_sample = False
            if hasattr(scheduler, 'set_alpha_to_one'):
                scheduler.set_alpha_to_one = False
                
        print(f"✅ {scheduler_name} optimized for {inference_steps} steps")
        
    except Exception as e:
        print(f"⚠️ Scheduler optimization warning: {e}")

# Auto-apply optimizations when imported
if not torch.cuda.is_available():
    optimize_for_cpu()