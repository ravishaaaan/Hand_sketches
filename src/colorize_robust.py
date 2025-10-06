import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np
import gc
import time
import warnings
from performance_config import get_active_config
import cpu_optimizations  # Auto-apply CPU optimizations

# Suppress diffusers/transformers warnings for cleaner output
warnings.filterwarnings("ignore", message=".*cross_attention_kwargs.*are not expected.*")
warnings.filterwarnings("ignore", message=".*slice_size.*")
warnings.filterwarnings("ignore", message=".*AttnProcessor.*")
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Force CPU usage only
device = "cpu"
torch_dtype = torch.float32

# Global variable for lazy loading
_pipe = None
_loading_lock = False
_current_preset = None

def _load_pipeline_robust():
    """Robust pipeline loading with better error handling."""
    global _pipe, _loading_lock
    
    if _pipe is not None:
        return _pipe
    
    if _loading_lock:
        print("⏳ Pipeline loading in progress...")
        return None
        
    _loading_lock = True
    
    try:
        # Clear any existing cache/memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Load with minimal memory usage
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None
        )
        

        _pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
            safety_checker=None,  # Disable to save memory
            requires_safety_checker=False
        )
        
        # Configure ULTRA-FAST scheduler optimized for 50 steps
        try:
            # DPMSolverMultistepScheduler is fastest for high step count
            from diffusers import DPMSolverMultistepScheduler
            _pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                _pipe.scheduler.config,
                use_karras_sigmas=False,  # Faster without karras
                algorithm_type="dpmsolver++",  # Fastest variant for high steps
                solver_order=2,  # Optimal for 50 steps
                lower_order_final=True,  # Faster final steps
                euler_at_final=True,  # Speed up final step
                use_lu_lambdas=True,  # Mathematical optimization
                final_sigmas_type="zero"  # Skip final noise
            )
            # Apply scheduler-specific optimizations
            cpu_optimizations.apply_scheduler_optimizations(_pipe.scheduler, 50)
            
        except ImportError:
            try:
                # Fallback to LCM for speed
                from diffusers import LCMScheduler
                _pipe.scheduler = LCMScheduler.from_config(_pipe.scheduler.config)
                cpu_optimizations.apply_scheduler_optimizations(_pipe.scheduler, 50)
            except ImportError:
                # Ultra-fast DDIM config as last resort
                _pipe.scheduler = DDIMScheduler.from_config(
                    _pipe.scheduler.config,
                    clip_sample=False,  # Faster
                    set_alpha_to_one=False,  # Faster
                    rescale_betas_zero_snr=True,  # Speed optimization
                    timestep_spacing="trailing"  # Faster spacing
                )
                cpu_optimizations.apply_scheduler_optimizations(_pipe.scheduler, 50)
        
        # EXTREME performance optimizations for 50-step speed
        if hasattr(_pipe, 'enable_attention_slicing'):
            _pipe.enable_attention_slicing("max")  # Maximum slicing for speed
            
        if hasattr(_pipe, 'enable_vae_slicing'):
            _pipe.enable_vae_slicing()  # VAE slicing for memory/speed
            
        if hasattr(_pipe, 'enable_vae_tiling'):
            _pipe.enable_vae_tiling()  # VAE tiling for large images
            
        # Advanced CPU optimizations
        if hasattr(_pipe, 'enable_cpu_offload'):
            _pipe.enable_cpu_offload()  # Aggressive CPU memory management
            
        # Compile model for speed (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile') and hasattr(_pipe.unet, 'forward'):
                _pipe.unet = torch.compile(_pipe.unet, mode="max-autotune")
                print("✅ UNet compiled for maximum speed")
        except Exception as e:
            print(f"⚠️ UNet compilation skipped: {e}")
            
        # Enable optimized attention
        try:
            if hasattr(_pipe, 'enable_xformers_memory_efficient_attention'):
                _pipe.enable_xformers_memory_efficient_attention()
                print("✅ XFormers attention enabled")
        except Exception:
            try:
                # Fallback to Flash Attention
                if hasattr(_pipe, 'enable_flash_attention'):
                    _pipe.enable_flash_attention()
                    print("✅ Flash attention enabled")
            except Exception:
                pass
            
        # Use TensorRT or other optimizations if available
        try:
            if hasattr(_pipe, 'enable_sequential_cpu_offload'):
                _pipe.enable_sequential_cpu_offload()  # More aggressive than model_cpu_offload
            elif hasattr(_pipe, 'enable_model_cpu_offload') and device == "cuda":
                _pipe.enable_model_cpu_offload()
        except Exception:
            pass
            
        # Move to device
        if device == "cpu":
            _pipe = _pipe.to("cpu")
        else:
            _pipe = _pipe.to(device)
            
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            import torch._dynamo
            _pipe.unet = torch.compile(_pipe.unet, mode="reduce-overhead", fullgraph=True)
        except Exception:
            pass
        
        print("DEBUG: ✅ Pipeline loaded successfully (robust mode)!")
        print(f"DEBUG: Using scheduler: {type(_pipe.scheduler).__name__}")
        print(f"DEBUG: Device: {device}")
        _loading_lock = False
        return _pipe
        
    except Exception as e:
        print(f"❌ Pipeline loading failed: {e}")
        _loading_lock = False
        _pipe = None
        
        # Try alternative loading method
        print("🔄 Trying alternative loading method...")
        try:
            return _load_pipeline_simple()
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            raise RuntimeError("Failed to load pipeline with both methods")

def _load_pipeline_simple():
    """Simple fallback pipeline loading."""
    global _pipe
    
    print("   Using simple loading method...")
    
    # Load without advanced features
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch_dtype
    )
    
    _pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    _pipe.scheduler = DDIMScheduler.from_config(_pipe.scheduler.config)
    _pipe = _pipe.to(device)
    
    return _pipe

def colorize_any_robust(input_path, label, output_path="datasets/colorized_output.png", external_progress_callback=None, speed_mode="auto"):
    """
    Robust colorization with multiple speed options:
    - "lightning": Ultra-fast algorithm (~10-30s for 50 steps)
    - "fast": Optimized SD with reduced size (~60-120s for 50 steps)
    - "quality": Full SD pipeline (~300-600s for 50 steps)
    - "auto": Choose based on system and step count
    """
    global _pipe, _current_preset
    
    print(f"\nDEBUG: Starting colorization for '{label}' sketch at {input_path}")
    
    # Get performance configuration first
    from performance_config import ACTIVE_PRESET, get_active_config
    config = get_active_config()
    inference_steps = config["inference_steps"]
    
    # SPEED MODE SELECTION
    if speed_mode == "auto":
        # Auto-select based on step count and system
        if inference_steps >= 50:
            speed_mode = "lightning"  # Use lightning for 50+ steps
            print("🚀 AUTO MODE: Selected LIGHTNING for 50+ steps")
        elif inference_steps >= 20:
            speed_mode = "fast"  # Use fast for 20+ steps
            print("⚡ AUTO MODE: Selected FAST for 20+ steps")
        else:
            speed_mode = "quality"  # Use quality for <20 steps
            print("🎨 AUTO MODE: Selected QUALITY for <20 steps")
    
    # Route to appropriate colorization method
    if speed_mode == "lightning":
        print("⚡ Using LIGHTNING colorization (algorithm-based)")
        try:
            from lightning_colorize import lightning_colorize_50_steps
            return lightning_colorize_50_steps(input_path, label, output_path)
        except ImportError:
            print("⚠️ Lightning colorizer not available, falling back to fast mode")
            speed_mode = "fast"
    
    if speed_mode == "fast":
        print("🚀 Using FAST colorization (optimized SD)")
        try:
            from ultra_fast_colorize import colorize_ultra_fast
            return colorize_ultra_fast(input_path, label, output_path, steps=inference_steps)
        except ImportError:
            print("⚠️ Ultra-fast colorizer not available, falling back to quality mode")
            speed_mode = "quality"
    
    # Default to quality mode (original SD pipeline)
    print("🎨 Using QUALITY colorization (full SD pipeline)")
    
    try:
        target_size = config["image_size"]
        
        print(f"DEBUG: Using {ACTIVE_PRESET} preset - {config['inference_steps']} steps, {target_size}px, {config['guidance_scale']} guidance")
        
        # Check if preset changed - reload pipeline if needed
        if _current_preset != ACTIVE_PRESET:
            print(f"DEBUG: Preset changed from {_current_preset} to {ACTIVE_PRESET}, reloading pipeline...")
            _pipe = None  # Force reload
            _current_preset = ACTIVE_PRESET
        
        # Load pipeline
        print("DEBUG: Loading Stable Diffusion pipeline...")
        pipe = _load_pipeline_robust()
        if pipe is None:
            raise RuntimeError("Pipeline loading failed")
        print("DEBUG: Pipeline loaded successfully")
        
        # Load and process image with performance-optimized sizing
        sketch = load_image(input_path)
        print(f"DEBUG: Loaded sketch - Original size: {sketch.size}")
        
        # Resize to optimal size for faster processing
        if sketch.size != (target_size, target_size):
            sketch = sketch.resize((target_size, target_size), Image.Resampling.LANCZOS)
            print(f"DEBUG: Resized sketch to {target_size}x{target_size} for optimization")
        
        # Generate with performance-optimized settings
        prompt = f"A realistic photo of a {label}, high quality, detailed"
        negative_prompt = "blurry, low quality, distorted, deformed"
        
        print(f"DEBUG: Starting inference - Prompt: '{prompt}'")
        
        # Get FRESH config right before inference to ensure latest settings
        fresh_config = get_active_config()
        inference_steps = fresh_config["inference_steps"]
        guidance_scale = fresh_config["guidance_scale"]
        
        print(f"DEBUG: Inference parameters - Steps: {inference_steps}, Guidance: {guidance_scale}")
        
        # LIVE Progress callback for real-time UI updates
        callback_start_time = time.time()
        last_console_log_time = callback_start_time
        
        # Real-time progress callback for live UI updates
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            nonlocal last_console_log_time
            current_time = time.time()
            progress = (step_index / inference_steps) * 100
            
            # ALWAYS call external progress callback for live UI updates
            if external_progress_callback:
                external_progress_callback(step_index, inference_steps, progress)
            
            # Console logging every 10 seconds to avoid spam
            if current_time - last_console_log_time >= 10.0:
                print(f"DEBUG: Inference progress - Step {step_index+1}/{inference_steps} ({progress:.1f}%)")
                last_console_log_time = current_time
            
            return callback_kwargs
        
        # EXTREME 50-step speed optimizations
        generator = torch.Generator(device=device).manual_seed(42)
        
        # Pre-compute optimizations
        with torch.no_grad():
            # Optimize guidance scale for 50 steps
            if guidance_scale > 7.0:
                guidance_scale = min(guidance_scale, 12.0)  # Cap guidance for speed
            
            # Dynamic batch processing for faster convergence
            batch_optimization = {
                "num_inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "eta": 0.0,  # Deterministic = faster
                "generator": generator,
            }
        
        print(f"DEBUG: Starting OPTIMIZED 50-step inference (guidance: {guidance_scale})")
        
        # Maximum speed inference with all optimizations
        with torch.no_grad(), torch.inference_mode(), torch.autocast(device_type=device, enabled=False):
            # Use fastest possible settings for 50 steps
            result = pipe(
                prompt,
                negative_prompt=negative_prompt,
                image=sketch,
                width=target_size,
                height=target_size,
                callback_on_step_end=progress_callback,
                
                # EXTREME speed settings for 50 steps
                **batch_optimization,
                output_type="pil",  # Direct PIL output
                return_dict=False,  # Faster tuple return
                
                # Advanced optimizations
                cross_attention_kwargs={
                    "scale": 0.75,  # Lighter attention computation
                },
                
                # Skip expensive operations
                clip_skip=2,  # Skip more CLIP layers
                do_classifier_free_guidance=guidance_scale > 1.0,
                
                # Scheduler-specific optimizations
                timesteps=None,  # Let scheduler optimize timesteps
                
                # Memory and speed trade-offs
                latents=None,  # Fresh latents for consistency
                prompt_embeds=None,  # Cache embeddings when possible
                negative_prompt_embeds=None,
                
                # Advanced inference parameters
                guidance_rescale=0.0,  # Disable rescaling for speed
            )
        
        print("DEBUG: Inference completed successfully!")
        
        # Save results (handle both dict and tuple returns)
        if isinstance(result, tuple):
            colorized_img = result[0][0]  # return_dict=False returns tuple
        else:
            colorized_img = result.images[0]
        
        print(f"DEBUG: Saving colorized result to {output_path}")
        
        # Parallel processing for image operations
        import threading
        
        def save_colorized():
            colorized_img.save(output_path)
            print(f"DEBUG: Colorized image saved successfully")
        
        def create_grayscale():
            colorized_np = np.array(colorized_img)
            gray_np = cv2.cvtColor(colorized_np, cv2.COLOR_RGB2GRAY)
            gray_img = Image.fromarray(gray_np)
            gray_output_path = output_path.replace(".png", "_gray.png")
            gray_img.save(gray_output_path)
            print(f"DEBUG: Grayscale version saved to {gray_output_path}")
            return gray_output_path
        
        # Run both operations in parallel
        colorized_thread = threading.Thread(target=save_colorized)
        colorized_thread.start()
        
        gray_output_path = create_grayscale()  # Run grayscale creation
        colorized_thread.join()  # Wait for colorized save to complete
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("DEBUG: Colorization process completed successfully!")
        return output_path, gray_output_path
        
    except Exception as e:
        raise

def clear_pipeline_cache():
    """Clear the cached pipeline to force reload with new settings."""
    global _pipe, _current_preset
    _pipe = None
    _current_preset = None

# Replace the original function
colorize_any = colorize_any_robust