import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np
import gc
import time
from performance_config import get_active_config
import cpu_optimizations  # Auto-apply CPU optimizations

# Decide tensor dtype based on device (CPU cannot handle float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

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
        
        # Configure ULTRA-FAST scheduler - LCMScheduler is fastest
        try:
            from diffusers import LCMScheduler
            _pipe.scheduler = LCMScheduler.from_config(_pipe.scheduler.config)
        except ImportError:
            try:
                from diffusers import DPMSolverMultistepScheduler
                _pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    _pipe.scheduler.config,
                    use_karras_sigmas=False,  # Faster without karras
                    algorithm_type="dpmsolver",  # Fastest variant
                    solver_order=1,  # Lower order = faster
                )
            except ImportError:
                # Ultra-fast DDIM config
                _pipe.scheduler = DDIMScheduler.from_config(
                    _pipe.scheduler.config,
                    clip_sample=False,  # Faster
                    set_alpha_to_one=False,  # Faster
                )
        
        # AGGRESSIVE performance optimizations
        if hasattr(_pipe, 'enable_attention_slicing'):
            _pipe.enable_attention_slicing("max")  # Maximum slicing for speed
            
        if hasattr(_pipe, 'enable_vae_slicing'):
            _pipe.enable_vae_slicing()  # VAE slicing for memory/speed
            
        if hasattr(_pipe, 'enable_vae_tiling'):
            _pipe.enable_vae_tiling()  # VAE tiling for large images
            
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

def colorize_any_robust(input_path, label, output_path="datasets/colorized_output.png", external_progress_callback=None):
    """
    Robust colorization with performance configuration support.
    """
    global _pipe, _current_preset
    
    print(f"\nDEBUG: Starting colorization for '{label}' sketch at {input_path}")
    
    try:
        # Get fresh performance configuration
        from performance_config import ACTIVE_PRESET, get_active_config
        config = get_active_config()
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
        
        # Time-based callback tracking (15-second intervals)
        callback_start_time = time.time()
        last_callback_time = callback_start_time
        
        # Progress callback with 15-second time-based frequency
        def progress_callback(step, timestep, latents):
            nonlocal last_callback_time
            current_time = time.time()
            progress = (step / inference_steps) * 100
            
            # Time-based callback: run every 15 seconds regardless of steps
            if current_time - last_callback_time >= 15.0:
                print(f"DEBUG: Inference progress - Step {step}/{inference_steps} ({progress:.1f}%) [15s interval]")
                
                # Call external progress callback if provided (for UI updates)
                if external_progress_callback:
                    external_progress_callback(step, inference_steps, progress)
                
                last_callback_time = current_time
        
        # ULTRA-AGGRESSIVE inference optimizations
        generator = torch.Generator(device=device).manual_seed(42)
        
        print("DEBUG: Starting Stable Diffusion inference...")
        # Use torch.no_grad() and inference_mode for maximum speed
        with torch.no_grad(), torch.inference_mode():
            result = pipe(
                prompt,
                negative_prompt=negative_prompt,
                image=sketch,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                width=target_size,
                height=target_size,
                generator=generator,
                callback=progress_callback,
                callback_steps=1,  # Check every step for 15-second time intervals
                # ULTRA-FAST settings
                eta=0.0,  # Deterministic = faster
                output_type="pil",  # Direct output
                return_dict=False,  # Faster return
                cross_attention_kwargs={"scale": 0.8},  # Lighter attention
                clip_skip=1,  # Skip CLIP layers for speed
                # Reduce quality for speed
                do_classifier_free_guidance=guidance_scale > 1.0,
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