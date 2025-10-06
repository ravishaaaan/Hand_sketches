import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np
import time
import warnings
from performance_config import get_active_config

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
_loading_in_progress = False
_current_preset = None

def _load_pipeline():
    """Lazy load the Stable Diffusion pipeline."""
    global _pipe, _loading_in_progress
    
    if _pipe is not None:
        return _pipe
    
    if _loading_in_progress:
        print("⏳ Pipeline loading already in progress, please wait...")
        return None
        
    _loading_in_progress = True
    
    try:
        print("🔄 Loading Stable Diffusion pipeline...")
        print("⏳ This may take a moment...")
        
        # Load ControlNet (for sketch-to-image)
        print("   Loading ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", 
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )

        print("   Loading Stable Diffusion pipeline...")
        _pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        print("   Configuring scheduler...")
        # Switch scheduler (fixes IndexError with PNDM on CPU)
        _pipe.scheduler = DDIMScheduler.from_config(_pipe.scheduler.config)

        print("   Moving to device...")
        # Move to device with proper memory management
        if device == "cuda":
            _pipe = _pipe.to(device)
            # Enable memory efficient attention for CUDA
            _pipe.enable_attention_slicing()
            _pipe.enable_model_cpu_offload()
        else:
            # For CPU, use to_empty to avoid meta tensor issues
            _pipe = _pipe.to("cpu")
        
        # Enable VAE slicing for memory efficiency
        _pipe.enable_vae_slicing()
        
        print("✅ Stable Diffusion pipeline loaded successfully!")
        _loading_in_progress = False
        return _pipe
        
    except Exception as e:
        print(f"❌ Error loading Stable Diffusion pipeline: {e}")
        print("💡 This might be a memory issue or corrupted cache.")
        print("💡 Try running 'python setup_models.py' to re-download models.")
        _loading_in_progress = False
        _pipe = None
        raise


def colorize_any(input_path, label, output_path="datasets/colorized_output.png", external_progress_callback=None, speed_mode="auto"):
    """
    Generate both colored and grayscale realistic images from a sketch with speed options.
    
    Args:
        input_path (str): Path to input sketch image.
        label (str): Predicted label from classifier.
        output_path (str): Path to save the colorized image.
        external_progress_callback: Optional progress callback function.
        speed_mode (str): "auto", "lightning", "fast", or "quality"
    
    Returns:
        tuple(str, str): Paths to (colorized_image, grayscale_image)
    """
    
    # Get performance configuration first
    from performance_config import ACTIVE_PRESET, get_active_config
    config = get_active_config()
    inference_steps = config["inference_steps"]
    
    print(f"🎯 Active preset: {ACTIVE_PRESET}")
    print(f"🎯 Config: {inference_steps} steps, {config['guidance_scale']} guidance, {config['image_size']}px")
    
    # SPEED MODE SELECTION (same logic as robust version)
    if speed_mode == "auto":
        # Auto-select based on step count
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
    print("🎨 Using QUALITY colorization (original pipeline)")
    
    global _pipe, _current_preset
    
    target_size = config["image_size"]
    
    # ALWAYS check if preset changed - reload pipeline if needed
    if _current_preset != ACTIVE_PRESET:
        print(f"🔄 Preset changed from {_current_preset} to {ACTIVE_PRESET} - FORCING pipeline reload...")
        _pipe = None  # Force reload
        _current_preset = ACTIVE_PRESET
    
    # Load pipeline if not already loaded
    pipe = _load_pipeline()
    
    if pipe is None:
        raise RuntimeError("Failed to load Stable Diffusion pipeline. Please check the setup.")
    
    # Load and optimize sketch
    sketch = load_image(input_path)
    print(f"Original sketch size: {sketch.size}")
    
    # Resize to optimal size for faster processing
    if sketch.size != (target_size, target_size):
        sketch = sketch.resize((target_size, target_size), Image.Resampling.LANCZOS)
        print(f"Resized sketch to {target_size}x{target_size} for optimal processing")

    # Enhanced prompt for better results
    prompt = f"A high quality realistic photo of a {label}, detailed, sharp focus"
    negative_prompt = "blurry, low quality, distorted, deformed"

    # Get FRESH config right before inference to ensure latest settings
    fresh_config = get_active_config()
    inference_steps = fresh_config["inference_steps"]
    guidance_scale = fresh_config["guidance_scale"]
    
    print(f"🚀 INFERENCE STARTING: {inference_steps} steps, {guidance_scale} guidance, {target_size}px")
    print(f"🔍 Double-check - Active preset: {ACTIVE_PRESET}")
    
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
            print(f"🎨 Colorization Progress: {progress:.1f}% (Step {step_index+1}/{inference_steps})")
            last_console_log_time = current_time
        
        return callback_kwargs
    
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=sketch,
        num_inference_steps=inference_steps,  # Use fresh config
        guidance_scale=guidance_scale,        # Use fresh config
        height=target_size,
        width=target_size,
        generator=torch.Generator().manual_seed(42),  # Consistent results
        callback_on_step_end=progress_callback
    )

    # Save colorized result
    colorized_img = result.images[0]
    colorized_img.save(output_path)
    print(f"Generated color image saved at: {output_path}")

    # Convert to grayscale using OpenCV
    colorized_np = np.array(colorized_img)
    gray_np = cv2.cvtColor(colorized_np, cv2.COLOR_RGB2GRAY)
    gray_img = Image.fromarray(gray_np)

    gray_output_path = output_path.replace(".png", "_gray.png")
    gray_img.save(gray_output_path)
    print(f"Generated grayscale image saved at: {gray_output_path}")

    return output_path, gray_output_path


def clear_pipeline_cache():
    """Clear the cached pipeline to force reload with new settings."""
    global _pipe, _current_preset
    _pipe = None
    _current_preset = None
    print("🗑️ Cleared pipeline cache - will reload with new settings")


if __name__ == "__main__":
    test_input = "datasets/sample_sketch.png"
    label = "tree"
    color_path, gray_path = colorize_any(test_input, label)
    print(f"Outputs: {color_path}, {gray_path}")
