import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np
import time
from performance_config import get_active_config

# Decide tensor dtype based on device (CPU cannot handle float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

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


def colorize_any(input_path, label, output_path="datasets/colorized_output.png"):
    """
    Generate both colored and grayscale realistic images from a sketch.
    
    Args:
        input_path (str): Path to input sketch image.
        label (str): Predicted label from classifier.
        output_path (str): Path to save the colorized image.
    
    Returns:
        tuple(str, str): Paths to (colorized_image, grayscale_image)
    """
    global _pipe, _current_preset
    
    # Get fresh performance configuration
    from performance_config import ACTIVE_PRESET, get_active_config
    config = get_active_config()
    target_size = config["image_size"]
    
    print(f"🎯 Active preset: {ACTIVE_PRESET}")
    print(f"🎯 Config: {config['inference_steps']} steps, {config['guidance_scale']} guidance, {target_size}px")
    
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
    
    # Time-based callback tracking (15-second intervals)
    callback_start_time = time.time()
    last_callback_time = callback_start_time
    
    # Progress callback with 15-second time-based frequency
    def progress_callback(step, timestep, latents):
        nonlocal last_callback_time
        current_time = time.time()
        progress = (step + 1) / inference_steps
        
        # Time-based callback: run every 15 seconds regardless of steps
        if current_time - last_callback_time >= 15.0:
            print(f"   Inference step {step+1}/{inference_steps} ({progress*100:.1f}%) [15s interval]")
            last_callback_time = current_time
    
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=sketch,
        num_inference_steps=inference_steps,  # Use fresh config
        guidance_scale=guidance_scale,        # Use fresh config
        height=target_size,
        width=target_size,
        generator=torch.Generator().manual_seed(42),  # Consistent results
        callback=progress_callback,
        callback_steps=1
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
