"""
Performance configuration for the Handsketch Recognition and Colorization app.
Adjust these settings to balance speed vs quality based on your hardware.
"""

# === INFERENCE SPEED SETTINGS ===
INFERENCE_STEPS = 20  # Default: 20 (was 50). Lower = faster, higher = better quality
GUIDANCE_SCALE = 7.5  # Default: 7.5. Lower = faster convergence
IMAGE_SIZE = 512      # Default: 512. Lower = much faster, higher = better quality

# === SIMPLIFIED QUALITY PRESETS ===
PRESETS = {
    "fast": {
        "inference_steps": 5,
        "guidance_scale": 5.0,
        "image_size": 256,
        "description": "Fast (~30s), lower quality"
    },
    "medium": {
        "inference_steps": 10,
        "guidance_scale": 6.0,
        "image_size": 384,
        "description": "Medium (~1min), good quality"
    },
    "high": {
        "inference_steps": 20,
        "guidance_scale": 7.5,
        "image_size": 512,
        "description": "High (~2min), great quality"
    },
    "ultra": {
        "inference_steps": 50,
        "guidance_scale": 9.0,
        "image_size": 768,
        "description": "Ultra (~5min), maximum quality"
    }
}

# === MEMORY OPTIMIZATION ===
ENABLE_ATTENTION_SLICING = True    # Reduces VRAM usage
ENABLE_VAE_SLICING = True          # Reduces VRAM usage
ENABLE_CPU_OFFLOAD = True          # Offload to CPU when possible
USE_HALF_PRECISION = True          # Use float16 on CUDA for speed

# === CURRENT ACTIVE PRESET ===
ACTIVE_PRESET = "medium"  # Change this to switch presets

def get_active_config():
    """Get the current active configuration."""
    return PRESETS[ACTIVE_PRESET]

def apply_preset(preset_name):
    """Apply a specific preset configuration."""
    global ACTIVE_PRESET, INFERENCE_STEPS, GUIDANCE_SCALE, IMAGE_SIZE
    
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    # Check if preset actually changed
    preset_changed = ACTIVE_PRESET != preset_name
    
    ACTIVE_PRESET = preset_name
    config = PRESETS[preset_name]
    
    INFERENCE_STEPS = config["inference_steps"]
    GUIDANCE_SCALE = config["guidance_scale"] 
    IMAGE_SIZE = config["image_size"]
    
    print(f"✅ Applied preset '{preset_name}': {config['description']}")
    
    # Clear pipeline cache if preset changed to force reload
    if preset_changed:
        try:
            from src.colorize import clear_pipeline_cache
            clear_pipeline_cache()
        except ImportError:
            pass  # Module not loaded yet
    
    return config

if __name__ == "__main__":
    print("🚀 Performance Configuration")
    print("=" * 50)
    
    for name, config in PRESETS.items():
        active = " (ACTIVE)" if name == ACTIVE_PRESET else ""
        print(f"{name}{active}: {config['description']}")
        print(f"  - Steps: {config['inference_steps']}")
        print(f"  - Guidance: {config['guidance_scale']}")
        print(f"  - Size: {config['image_size']}px")
        print()
    
    print(f"Current active preset: {ACTIVE_PRESET}")
    print(f"Configuration: {get_active_config()}")