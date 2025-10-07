import streamlit as st
from streamlit_drawable_canvas import st_canvas
import warnings

# Suppress all warnings for clean UI output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cross_attention_kwargs.*")
warnings.filterwarnings("ignore", message=".*slice_size.*")
warnings.filterwarnings("ignore", message=".*AttnProcessor.*")

from src.grayscale import convert_to_grayscale
try:
    from src.colorize_robust import colorize_any
except ImportError:
    from src.colorize import colorize_any
from src.recognition import predict_sketch
import numpy as np
import cv2
import os
import time
from datetime import datetime
from pathlib import Path

st.title("Handsketch Recognition and Colorization")

# Performance preset selector
from performance_config import PRESETS, apply_preset, get_active_config

st.markdown("### ⚙️ **Performance Settings**")

# Create columns for performance selector
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    # Performance preset options
    preset_options = {
        "⚡ Fast (5 steps)": "fast",
        "⚖️ Medium (10 steps)": "medium", 
        "🎨 High (20 steps)": "high",
        "💎 Ultra (50 steps)": "ultra"
    }
    
    selected_preset_display = st.selectbox(
        "Choose Quality Level:",
        options=list(preset_options.keys()),
        index=1,  # Default to "Medium" (index 1)
        help="Fast ~40s, Medium ~2min, High ~15min, Ultra ~30min"
    )
    
    selected_preset = preset_options[selected_preset_display]

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacer

with col3:
    # Show current preset info and hardware recommendation
    if selected_preset in PRESETS:
        config = PRESETS[selected_preset]
        
        # Simple CPU-only hardware info
        recommended = "fast"
        hardware_info = "🖥️ CPU Only\n💡 Recommended: Fast"
        
        st.info(f"""
**{config['description']}**
- Steps: {config['inference_steps']}
- Size: {config['image_size']}px
- Guidance: {config['guidance_scale']}

{hardware_info}
        """)

# Apply the selected preset
if selected_preset:
    try:
        apply_preset(selected_preset)
        current_config = get_active_config()
    except Exception as e:
        st.error(f"Error applying preset: {e}")
        current_config = get_active_config()



st.markdown("---")

# Check if models are ready
@st.cache_data
def check_models_ready():
    """Check if all required models are cached."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return False
    
    required_patterns = [
        "models--kmewhort--beit-sketch-classifier",
        "models--lllyasviel--sd-controlnet-scribble", 
        "models--runwayml--stable-diffusion-v1-5"
    ]
    
    cached_models = [d.name for d in cache_dir.iterdir() if d.is_dir()]
    
    for pattern in required_patterns:
        if not any(pattern == cached for cached in cached_models):
            return False
    
    return True

# Display model status
if not check_models_ready():
    st.error("🚨 **Models not found!**")
    st.markdown("""
    **Required AI models are not downloaded yet.**
    
    Please run the setup first:
    ```bash
    python setup_models.py
    ```
    
    This will download ~6-7GB of AI models (one-time setup).
    """)
    st.stop()
else:
    st.success("✅ **Models ready!** All AI models are cached and ready for instant use.")

# Create required folders
os.makedirs("datasets", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Initialize session state
if "input_image" not in st.session_state:
    st.session_state.input_image = None
if "processing_started" not in st.session_state:
    st.session_state.processing_started = False

# --- User choice: upload or draw ---
st.markdown("### 🎨 **Step 1: Create Your Sketch**")
option = st.radio("Choose input method:", ["Upload Sketch", "Draw Sketch"])

input_image = None
sketch_ready = False

# --- Upload Sketch ---
if option == "Upload Sketch":
    uploaded_file = st.file_uploader("Upload a sketch image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img_path = f"datasets/{uploaded_file.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_image = img_path
        st.session_state.input_image = img_path
        sketch_ready = True
        
        # Show preview
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(input_image, caption="📁 Uploaded Sketch", width="stretch")
            st.success("✅ Sketch uploaded successfully!")

# --- Draw Sketch ---
elif option == "Draw Sketch":
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = f"canvas_{datetime.now().timestamp()}"

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ New Sketch", width="stretch"):
            st.session_state.canvas_key = f"canvas_{datetime.now().timestamp()}"
            st.session_state.input_image = None

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=5,
        stroke_color="black",
        background_color="white",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key
    )

    if canvas_result.image_data is not None:
        # Check if there's actual drawing (not just white canvas)
        img_array = canvas_result.image_data.astype(np.uint8)
        drawn_img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Check if there's any non-white pixels (indicating drawing)
        if not np.all(drawn_img >= 250):  # Not all pixels are near-white
            # Store the image data in session state, but don't save to disk yet
            st.session_state.canvas_image_data = drawn_img
            sketch_ready = True
            st.success("✅ Sketch drawn successfully!")
        else:
            st.info("👆 Please draw something on the canvas above")
            st.session_state.canvas_image_data = None

# --- Step 2: Process Button ---
st.markdown("---")
st.markdown("### 🚀 **Step 2: Process Your Sketch**")

if (st.session_state.input_image or st.session_state.get('canvas_image_data') is not None) and not st.session_state.processing_started:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 **Process Sketch with AI**", 
                    type="primary", 
                    width="stretch",
                    help="Click to start AI recognition and colorization"):
            
            # Save canvas image to file only when processing starts
            if st.session_state.get('canvas_image_data') is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                input_image = f"datasets/drawn_sketch_{timestamp}.png"
                cv2.imwrite(input_image, st.session_state.canvas_image_data)
                st.session_state.input_image = input_image
                print(f"DEBUG: Saved canvas sketch to {input_image}")
            
            st.session_state.processing_started = True
            st.rerun()

elif not st.session_state.input_image and st.session_state.get('canvas_image_data') is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("🎯 **Process Sketch with AI**", 
                 disabled=True, 
                 width="stretch",
                 help="Please upload or draw a sketch first")

# --- Process the image if button was clicked ---
if st.session_state.input_image and st.session_state.processing_started:
    
    # Show active performance preset and force clear cache
    current_config = get_active_config()
    st.info(f"🚀 **Processing with {selected_preset_display} preset:** {current_config['description']}")
    
    # Force clear pipeline cache to ensure fresh settings
    from src.colorize import clear_pipeline_cache
    clear_pipeline_cache()
    
    # Add CSS for modern progress indicators
    st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        border-radius: 10px;
    }
    .progress-text {
        font-size: 16px;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin: 10px 0;
    }
    .step-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 **AI Processing in Progress...**")
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        
        import time
        start_time = time.time()
        
        # Ultra-optimized timing estimates
        step_timings = {
            "lightning": {"grayscale": 0.5, "recognition": 2, "colorization": 30},
            "ultra_fast": {"grayscale": 0.5, "recognition": 2, "colorization": 60},
            "fast": {"grayscale": 1, "recognition": 3, "colorization": 120}, 
            "balanced": {"grayscale": 1, "recognition": 3, "colorization": 180},
            "quality": {"grayscale": 1, "recognition": 3, "colorization": 300},
            "ultra_quality": {"grayscale": 1, "recognition": 3, "colorization": 450}
        }
        
        def update_progress_with_eta(step, elapsed_time):
            """Update progress with accurate estimated time remaining"""
            preset = selected_preset if selected_preset in step_timings else "fast"
            timings = step_timings[preset]
            
            if step == "grayscale":
                remaining = timings["recognition"] + timings["colorization"]
                total_estimated = timings["grayscale"] + timings["recognition"] + timings["colorization"]
                progress_percent = 15
            elif step == "recognition": 
                remaining = timings["colorization"]
                total_estimated = timings["grayscale"] + timings["recognition"] + timings["colorization"]
                progress_percent = 40
            elif step == "colorization":
                # During colorization, estimate based on inference steps
                config = get_active_config()
                steps = config["inference_steps"]
                # Rough estimate: each step takes about colorization_time/steps seconds
                step_time = timings["colorization"] / steps
                remaining = max(0, timings["colorization"] - elapsed_time + timings["grayscale"] + timings["recognition"])
                total_estimated = timings["grayscale"] + timings["recognition"] + timings["colorization"]
                progress_percent = min(90, 40 + (elapsed_time - timings["grayscale"] - timings["recognition"]) / timings["colorization"] * 50)
            else:
                remaining = 0
                total_estimated = elapsed_time
                progress_percent = 100
            
            if remaining > 0:
                return f"⏱️ *Elapsed: {elapsed_time:.1f}s | ETA: ~{remaining:.0f}s remaining*"
            else:
                return f"⏱️ *Elapsed: {elapsed_time:.1f}s | Almost done...*"
        
        # Step 1: Convert to grayscale
        status_text.markdown('<div class="progress-text">🔄 Step 1/3: Converting to grayscale...</div>', unsafe_allow_html=True)
        progress_bar.progress(10)
        elapsed = time.time() - start_time
        time_text.markdown(update_progress_with_eta("grayscale", elapsed))
        
        gray_path = convert_to_grayscale(st.session_state.input_image)
        progress_bar.progress(25)
        elapsed = time.time() - start_time
        time_text.markdown(update_progress_with_eta("grayscale", elapsed))
        
        # Step 2: AI Recognition (Optimized)
        status_text.markdown('<div class="progress-text">🧠 Step 2/3: AI recognizing sketch...</div>', unsafe_allow_html=True)
        progress_bar.progress(35)
        elapsed = time.time() - start_time
        time_text.markdown(update_progress_with_eta("recognition", elapsed))
        
        # Ensure preset is applied before any pipeline operations
        from performance_config import apply_preset
        apply_preset(selected_preset)  # Force apply the current preset
        
        # Pre-load colorization pipeline while doing recognition (parallel processing)
        import threading
        from src.colorize import _load_pipeline
        
        # Start loading colorization pipeline in background
        pipeline_thread = threading.Thread(target=_load_pipeline)
        pipeline_thread.start()
        
        print(f"\n🚀 DEBUG: Processing started with {selected_preset_display} preset")
        print(f"DEBUG: Input file: {st.session_state.input_image}")
        
        label, confidence = predict_sketch(gray_path)
        progress_bar.progress(50)
        elapsed = time.time() - start_time
        time_text.markdown(update_progress_with_eta("recognition", elapsed))
        
        prediction_text = f"**🎯 Recognized:** *{label}* (Confidence: {confidence*100:.1f}%)"
        print(f"DEBUG: ✅ Recognition completed: '{label}' ({confidence*100:.1f}% confidence)")
        
        # Step 3: AI Colorization (Pipeline should be loaded by now)
        status_text.markdown('<div class="progress-text">🎨 Step 3/3: AI generating colorized image...</div>', unsafe_allow_html=True)
        progress_bar.progress(60)
        elapsed = time.time() - start_time
        time_text.markdown(update_progress_with_eta("colorization", elapsed))
        
        # Wait for pipeline loading to complete
        pipeline_thread.join()
        
        # Try colorization + grayscale-real generation
        try:
            output_file = f"outputs/colorized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            # Show current preset being used
            config = get_active_config()
            status_text.markdown(f'<div class="progress-text">🎨 Generating with {config["inference_steps"]} steps at {config["image_size"]}px...</div>', unsafe_allow_html=True)
            
            print(f"DEBUG: Starting colorization for '{label}'...")
            
            # Create LIVE progress callback for real-time UI updates
            def colorization_progress_callback(current_step, total_steps, step_progress):
                # Map colorization progress to overall progress (60% to 90%)
                base_progress = 60  # Progress after recognition
                colorization_range = 30  # Range for colorization (60% to 90%)
                overall_progress = base_progress + (step_progress / 100) * colorization_range
                
                # Update progress bar IMMEDIATELY
                progress_bar.progress(int(overall_progress))
                
                # Update status with LIVE step information
                elapsed = time.time() - start_time
                step_text = f"Step {current_step + 1}/{total_steps}"  # +1 for user-friendly 1-based counting
                status_text.markdown(f'<div class="progress-text">🎨 Generating colorized image... {step_text} ({step_progress:.1f}%)</div>', unsafe_allow_html=True)
                
                # Force immediate UI update using placeholder refresh
                try:
                    # Update session state for live tracking
                    st.session_state.current_step = current_step
                    st.session_state.total_steps = total_steps
                    st.session_state.step_progress = step_progress
                    st.session_state.elapsed_time = elapsed
                    
                    # Force UI refresh by updating container content
                    progress_bar.progress(int(overall_progress))
                    
                except Exception:
                    pass  # Continue if UI refresh fails
                
                # Calculate LIVE ETA for ENTIRE process completion
                if current_step >= 0:  # Show ETA immediately (starts from step 0)
                    # Calculate colorization progress
                    completed_steps = current_step + 1  # +1 because step counting starts from 0
                    colorization_progress = completed_steps / total_steps  # 0.0 to 1.0
                    
                    # Overall process stages:
                    # 1. Setup & Loading: 0% - 10% (already done)
                    # 2. Recognition: 10% - 60% (already done)  
                    # 3. Colorization: 60% - 90% (current stage)
                    # 4. Saving & Cleanup: 90% - 100% (estimated ~2-5s)
                    
                    current_overall_progress = 0.6 + (colorization_progress * 0.3)  # 60% to 90%
                    
                    # LIVE ETA calculation - update every step
                    if current_step >= 1:  # After at least 1 step completed
                        # Calculate time per step for more accurate ETA
                        time_per_step = elapsed / completed_steps
                        remaining_steps = total_steps - completed_steps
                        colorization_eta = remaining_steps * time_per_step
                        
                        # Add buffer for final saving/cleanup (90%-100%)
                        final_stage_buffer = 3  # Estimated 3 seconds for saving
                        total_remaining_time = colorization_eta + final_stage_buffer
                        
                        if total_remaining_time > 2:  # Show ETA if meaningful
                            minutes = int(total_remaining_time // 60)
                            seconds = int(total_remaining_time % 60)
                            if minutes > 0:
                                time_text.markdown(f"⏱️ *Elapsed: {elapsed:.0f}s | ETA: ~{minutes}m {seconds}s remaining*")
                            else:
                                time_text.markdown(f"⏱️ *Elapsed: {elapsed:.0f}s | ETA: ~{seconds}s remaining*")
                        else:
                            time_text.markdown(f"⏱️ *Elapsed: {elapsed:.0f}s | Almost complete!*")
                    else:
                        # First step - show basic info
                        time_text.markdown(f"⏱️ *Elapsed: {elapsed:.0f}s | Calculating ETA...*")
                else:
                    time_text.markdown(f"⏱️ *Elapsed: {elapsed:.1f}s | Processing...*")
            
            # Call colorization with progress callback
            colorized_path, gray_real_path = colorize_any(gray_path, label, output_file, 
                                                        external_progress_callback=colorization_progress_callback)
            
            progress_bar.progress(90)
            elapsed = time.time() - start_time
            time_text.markdown(f"⏱️ *Total time: {elapsed:.1f}s*")
            print(f"DEBUG: ✅ Colorization completed successfully!")
            
        except Exception as e:
            colorized_path, gray_real_path = None, None
            st.error(f"❌ Colorization failed: {e}")
            progress_bar.progress(100)
            elapsed = time.time() - start_time
            time_text.markdown(f"⏱️ *Total time: {elapsed:.1f}s*")
        
        # Completion
        if colorized_path and gray_real_path:
            progress_bar.progress(100)
            elapsed = time.time() - start_time
            status_text.markdown('<div class="progress-text">✅ Processing completed successfully!</div>', unsafe_allow_html=True)
            time_text.markdown(f"⏱️ *Total processing time: {elapsed:.1f}s*")
            
            print(f"DEBUG: 🎉 All processing completed in {elapsed:.1f}s")
            print(f"DEBUG: Results saved to {colorized_path} and {gray_real_path}")
            
            # Small delay to show completion
            time.sleep(0.5)
            
    # Clear progress indicators after completion
    if colorized_path and gray_real_path:
        progress_container.empty()
        
        # Display results section
        st.markdown("---")
        st.markdown("### 🎉 **Results**")
        
        # Show prediction info
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="step-indicator">
                {prediction_text}
            </div>
            """, unsafe_allow_html=True)
        
        # Display images in horizontal line
        st.markdown("#### 📸 **Generated Images**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(st.session_state.input_image, caption="🖼️ Original Sketch", width="stretch")
            
        with col2:
            st.image(gray_real_path, caption="⚫ Generated Grayscale", width="stretch")
            
        with col3:
            st.image(colorized_path, caption="🌈 AI Colorized Result", width="stretch")
        
        # Reset button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 **Process Another Sketch**", 
                        type="secondary", 
                        width="stretch"):
                # Reset session state
                st.session_state.input_image = None
                st.session_state.processing_started = False
                if "canvas_key" in st.session_state:
                    st.session_state.canvas_key = f"canvas_{datetime.now().timestamp()}"
                st.rerun()

# --- Sidebar with Performance Tips ---
with st.sidebar:
    st.markdown("## 🚀 **Performance Guide**")
    
    st.markdown("### ⚡ **Quality Levels:**")
    st.markdown("""
    - **⚡ Fast (5 steps)**: Quick results (~40s) 
    - **⚖️ Medium (10 steps)**: Good balance (~2min)
    - **🎨 High (20 steps)**: Great quality (~15min)  
    - **💎 Ultra (50 steps)**: Maximum quality (~30min)
    """)
    
    st.markdown("### 💡 **Speed Tips:**")
    st.markdown("""
    - **CPU Users**: Use Fast (5 steps) for speed
    - **GPU Users**: Can use High/Ultra (20-50 steps)
    - **First time**: Try Medium (10 steps) for balance
    - **Simple drawings**: Work with any preset
    """)
    
    st.markdown("### 🎯 **Quality Tips:**")
    st.markdown("""
    - **Clean sketches**: Give better results
    - **Clear outlines**: Improve recognition
    - **Simple objects**: Work best (animals, objects)
    - **Avoid text**: Focus on drawable objects
    """)
    
    # Hardware info
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"🖥️ **GPU Detected:**\n{gpu_name}")
    else:
        st.info("🖥️ **CPU Mode**\nConsider faster presets")
    
    st.markdown("---")
    st.markdown("**💬 Having issues?**\nTry a faster preset or simpler sketches!")
