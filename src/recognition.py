# src/recognition.py
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os
import warnings

# Suppress transformers warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_NAME = "kmewhort/beit-sketch-classifier"

# Global variables for lazy loading
_processor = None
_model = None

def _load_model():
    """Lazy load the recognition model and processor."""
    global _processor, _model
    
    if _processor is None or _model is None:
        print("🔄 Loading sketch recognition model...")
        try:
            _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
            _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
            _model.eval()
            print("✅ Recognition model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading recognition model: {e}")
            print("💡 Please run 'python setup_models.py' first to download models.")
            raise
    
    return _processor, _model

def predict_sketch(image_path: str):
    """Predict the label of a sketch image with confidence score."""
    print(f"\nDEBUG: Starting sketch recognition for {image_path}")
    
    # Load models if not already loaded
    print("DEBUG: Loading BEiT model for sketch recognition...")
    processor, model = _load_model()
    print("DEBUG: BEiT model loaded successfully")
    
    # Optimize image loading and preprocessing
    img = Image.open(image_path).convert("RGB")
    print(f"DEBUG: Processing image - Original size: {img.size}")
    
    # Resize to optimal size for faster processing
    if img.size != (224, 224):
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        print("DEBUG: Image resized to 224x224 for BEiT")
    
    inputs = processor(images=img, return_tensors="pt")

    # Optimized inference with torch optimizations
    print("DEBUG: Running BEiT inference...")
    with torch.no_grad():
        with torch.inference_mode():  # Additional optimization
            outputs = model(**inputs)

    logits = outputs.logits
    pred_idx = logits.argmax(dim=-1).item()
    label = model.config.id2label[pred_idx]
    confidence = torch.softmax(logits, dim=-1)[0][pred_idx].item()
    
    print(f"DEBUG: Recognition complete - Label: '{label}', Confidence: {confidence:.3f}")
    return label, confidence
