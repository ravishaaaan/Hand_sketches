#!/bin/bash
# Activation script for the Handsketch Recognition project

echo "🚀 Activating Handsketch Recognition Virtual Environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated!"
echo ""
echo "📦 Installed packages:"
echo "   - streamlit: $(python -c 'import streamlit; print(streamlit.__version__)')"
echo "   - torch: $(python -c 'import torch; print(torch.__version__)')"
echo "   - opencv: $(python -c 'import cv2; print(cv2.__version__)')"
echo "   - transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "   - diffusers: $(python -c 'import diffusers; print(diffusers.__version__)')"
echo ""
echo "🔍 Checking model status..."
python check_models.py
echo ""
echo "🎯 **Workflow:**"
echo "   1. First time: python setup_models.py  (downloads AI models)"
echo "   2. Check status: python check_models.py"
echo "   3. Run app: streamlit run app.py  (instant startup!)"
echo ""
echo "📝 To deactivate when done:"
echo "   deactivate"