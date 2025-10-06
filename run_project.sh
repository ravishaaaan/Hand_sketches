#!/bin/bash

# ===================================================================================
# RUN PROJECT SCRIPT - Handsketch Recognition and Colorization
# ===================================================================================
# This script activates the virtual environment and runs the Streamlit application
# Usage: ./run_project.sh
# ===================================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project banner
echo -e "${PURPLE}============================================================${NC}"
echo -e "${CYAN}🎨 Handsketch Recognition and Colorization Project${NC}"
echo -e "${PURPLE}============================================================${NC}"
echo ""

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ Error: app.py not found in current directory${NC}"
    echo -e "${YELLOW}💡 Please run this script from the project root directory${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Error: Virtual environment (.venv) not found${NC}"
    echo -e "${YELLOW}💡 Please create virtual environment first:${NC}"
    echo -e "${YELLOW}   python -m venv .venv${NC}"
    echo -e "${YELLOW}   source .venv/bin/activate${NC}"
    echo -e "${YELLOW}   pip install -r requirements.txt${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}🚀 Activating virtual environment...${NC}"
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Virtual environment activated!${NC}"
echo ""

# Check Python packages
echo -e "${BLUE}📦 Checking installed packages...${NC}"
python -c "
try:
    import streamlit
    import torch
    import transformers
    import diffusers
    import cv2
    import numpy
    print('✅ All required packages are installed!')
    print(f'   - Streamlit: {streamlit.__version__}')
    print(f'   - PyTorch: {torch.__version__}')
    print(f'   - Transformers: {transformers.__version__}')
    print(f'   - Diffusers: {diffusers.__version__}')
    print(f'   - OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'❌ Missing package: {e}')
    print('💡 Please install requirements: pip install -r requirements.txt')
    exit(1)
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Some required packages are missing${NC}"
    echo -e "${YELLOW}💡 Installing requirements...${NC}"
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install requirements${NC}"
        exit 1
    fi
fi

echo ""

# Hardware detection
echo -e "${BLUE}🖥️  Hardware Detection:${NC}"
python -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'✅ GPU Available: {gpu_name}')
    print(f'   Memory: {gpu_memory:.1f}GB')
    if gpu_memory >= 8:
        print('💡 Recommended: High or Ultra quality presets')
    elif gpu_memory >= 6:
        print('💡 Recommended: Medium or High quality presets')
    else:
        print('💡 Recommended: Fast or Medium quality presets')
else:
    print('🖥️  CPU Only Mode')
    print('💡 Recommended: Fast quality preset for best performance')
" 2>/dev/null

echo ""

# Check for AI models (optional)
echo -e "${BLUE}🤖 Checking AI models status...${NC}"
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    model_count=$(find "$HOME/.cache/huggingface/hub" -name "models--*" -type d 2>/dev/null | wc -l)
    if [ $model_count -gt 0 ]; then
        echo -e "${GREEN}✅ AI models cache found ($model_count models)${NC}"
        echo -e "${GREEN}   Models will load from cache for faster startup${NC}"
    else
        echo -e "${YELLOW}⚠️  No AI models cached yet${NC}"
        echo -e "${YELLOW}   Models will be downloaded on first use (~6GB)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No model cache directory found${NC}"
    echo -e "${YELLOW}   Models will be downloaded on first use (~6GB)${NC}"
fi

echo ""

# Create necessary directories
echo -e "${BLUE}📁 Creating project directories...${NC}"
mkdir -p datasets outputs
echo -e "${GREEN}✅ Directories ready${NC}"

echo ""

# Performance recommendations
echo -e "${CYAN}⚡ Performance Recommendations:${NC}"
echo -e "${YELLOW}   • Fast (5 steps):    ~30 seconds  - Good for testing${NC}"
echo -e "${YELLOW}   • Medium (10 steps):  ~1 minute   - Balanced quality/speed${NC}"
echo -e "${YELLOW}   • High (20 steps):    ~2 minutes  - Great quality${NC}"
echo -e "${YELLOW}   • Ultra (50 steps):   ~5 minutes  - Maximum quality${NC}"

echo ""

# Launch application
echo -e "${GREEN}🚀 Starting Streamlit application...${NC}"
echo -e "${CYAN}   Local URL: http://localhost:8501${NC}"
echo -e "${CYAN}   Network URL: Will be displayed below${NC}"
echo ""
echo -e "${PURPLE}============================================================${NC}"
echo -e "${YELLOW}📝 To stop the application: Press Ctrl+C${NC}"
echo -e "${YELLOW}📝 To deactivate venv later: Type 'deactivate'${NC}"
echo -e "${PURPLE}============================================================${NC}"
echo ""

# Run the Streamlit app
streamlit run app.py --server.port 8501 --server.address localhost

# Check if streamlit command failed
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Failed to start Streamlit application${NC}"
    echo -e "${YELLOW}💡 Troubleshooting steps:${NC}"
    echo -e "${YELLOW}   1. Check if all requirements are installed${NC}"
    echo -e "${YELLOW}   2. Verify virtual environment is activated${NC}"
    echo -e "${YELLOW}   3. Try: pip install streamlit --upgrade${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Application stopped successfully${NC}"
echo -e "${YELLOW}💡 Virtual environment is still active${NC}"
echo -e "${YELLOW}   Type 'deactivate' to exit virtual environment${NC}"