#!/bin/bash

# TOTNet Ball Tracking Demo Script
# This script sets up the environment and runs ball tracking on a video

set -e

# Configuration
VIDEO_PATH="${1:-}"  # First argument is video path, or empty
MODEL_TYPE="${2:-tennis}"  # Second argument is model type: tennis or badminton

# Determine which weights to use based on model type
if [ "$MODEL_TYPE" = "badminton" ]; then
    WEIGHTS_FILE="TOTNet_Badminton_(5)_(288,512)_30epochs_Occl(0.25)_WBCE[1,2,3,3]_bs8_ch64_best.pth"
    WEIGHTS_DIR="TOTNet_Badminton_(5)_(288,512)_30epochs_Occl(0.25)_WBCE[1,2,3,3]_bs8_ch64"
    DATASET_CHOICE="badminton"
else
    WEIGHTS_FILE="TOTNet_Tennis_(5)_(288,512)_30epochs_Occl(0.25)_WBCE[1,2,3,3]_bs8_ch64_best.pth"
    WEIGHTS_DIR="TOTNet_Tennis_(5)_(288,512)_30epochs_Occl(0.25)_WBCE[1,2,3,3]_bs8_ch64"
    DATASET_CHOICE="tennis"
fi

echo "========================================"
echo "TOTNet Ball Tracking Demo"
echo "========================================"
echo "Model type: $MODEL_TYPE"
echo "Video path: $VIDEO_PATH"
echo "========================================"

# Check if video path is provided
if [ -z "$VIDEO_PATH" ]; then
    echo "Usage: ./run_demo.sh <video_path> [tennis|badminton]"
    echo ""
    echo "Example:"
    echo "  ./run_demo.sh /path/to/tennis_match.mp4 tennis"
    echo "  ./run_demo.sh /path/to/badminton_match.mp4 badminton"
    exit 1
fi

# Convert to absolute path
VIDEO_PATH="$(cd "$(dirname "$VIDEO_PATH")" 2>/dev/null && pwd)/$(basename "$VIDEO_PATH")"

# Check if video file exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found at $VIDEO_PATH"
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

echo ""
echo "Step 1: Setting up virtual environment with uv..."
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.10
    echo "Created new virtual environment"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

echo ""
echo "Step 2: Installing dependencies..."
echo ""

# Install PyTorch first (CPU version for Mac, CUDA version will be auto-detected on Linux/Windows)
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS - installing PyTorch for CPU/MPS..."
    uv pip install torch torchvision torchaudio
else
    echo "Detected Linux/Windows - installing PyTorch with CUDA support..."
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install other dependencies
uv pip install \
    easydict \
    matplotlib \
    opencv-python \
    scikit-learn \
    cython \
    pycocotools \
    tqdm \
    scipy \
    ninja \
    tensorboard \
    ptflops \
    einops \
    pyyaml

echo ""
echo "Step 3: Running ball tracking demo..."
echo ""

# Enable MPS fallback for unsupported operations (like adaptive_max_pool3d)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the demo
cd "$PROJECT_ROOT/src"

python demo.py \
    --model_choice TOTNet \
    --pretrained_path "$PROJECT_ROOT/weights/$WEIGHTS_DIR/$WEIGHTS_FILE" \
    --video_path "$VIDEO_PATH" \
    --dataset_choice "$DATASET_CHOICE" \
    --num_frames 5 \
    --img_size 288 512 \
    --num_channels 64 \
    --save_demo_output \
    --output_format video \
    --saved_fn "${MODEL_TYPE}_demo"

echo ""
echo "========================================"
echo "Demo complete!"
echo "Output saved to: $PROJECT_ROOT/results/demo/${MODEL_TYPE}_demo/"
echo "  - Frames: results/demo/${MODEL_TYPE}_demo/frame/"
echo "  - Video:  results/demo/${MODEL_TYPE}_demo/result.mp4"
echo "========================================"
