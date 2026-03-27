#!/bin/bash
# Football Demo Script with UV
# ============================
# This script creates a uv environment, installs dependencies, and runs the demo

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TOTNet Football Demo ===${NC}"

# Change to project root
cd "$(dirname "$0")"

# Configuration - Edit these as needed
VIDEO_PATH="${VIDEO_PATH:-data/football_dataset/frames/football-}"
MODEL_PATH="weights/TOTNET_Football/TOTNet_Football_best.pth"
OUTPUT_NAME="${OUTPUT_NAME:-football_demo}"

# Check if video/frame path exists
if [ ! -e "$VIDEO_PATH" ]; then
    echo -e "${RED}Error: Input path not found: $VIDEO_PATH${NC}"
    echo -e "${YELLOW}Please set VIDEO_PATH environment variable or edit this script.${NC}"
    echo "Example: VIDEO_PATH=/path/to/video.mp4 ./demo_football.sh"
    exit 1
fi

# Check if model weights exist
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model weights not found: $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Setting up Python environment with uv...${NC}"

# Create venv and sync dependencies using uv
uv venv .venv --python 3.10
source .venv/bin/activate

echo -e "${GREEN}Step 2: Installing dependencies...${NC}"
uv pip sync requirements.txt 2>/dev/null || uv pip install -r requirements.txt

echo -e "${GREEN}Step 3: Running demo...${NC}"
echo -e "  Input: ${YELLOW}$VIDEO_PATH${NC}"
echo -e "  Model: ${YELLOW}$MODEL_PATH${NC}"
echo -e "  Output: ${YELLOW}results/demo/$OUTPUT_NAME/result.mp4${NC}"

cd src

python demo.py \
    --model_choice 'motion_light' \
    --num_frames 5 \
    --dataset_choice football \
    --video_path "../$VIDEO_PATH" \
    --pretrained_path "../$MODEL_PATH" \
    --save_demo_output \
    --output_format video \
    --saved_fn "$OUTPUT_NAME"

echo -e "${GREEN}=== Demo Complete ===${NC}"
echo -e "Output video: ${YELLOW}results/demo/$OUTPUT_NAME/result.mp4${NC}"
echo -e "Ball coordinates: ${YELLOW}results/demo/$OUTPUT_NAME/ball_coordinates.json${NC}"
