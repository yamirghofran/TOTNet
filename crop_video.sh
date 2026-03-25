#!/bin/bash
#
# Generate 9:16 vertical video following ball trajectory
# 
# This script creates a vertical video crop that follows the ball with
# smooth camera movement and dynamic lead room.
#
# Usage:
#   ./crop_video.sh <coordinates.json> <input_video.mp4> <output_video.mp4> [options]
#
# Example:
#   ./crop_video.sh results/demo/tennis_demo/ball_coordinates.json \
#                    input.mp4 \
#                    output_vertical.mp4 \
#                    --smoothing 0.5 \
#                    --lead-room 0.15

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    echo "Usage: $0 <coordinates.json> <input_video.mp4> <output_video.mp4> [options]"
    echo ""
    echo "Arguments:"
    echo "  coordinates.json  Path to ball_coordinates.json from demo.py"
    echo "  input_video.mp4   Path to input video file"
    echo "  output_video.mp4  Path to output vertical video"
    echo ""
    echo "Options:"
    echo "  --smoothing, -s <factor>    Smoothing factor (0-1, default: 0.5)"
    echo "  --lead-room, -l <factor>    Lead room factor (0-0.3, default: 0.15)"
    echo "  --output-height <height>    Output height (default: 1080)"
    echo "  --preview, -p               Show preview window during processing"
    echo "  --help, -h                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 results/demo/tennis_demo/ball_coordinates.json input.mp4 output.mp4"
    echo ""
    echo "  # With custom smoothing and lead room"
    echo "  $0 coords.json in.mp4 out.mp4 --smoothing 0.7 --lead-room 0.2"
    echo ""
    echo "  # With preview"
    echo "  $0 coords.json in.mp4 out.mp4 --preview"
    exit 0
}

# Check minimum arguments
if [ $# -lt 3 ]; then
    echo -e "${RED}Error: Not enough arguments${NC}"
    usage
fi

# Parse required arguments
COORDS_FILE="$1"
INPUT_VIDEO="$2"
OUTPUT_VIDEO="$3"
shift 3

# Default options
SMOOTHING="0.5"
LEAD_ROOM="0.15"
OUTPUT_HEIGHT="1080"
PREVIEW=""
EXTRA_ARGS=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoothing|-s)
            SMOOTHING="$2"
            shift 2
            ;;
        --lead-room|-l)
            LEAD_ROOM="$2"
            shift 2
            ;;
        --output-height)
            OUTPUT_HEIGHT="$2"
            shift 2
            ;;
        --preview|-p)
            PREVIEW="--preview"
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Validate inputs and convert to absolute paths
if [ ! -f "$COORDS_FILE" ]; then
    echo -e "${RED}Error: Coordinates file not found: $COORDS_FILE${NC}"
    exit 1
fi

if [ ! -f "$INPUT_VIDEO" ]; then
    echo -e "${RED}Error: Video file not found: $INPUT_VIDEO${NC}"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Convert to absolute paths (before changing directory)
COORDS_FILE="$(cd "$(dirname "$COORDS_FILE")" && pwd)/$(basename "$COORDS_FILE")"
INPUT_VIDEO="$(cd "$(dirname "$INPUT_VIDEO")" && pwd)/$(basename "$INPUT_VIDEO")"

# For output, just get the absolute directory if it exists, or use project root
if [ -d "$(dirname "$OUTPUT_VIDEO")" ]; then
    OUTPUT_VIDEO="$(cd "$(dirname "$OUTPUT_VIDEO")" && pwd)/$(basename "$OUTPUT_VIDEO")"
else
    OUTPUT_VIDEO="$PROJECT_ROOT/$OUTPUT_VIDEO"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Vertical Video Crop Generator${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Coordinates:  $COORDS_FILE"
echo -e "Input video:  $INPUT_VIDEO"
echo -e "Output video: $OUTPUT_VIDEO"
echo -e "Smoothing:    $SMOOTHING"
echo -e "Lead room:    $LEAD_ROOM"
echo -e "Output size:  ${OUTPUT_HEIGHT}x$((OUTPUT_HEIGHT * 9 / 16))"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if virtual environment exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Navigate to src directory
cd "$PROJECT_ROOT/src"

# Run the vertical video generator
echo -e "${YELLOW}Generating vertical video...${NC}"
echo ""

python post_process/ball_crop_video.py \
    --coords "$COORDS_FILE" \
    --video "$INPUT_VIDEO" \
    --output "$OUTPUT_VIDEO" \
    --smoothing "$SMOOTHING" \
    --lead-room "$LEAD_ROOM" \
    --output-height "$OUTPUT_HEIGHT" \
    $PREVIEW \
    $EXTRA_ARGS

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Output saved to: ${YELLOW}$OUTPUT_VIDEO${NC}"
echo -e "${GREEN}========================================${NC}"
