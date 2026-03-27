#!/bin/bash
# =============================================================================
# TOTNet Football Training Script for Cloud GPU Environments
# =============================================================================
# 
# Usage:
#   ./scripts/train_football_cloud.sh [OPTIONS]
#
# Options:
#   --epochs N         Number of training epochs (default: 30)
#   --batch-size N     Batch size (default: 16)
#   --lr RATE          Learning rate (default: 5e-4)
#   --gpus N           Number of GPUs (default: 1)
#   --pretrained PATH  Path to pretrained weights
#   --resume PATH      Path to checkpoint to resume from
#   --no-pretrained    Don't use pretrained weights (train from scratch)
#   --no-uv            Use pip instead of uv for dependency management
#   --dry-run          Show command without executing
#
# Examples:
#   # Single GPU training with pretrained weights (default, uses uv)
#   ./scripts/train_football_cloud.sh
#
#   # Multi-GPU distributed training
#   ./scripts/train_football_cloud.sh --gpus 2
#
#   # Resume from checkpoint
#   ./scripts/train_football_cloud.sh --resume checkpoints/TOTNet_Football/TOTNet_Football_epoch_10.pth
#
#   # Use pip instead of uv
#   ./scripts/train_football_cloud.sh --no-uv
#
# =============================================================================

set -e  # Exit on error

# Default values
EPOCHS=30
BATCH_SIZE=16
LR=5e-4
GPUS=1
NUM_FRAMES=5
IMG_SIZE="288 512"
OCCLUDED_PROB=0.1
BALL_SIZE=4
VAL_SIZE=0.2
SEED=2024
PRETRAINED=""
RESUME=""
NO_PRETRAINED=false
DRY_RUN=false
USE_UV=true

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --pretrained)
            PRETRAINED="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --no-pretrained)
            NO_PRETRAINED=true
            shift
            ;;
        --no-uv)
            USE_UV=false
            shift
            ;;
        --football-dataset-dir)
            FOOTBALL_DATASET_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -32 "$0" | tail -30
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if uv is installed
if [ "$USE_UV" = true ]; then
    if ! command -v uv &> /dev/null; then
        echo "Warning: uv not found, falling back to pip"
        USE_UV=false
    fi
fi

# Install dependencies
echo "=============================================="
echo "Installing dependencies..."
echo "=============================================="

if [ "$USE_UV" = true ]; then
    echo "Using uv to install dependencies..."
    uv sync
else
    echo "Using pip to install dependencies..."
    pip install -r requirements.txt
fi

echo ""

# Find pretrained weights if not specified
if [ -z "$PRETRAINED" ] && [ "$NO_PRETRAINED" = false ]; then
    # Look for tennis pretrained weights
    TENNIS_WEIGHTS=$(find weights -name "*Tennis*best.pth" 2>/dev/null | head -1)
    if [ -n "$TENNIS_WEIGHTS" ]; then
        PRETRAINED="$TENNIS_WEIGHTS"
        echo "Found pretrained weights: $PRETRAINED"
    fi
fi

# Build the training command
CMD=""

if [ "$USE_UV" = true ]; then
    if [ "$GPUS" -gt 1 ]; then
        # Multi-GPU distributed training with torchrun
        CMD="uv run torchrun --nproc_per_node=$GPUS src/main.py"
        CMD="$CMD --distributed"
        CMD="$CMD --multiprocessing_distributed"
        CMD="$CMD --dist_url 'env://'"
        CMD="$CMD --dist_backend 'nccl'"
    else
        # Single GPU training
        CMD="uv run python src/main.py"
    fi
else
    if [ "$GPUS" -gt 1 ]; then
        CMD="torchrun --nproc_per_node=$GPUS src/main.py"
        CMD="$CMD --distributed"
        CMD="$CMD --multiprocessing_distributed"
        CMD="$CMD --dist_url 'env://'"
        CMD="$CMD --dist_backend 'nccl'"
    else
        CMD="python src/main.py"
    fi
fi

# Add common arguments
CMD="$CMD --num_epochs $EPOCHS"
CMD="$CMD --saved_fn 'TOTNet_Football'"
CMD="$CMD --num_frames $NUM_FRAMES"
CMD="$CMD --optimizer_type adamw"
CMD="$CMD --lr $LR"
CMD="$CMD --loss_function WBCE"
CMD="$CMD --weight_decay 5e-5"
CMD="$CMD --img_size $IMG_SIZE"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --print_freq 100"
CMD="$CMD --dataset_choice 'football'"
CMD="$CMD --model_choice 'TOTNet'"
CMD="$CMD --weighting_list 1 2 2 3"
CMD="$CMD --occluded_prob $OCCLUDED_PROB"
CMD="$CMD --ball_size $BALL_SIZE"
CMD="$CMD --val-size $VAL_SIZE"
CMD="$CMD --seed $SEED"
CMD="$CMD --no_test"

# Add pretrained weights if available
if [ -n "$PRETRAINED" ] && [ "$NO_PRETRAINED" = false ]; then
    CMD="$CMD --pretrained_path '$PRETRAINED'"
fi

# Add resume checkpoint if specified
if [ -n "$RESUME" ]; then
    CMD="$CMD --resume '$RESUME'"
fi

# Print configuration
echo "=============================================="
echo "TOTNet Football Training"
echo "=============================================="
echo "Project dir:    $PROJECT_DIR"
echo "GPUs:           $GPUS"
echo "Epochs:         $EPOCHS"
echo "Batch size:     $BATCH_SIZE"
echo "Learning rate:  $LR"
echo "Image size:     $IMG_SIZE"
echo "Num frames:     $NUM_FRAMES"
echo "Pretrained:     ${PRETRAINED:-None (training from scratch)}"
echo "Resume:         ${RESUME:-None}"
echo "Using uv:       $USE_UV"
echo "=============================================="

# Execute or print command
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Command (dry run):"
    echo "$CMD"
else
    echo ""
    echo "Starting training..."
    echo ""
    eval $CMD
fi
