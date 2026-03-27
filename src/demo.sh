#!/bin/bash
# Football Demo - TOTNet

# Enable MPS fallback for unsupported ops on Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

python demo.py \
    --save_demo_output \
    --output_format video \
    --model_choice 'motion_light' \
    --num_frames 5 \
    --dataset_choice football \
    --video_path '../data/football_dataset/frames/football-' \
    --pretrained_path '../weights/TOTNET_Football/TOTNet_Football_best.pth' \
    --saved_fn football_demo
