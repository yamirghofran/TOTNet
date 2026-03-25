import os
import sys
from collections import deque
import subprocess
import json

import cv2
import numpy as np
import torch
import time

sys.path.append('./')

from data_process.video_loader import Video_Loader
from data_process.folder_loader import Folder_Loader
from model.model_utils import load_pretrained_model
from model.TOTNet import build_motion_model_light
from model.TOTNet_OF import build_motion_model_light_opticalflow
from model.tracknet import build_TrackNetV2
from model.wasb import build_wasb
from config.config import parse_configs
from utils.misc import time_synchronized
from losses_metrics.metrics import extract_coords2d


class BallCoordinateLogger:
    """
    Logs ball coordinates during inference and saves to JSON file.
    Used for post-processing operations like vertical video cropping.
    
    Note: Coordinates are logged in MODEL INPUT space (the resolution used for inference),
    not in original video space. The JSON stores both dimensions for proper scaling.
    """
    
    def __init__(self, original_video_w, original_video_h, fps, model_input_w, model_input_h):
        """
        Initialize the coordinate logger.
        
        Args:
            original_video_w: Original video width (e.g., 1920)
            original_video_h: Original video height (e.g., 1080)
            fps: Video frames per second
            model_input_w: Model input width (what coordinates are in, e.g., 512)
            model_input_h: Model input height (what coordinates are in, e.g., 288)
        """
        self.video_info = {
            "original_width": original_video_w,
            "original_height": original_video_h,
            "model_input_width": model_input_w,
            "model_input_height": model_input_h,
            "fps": fps,
            "total_frames": 0
        }
        self.coordinates = []
    
    def log(self, frame_idx, x, y, confidence=1.0):
        """
        Log a ball coordinate.
        
        Args:
            frame_idx: Frame number
            x: X coordinate (in model input space)
            y: Y coordinate (in model input space)
            confidence: Detection confidence (default 1.0)
        """
        self.coordinates.append({
            "frame": frame_idx,
            "x": float(x),
            "y": float(y),
            "confidence": float(confidence)
        })
    
    def save(self, output_path):
        """
        Save coordinates to JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        self.video_info["total_frames"] = len(self.coordinates)
        data = {
            "video_info": self.video_info,
            "coordinates": self.coordinates
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Ball coordinates saved to: {output_path}")


def get_device(gpu_idx=None):
    """Get the best available device: CUDA, MPS (Mac), or CPU."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_idx}' if gpu_idx is not None else 'cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon) device")
        return device
    else:
        device = torch.device('cpu')
        print("Using CPU device")
        return device


def demo(configs):

    # Auto-detect input type: video file or folder of images
    if os.path.isfile(configs.video_path):
        # Input is a video file
        print(f"Loading video from: {configs.video_path}")
        data_loader = Video_Loader(configs.video_path, configs.img_size, configs.num_frames)
    elif os.path.isdir(configs.video_path):
        # Input is a folder of images
        print(f"Loading images from folder: {configs.video_path}")
        data_loader = Folder_Loader(configs.video_path, configs.img_size, configs.num_frames)
    else:
        raise FileNotFoundError(f"Input path not found: {configs.video_path}")
    

    if configs.save_demo_output:
        configs.frame_dir = os.path.join(configs.save_demo_dir, 'frame')
        if not os.path.isdir(configs.frame_dir):
            os.makedirs(configs.frame_dir)

    # Get the best available device (CUDA, MPS, or CPU)
    configs.device = get_device(configs.gpu_idx)
    
    # Initialize ball coordinate logger for post-processing
    # Get video dimensions - different attributes for Video_Loader vs Folder_Loader
    if isinstance(data_loader, Video_Loader):
        video_w = data_loader.video_w
        video_h = data_loader.video_h
        video_fps = data_loader.video_fps
    else:
        # For Folder_Loader, use config values or defaults
        video_w = configs.img_size[1]  # width from config
        video_h = configs.img_size[0]  # height from config
        video_fps = 30  # default fps for image sequences
    
    # IMPORTANT: Coordinates are in MODEL INPUT space (configs.img_size)
    # We save the model input dimensions so scaling is correct later
    model_input_w = configs.img_size[1]  # width (e.g., 512)
    model_input_h = configs.img_size[0]  # height (e.g., 288)
    
    coord_logger = BallCoordinateLogger(
        original_video_w=video_w,
        original_video_h=video_h,
        fps=video_fps,
        model_input_w=model_input_w,
        model_input_h=model_input_h
    )

    # Model
    if configs.model_choice in ['motion_light', 'TOTNet']:
        model = build_motion_model_light(configs)
    elif configs.model_choice == 'wasb':
        model = build_wasb(configs)
    elif configs.model_choice in ['motion_light_opticalflow', 'TOTNet_OF']:
        print("Building Motion Light Optical Flow model...")
        model = build_motion_model_light_opticalflow(configs)
    elif configs.model_choice == 'tracknetv2':
        model = build_TrackNetV2(configs)
    else:
        raise ValueError(f"Unknown model choice: {configs.model_choice}")
    
    # Move model to device (not just .cuda())
    model = model.to(configs.device)


    assert configs.pretrained_path is not None, "Need to load the pre-trained model"
    try:
        model = load_pretrained_model(model, configs.pretrained_path, configs.device)
        print(f"Model loaded successfully from {configs.pretrained_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

    model.eval()
    frame_idx = int(configs.num_frames - 1)

    with torch.no_grad():
        for item in data_loader:
            # Handle different loader return formats
            if len(item) == 4:
                count, resized_imgs, current_frame, original_frame = item
            else:
                count, resized_imgs, current_frame = item
            
            resized_imgs = torch.from_numpy(resized_imgs.astype(np.float32)).to(configs.device, non_blocking=True).unsqueeze(0)
            batched_data = resized_imgs
            t1 = time.time()

            if configs.model_choice in ['wasb', 'tracknetv2']:
                B, N, C, H, W = batched_data.shape
                # Permute to bring frames and channels together
                batched_data = batched_data.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
                # Reshape to combine frames into the channel dimension
                batched_data = batched_data.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]
   
            heatmap_output = model(batched_data)
            t2 = time.time()
          

            # Extract coordinates from heatmap
            H, W = configs.img_size[0], configs.img_size[1]
            post_processed_coord = extract_coords2d(heatmap_output, H, W)
            
            x_pred, y_pred = post_processed_coord[0][0], post_processed_coord[0][1]
            ball_pos = (int(x_pred), int(y_pred))  # Ensure integer coordinates
            print(ball_pos)
            
            # Log ball coordinates for post-processing (vertical crop, etc.)
            coord_logger.log(frame_idx, x_pred, y_pred)

            events = (0.0, 0.0)  # TOTNet doesn't predict events

            ploted_img = plot_detection(current_frame.copy(), ball_pos, events)

            ploted_img = cv2.cvtColor(ploted_img, cv2.COLOR_RGB2BGR)
            if configs.show_image:
                cv2.imshow('ploted_img.png', ploted_img)
                time.sleep(0.01)
            if configs.save_demo_output:
                cv2.imwrite(os.path.join(configs.frame_dir, '{:06d}.jpg'.format(frame_idx)), ploted_img)

            if count == 3000:
                break

            frame_idx += 1
            print('Done frame_idx {} - time {:.3f}s'.format(frame_idx, t2 - t1))

    if configs.output_format == 'video':
        output_video_path = os.path.join(configs.save_demo_dir, 'result.mp4')
        frames_dir = configs.frame_dir

        print(f"Frames directory: {frames_dir}")
        if not os.path.isdir(frames_dir):
            print(f"Error: Frame directory {frames_dir} does not exist!")
        else:
            print(f"Frame files: {os.listdir(frames_dir)}")

        if not os.path.isdir(configs.save_demo_dir):
            print(f"Output directory does not exist. Creating: {configs.save_demo_dir}")
            os.makedirs(configs.save_demo_dir)

        cmd_str = f'ffmpeg -f image2 -i {frames_dir}/%06d.jpg -b:v 5000k -c:v mpeg4 {output_video_path}'
        print(f"Running ffmpeg command: {cmd_str}")
        process = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)

        if process.returncode != 0:
            print("Error: ffmpeg command failed.")
            print(f"ffmpeg stdout: {process.stdout}")
            print(f"ffmpeg stderr: {process.stderr}")
        else:
            if os.path.isfile(output_video_path):
                print(f"Video saved at: {output_video_path}")
            else:
                print(f"Error: Video file not found at {output_video_path}")
    
    # Save ball coordinates to JSON file for post-processing
    if configs.save_demo_output:
        coords_path = os.path.join(configs.save_demo_dir, 'ball_coordinates.json')
        coord_logger.save(coords_path)


def plot_detection(img, ball_pos, events):
    """Show the predicted information in the image"""
    if not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")
    img = img.astype(np.uint8)
    if ball_pos != (0, 0):
        img = cv2.circle(img, ball_pos, 5, (255, 0, 255), -1)
    # event_name = 'is bounce: {:.2f}, is net: {:.2f}'.format(events[0], events[1])
    # img = cv2.putText(img, event_name, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return img



if __name__ == '__main__':
    configs = parse_configs()
    demo(configs=configs)
