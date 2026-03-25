"""
Generate 9:16 vertical video with ball-following crop.

This script creates a vertical (9:16 aspect ratio) video that follows
the ball trajectory with smooth camera movement and dynamic lead room.

Usage:
    python ball_crop_video.py \
        --coords results/demo/tennis_demo/ball_coordinates.json \
        --video input.mp4 \
        --output output_vertical.mp4 \
        --smoothing 0.5 \
        --lead-room 0.15
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Optional
from dataclasses import asdict

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from post_process.smoothing import (
    BallTrajectorySmoother,
    CropWindowCalculator,
    TrajectoryData
)


class VerticalVideoGenerator:
    """
    Generates a 9:16 vertical video that follows the ball with smooth movement.
    
    Features:
    - Cubic spline smoothing for camera movement
    - Dynamic lead room based on ball velocity
    - Interpolation of missing ball detections
    - 1080x1920 Full HD vertical output
    """
    
    def __init__(
        self,
        output_height: int = 1080,
        output_width: Optional[int] = None,
        lead_room_factor: float = 0.15,
        smoothing_factor: float = 0.5,
        max_velocity_threshold: float = 100.0,
        interpolation_method: str = 'cubic'
    ):
        """
        Initialize the vertical video generator.
        
        Args:
            output_height: Output video height (default: 1080 for Full HD)
            output_width: Output video width (default: auto-calculated for 9:16)
            lead_room_factor: How much to lead the ball (0.15 = 15% of frame)
            smoothing_factor: Trajectory smoothing (0 = exact, 1 = very smooth)
            max_velocity_threshold: Max expected velocity for lead room calculation
            interpolation_method: Method for interpolating missing detections
        """
        self.output_height = output_height
        self.output_width = output_width or int(output_height * 9 / 16)
        self.lead_room_factor = lead_room_factor
        self.smoothing_factor = smoothing_factor
        self.max_velocity_threshold = max_velocity_threshold
        self.interpolation_method = interpolation_method
        
        print(f"Output resolution: {self.output_width}x{self.output_height}")
    
    def _get_video_info(self, video_path: str) -> Dict:
        """Get video metadata using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    
    def _scale_coordinates_to_original(
        self,
        trajectory: TrajectoryData,
        original_width: int,
        original_height: int
    ) -> TrajectoryData:
        """
        Scale coordinates from model input space to original video space.
        
        The ball coordinates are in the model's input resolution (e.g., 512x288).
        We need to scale them to the original video resolution for cropping.
        """
        # Get the dimensions used during inference
        # Try new format first, fall back to old format
        model_width = trajectory.video_info.get('model_input_width') or trajectory.video_info.get('width', 512)
        model_height = trajectory.video_info.get('model_input_height') or trajectory.video_info.get('height', 288)
        
        # Calculate scale factors
        scale_x = original_width / model_width
        scale_y = original_height / model_height
        
        # Scale coordinates
        scaled_coords = trajectory.coords.copy()
        scaled_coords[:, 0] *= scale_x  # Scale x
        scaled_coords[:, 1] *= scale_y  # Scale y
        
        # Scale velocities too
        scaled_velocities = trajectory.velocities.copy()
        scaled_velocities[:, 0] *= scale_x
        scaled_velocities[:, 1] *= scale_y
        
        # Create new trajectory with scaled values
        return TrajectoryData(
            frames=trajectory.frames,
            coords=scaled_coords,
            velocities=scaled_velocities,
            video_info={
                **trajectory.video_info,
                'width': original_width,
                'height': original_height
            }
        )
    
    def process(
        self,
        coords_json_path: str,
        input_video_path: str,
        output_video_path: str,
        codec: str = 'mp4v',
        show_progress: bool = True,
        preview: bool = False
    ) -> str:
        """
        Main processing pipeline.
        
        Args:
            coords_json_path: Path to ball_coordinates.json
            input_video_path: Path to input video file
            output_video_path: Path to output vertical video
            codec: Video codec to use (default: mp4v)
            show_progress: Print progress messages
            preview: Show preview window during processing
            
        Returns:
            Path to output video
        """
        # Step 1: Get input video info
        video_info = self._get_video_info(input_video_path)
        if show_progress:
            print(f"Input video: {video_info['width']}x{video_info['height']} "
                  f"@ {video_info['fps']}fps, {video_info['duration']:.1f}s")
        
        # Step 2: Load and smooth trajectory
        if show_progress:
            print("Loading and smoothing ball trajectory...")
        
        smoother = BallTrajectorySmoother(smoothing_factor=self.smoothing_factor)
        trajectory = smoother.process(coords_json_path, self.interpolation_method)
        
        # Step 3: Scale coordinates to original video space
        trajectory = self._scale_coordinates_to_original(
            trajectory,
            video_info['width'],
            video_info['height']
        )
        
        # Step 4: Calculate crop windows based on INPUT video dimensions
        # The crop is a 9:16 region that slides around the input video
        if show_progress:
            print("Calculating crop windows...")
        
        crop_calculator = CropWindowCalculator(
            input_width=video_info['width'],
            input_height=video_info['height'],
            aspect_ratio=(9, 16),
            lead_room_factor=0.0  # Centered ball, no lead room
        )
        
        # Store crop dimensions for use in video generation
        self.crop_width = crop_calculator.crop_width
        self.crop_height = crop_calculator.crop_height
        
        crop_windows = crop_calculator.calculate_all_crops(
            trajectory,
            video_info['width'],
            video_info['height']
        )
        
        # Step 5: Generate output video
        if show_progress:
            print(f"Generating vertical video ({self.output_width}x{self.output_height})...")
        
        output_path = self._generate_video(
            input_video_path,
            output_video_path,
            crop_windows,
            video_info['fps'],
            codec,
            show_progress,
            preview
        )
        
        if show_progress:
            print(f"\nOutput saved to: {output_path}")
        
        return output_path
    
    def _generate_video(
        self,
        input_path: str,
        output_path: str,
        crop_windows: List[Dict],
        fps: int,
        codec: str,
        show_progress: bool,
        preview: bool
    ) -> str:
        """
        Generate the output video by applying crops frame by frame.
        
        Note: crop_windows contains entries with 'frame' field indicating which
        video frame they correspond to. We must match by frame number, not by
        list index.
        """
        cap = cv2.VideoCapture(input_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                              (self.output_width, self.output_height))
        
        if not out.isOpened():
            raise RuntimeError(f"Could not create video writer for {output_path}")
        
        # Create a lookup dict: frame_number -> crop_window
        # This handles the case where frame indices don't start at 0
        crop_lookup = {cw['frame']: cw for cw in crop_windows}
        
        # Get the range of frames we have crop data for
        min_frame = min(crop_lookup.keys()) if crop_lookup else 0
        max_frame = max(crop_lookup.keys()) if crop_lookup else 0
        
        # Default center crop for frames without data
        h_img_default = self.crop_height
        w_img_default = self.crop_width
        
        frame_idx = 0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h_img, w_img = frame.shape[:2]
                
                # Look up crop window for this specific frame
                if frame_idx in crop_lookup:
                    cw = crop_lookup[frame_idx]
                    x, y, w, h = cw['x'], cw['y'], cw['w'], cw['h']
                    
                    # Ensure crop is within bounds
                    x = max(0, min(x, w_img - w))
                    y = max(0, min(y, h_img - h))
                    
                    cropped = frame[y:y+h, x:x+w]
                else:
                    # No crop data for this frame - use center crop
                    crop_w = min(self.crop_width, w_img)
                    crop_h = min(self.crop_height, h_img)
                    x = (w_img - crop_w) // 2
                    y = (h_img - crop_h) // 2
                    
                    cropped = frame[y:y+crop_h, x:x+crop_w]
                
                # Resize to output dimensions
                resized = cv2.resize(
                    cropped,
                    (self.output_width, self.output_height),
                    interpolation=cv2.INTER_LANCZOS4
                )
                
                # Write frame
                out.write(resized)
                
                # Show preview if requested
                if preview:
                    cv2.imshow('Vertical Preview', resized)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                frame_idx += 1
                if show_progress and frame_idx % 100 == 0:
                    progress = frame_idx / total_video_frames * 100
                    print(f"  Progress: {frame_idx}/{total_video_frames} frames ({progress:.1f}%)")
        
        finally:
            cap.release()
            out.release()
            if preview:
                cv2.destroyAllWindows()
        
        return output_path
    
    def save_crop_debug_info(
        self,
        output_path: str,
        crop_windows: List[Dict]
    ):
        """Save crop window information to JSON for debugging."""
        with open(output_path, 'w') as f:
            json.dump({
                'crop_windows': [{k: int(v) if isinstance(v, (int, np.integer)) else v 
                                  for k, v in cw.items()} 
                                 for cw in crop_windows]
            }, f, indent=2)
        print(f"Crop debug info saved to: {output_path}")


def main():
    """Command-line interface for vertical video generation."""
    parser = argparse.ArgumentParser(
        description='Generate 9:16 vertical video following ball trajectory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python ball_crop_video.py --coords ball_coordinates.json --video input.mp4 --output output.mp4
  
  # With custom smoothing and lead room
  python ball_crop_video.py --coords coords.json --video in.mp4 --output out.mp4 \\
      --smoothing 0.7 --lead-room 0.2
  
  # With preview window
  python ball_crop_video.py --coords coords.json --video in.mp4 --output out.mp4 --preview
        """
    )
    
    parser.add_argument(
        '--coords', '-c',
        required=True,
        help='Path to ball_coordinates.json from demo.py'
    )
    parser.add_argument(
        '--video', '-v',
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Path to output vertical video file'
    )
    parser.add_argument(
        '--smoothing', '-s',
        type=float,
        default=0.5,
        help='Smoothing factor (0.0 = exact fit, 1.0 = very smooth). Default: 0.5'
    )
    parser.add_argument(
        '--lead-room', '-l',
        type=float,
        default=0.15,
        help='Lead room factor (0.0-0.3). Ball positioned with offset based on velocity. Default: 0.15'
    )
    parser.add_argument(
        '--output-height',
        type=int,
        default=1080,
        help='Output video height. Default: 1080 (Full HD)'
    )
    parser.add_argument(
        '--max-velocity',
        type=float,
        default=100.0,
        help='Maximum expected ball velocity for lead room calculation. Default: 100'
    )
    parser.add_argument(
        '--interpolation',
        choices=['cubic', 'linear', 'quadratic'],
        default='cubic',
        help='Interpolation method for missing detections. Default: cubic'
    )
    parser.add_argument(
        '--codec',
        default='mp4v',
        help='Output video codec. Default: mp4v'
    )
    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Show preview window during processing'
    )
    parser.add_argument(
        '--debug-crops',
        action='store_true',
        help='Save crop window info to JSON for debugging'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.coords):
        raise FileNotFoundError(f"Coordinates file not found: {args.coords}")
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Create generator and process
    generator = VerticalVideoGenerator(
        output_height=args.output_height,
        lead_room_factor=args.lead_room,
        smoothing_factor=args.smoothing,
        max_velocity_threshold=args.max_velocity,
        interpolation_method=args.interpolation
    )
    
    output_path = generator.process(
        coords_json_path=args.coords,
        input_video_path=args.video,
        output_video_path=args.output,
        codec=args.codec,
        show_progress=True,
        preview=args.preview
    )
    
    # Save debug info if requested
    if args.debug_crops:
        smoother = BallTrajectorySmoother(smoothing_factor=args.smoothing)
        trajectory = smoother.process(args.coords)
        
        # Get video info for debug
        import cv2
        cap = cv2.VideoCapture(args.video)
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        crop_calculator = CropWindowCalculator(
            input_width=video_w,
            input_height=video_h,
            lead_room_factor=args.lead_room
        )
        crop_windows = crop_calculator.calculate_all_crops(
            trajectory, video_w, video_h
        )
        
        debug_path = args.output.replace('.mp4', '_crops.json')
        generator.save_crop_debug_info(debug_path, crop_windows)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
