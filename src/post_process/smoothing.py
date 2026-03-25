"""
Ball trajectory smoothing utilities using cubic spline interpolation.

This module provides tools for:
- Loading ball coordinates from JSON
- Interpolating missing detections (where ball was not found)
- Applying cubic spline smoothing for smooth camera movement
- Calculating ball velocity for lead room computation
"""

import json
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TrajectoryData:
    """Container for trajectory data."""
    frames: np.ndarray
    coords: np.ndarray  # Shape: (N, 2) for (x, y)
    velocities: np.ndarray  # Shape: (N, 2) for (vx, vy)
    video_info: Dict[str, Any]


class BallTrajectorySmoother:
    """
    Smooths ball trajectory using cubic spline interpolation with
    handling for missing detections.
    """
    
    def __init__(self, smoothing_factor: float = 0.5):
        """
        Args:
            smoothing_factor: 0.0 = exact fit, 1.0 = very smooth
        """
        self.smoothing_factor = smoothing_factor
        self.video_info = {}  # Store video info for edge detection
    
    def load_coordinates(self, json_path: str) -> Dict[str, Any]:
        """Load ball coordinates from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.video_info = data.get('video_info', {})
        return data
    
    def _is_valid_detection(self, x: float, y: float) -> bool:
        """
        Check if a coordinate represents a valid ball detection.
        
        Invalid detections:
        - (0, 0) - ball not in frame
        - Edge coordinates like (511, 0) when model input is 512x288
        """
        # Get model input dimensions
        model_w = self.video_info.get('model_input_width', 512)
        model_h = self.video_info.get('model_input_height', 288)
        
        # (0, 0) is invalid
        if x == 0 and y == 0:
            return False
        
        # Edge coordinates are invalid (model couldn't find ball)
        # x at right edge, y at 0 is a common "not found" pattern
        if x >= model_w - 1 and y == 0:
            return False
        
        return True
    
    def interpolate_missing(
        self, 
        frames: np.ndarray, 
        coords: np.ndarray,
        method: str = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate coordinates where ball was not detected.
        """
        # Find valid detections
        valid_mask = np.array([
            self._is_valid_detection(c[0], c[1]) for c in coords
        ])
        
        valid_frames = frames[valid_mask]
        valid_coords = coords[valid_mask]
        
        num_valid = len(valid_frames)
        if num_valid < 2:
            raise ValueError(
                f"Not enough valid detections ({num_valid}) to interpolate. "
                "Need at least 2 valid ball positions."
            )
        
        print(f"Valid detections: {num_valid}/{len(frames)} ({100*num_valid/len(frames):.1f}%)")
        
        # Choose interpolation method based on available points
        if method == 'cubic' and num_valid < 4:
            method = 'linear'
            print(f"Warning: Only {num_valid} valid points, using linear interpolation")
        
        # Create interpolation functions
        interp_x = interp1d(valid_frames, valid_coords[:, 0], 
                           kind=method, fill_value='extrapolate')
        interp_y = interp1d(valid_frames, valid_coords[:, 1], 
                           kind=method, fill_value='extrapolate')
        
        # Generate interpolated coordinates for all frames
        interpolated_coords = np.column_stack([
            interp_x(frames),
            interp_y(frames)
        ])
        
        return frames, interpolated_coords
    
    def smooth_trajectory(
        self, 
        frames: np.ndarray, 
        coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply smoothing to trajectory using a simple moving average.
        This is more stable than spline smoothing for avoiding lag.
        """
        # Use simple exponential smoothing to avoid lag
        smoothed = coords.copy()
        
        # Apply light smoothing with a small window
        window = max(3, int(len(frames) * self.smoothing_factor * 0.05))
        window = min(window, 7)  # Cap at 7 frames
        
        if window > 1:
            kernel = np.ones(window) / window
            # Pad edges
            padded_x = np.pad(coords[:, 0], window//2, mode='edge')
            padded_y = np.pad(coords[:, 1], window//2, mode='edge')
            
            smoothed[:, 0] = np.convolve(padded_x, kernel, mode='valid')[:len(frames)]
            smoothed[:, 1] = np.convolve(padded_y, kernel, mode='valid')[:len(frames)]
        
        return frames, smoothed
    
    def calculate_velocity(
        self, 
        frames: np.ndarray, 
        coords: np.ndarray,
        smooth_window: int = 3
    ) -> np.ndarray:
        """Calculate ball velocity."""
        velocities = np.zeros_like(coords)
        
        # Calculate raw velocity
        velocities[1:] = coords[1:] - coords[:-1]
        velocities[0] = velocities[1]
        
        # Light smoothing
        if smooth_window > 1:
            velocities[:, 0] = gaussian_filter1d(velocities[:, 0], sigma=smooth_window/2)
            velocities[:, 1] = gaussian_filter1d(velocities[:, 1], sigma=smooth_window/2)
        
        return velocities
    
    def process(
        self, 
        json_path: str,
        interpolate_method: str = 'linear'
    ) -> TrajectoryData:
        """
        Full processing pipeline: load → interpolate → smooth → velocity.
        """
        # Load
        data = self.load_coordinates(json_path)
        coords_list = data['coordinates']
        video_info = data['video_info']
        
        frames = np.array([c['frame'] for c in coords_list])
        coords = np.array([[c['x'], c['y']] for c in coords_list])
        
        print(f"Loaded {len(frames)} frames from {json_path}")
        
        # Interpolate missing
        frames, coords = self.interpolate_missing(frames, coords, interpolate_method)
        
        # Smooth
        frames, coords = self.smooth_trajectory(frames, coords)
        print(f"Applied smoothing with factor {self.smoothing_factor}")
        
        # Calculate velocities
        velocities = self.calculate_velocity(frames, coords)
        
        return TrajectoryData(
            frames=frames,
            coords=coords,
            velocities=velocities,
            video_info=video_info
        )


class CropWindowCalculator:
    """
    Calculates crop window positions for vertical video following the ball.
    """
    
    def __init__(
        self,
        input_width: int,
        input_height: int,
        aspect_ratio: Tuple[int, int] = (9, 16),
        lead_room_factor: float = 0.0  # Disabled by default
    ):
        """
        Initialize the crop calculator.
        
        Args:
            input_width: Width of the INPUT video
            input_height: Height of the INPUT video
            aspect_ratio: (width_ratio, height_ratio) for crop (default 9:16)
            lead_room_factor: Set to 0 for centered ball
        """
        self.input_width = input_width
        self.input_height = input_height
        self.aspect_ratio = aspect_ratio
        self.lead_room_factor = lead_room_factor
        
        # Calculate crop dimensions to fit within input video
        width_ratio, height_ratio = aspect_ratio
        
        # For 9:16 vertical: use full height, calculate width
        crop_w_from_height = int(input_height * width_ratio / height_ratio)
        
        if crop_w_from_height <= input_width:
            self.crop_height = input_height
            self.crop_width = crop_w_from_height
        else:
            self.crop_width = input_width
            self.crop_height = int(input_width * height_ratio / width_ratio)
        
        print(f"Input video: {input_width}x{input_height}")
        print(f"Crop dimensions: {self.crop_width}x{self.crop_height}")
    
    def calculate_crop_position(
        self,
        ball_x: float,
        ball_y: float,
        video_width: int,
        video_height: int
    ) -> Dict[str, int]:
        """Calculate crop window centered on ball."""
        # Center crop on ball
        crop_x = ball_x - self.crop_width / 2
        crop_y = ball_y - self.crop_height / 2
        
        # Bounds checking
        crop_x = max(0, min(int(crop_x), video_width - self.crop_width))
        crop_y = max(0, min(int(crop_y), video_height - self.crop_height))
        
        return {
            'x': crop_x,
            'y': crop_y,
            'w': self.crop_width,
            'h': self.crop_height
        }
    
    def calculate_all_crops(
        self,
        trajectory: TrajectoryData,
        video_width: int,
        video_height: int
    ) -> List[Dict]:
        """Calculate crop windows for all frames."""
        crop_windows = []
        
        for frame_idx, (ball_x, ball_y) in zip(trajectory.frames, trajectory.coords):
            crop_pos = self.calculate_crop_position(
                ball_x, ball_y, video_width, video_height
            )
            crop_pos['frame'] = int(frame_idx)
            crop_windows.append(crop_pos)
        
        return crop_windows
