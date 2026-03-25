"""
Post-processing utilities for ball tracking results.

This module provides tools for:
- Smoothing ball trajectories with cubic spline interpolation
- Generating vertical (9:16) video crops that follow the ball
- Interpolating missing ball detections
"""

from .smoothing import (
    BallTrajectorySmoother,
    CropWindowCalculator,
    TrajectoryData
)

__all__ = [
    'BallTrajectorySmoother',
    'CropWindowCalculator',
    'TrajectoryData',
]
