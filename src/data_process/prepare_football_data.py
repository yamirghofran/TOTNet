#!/usr/bin/env python3
"""
Prepare Football Dataset for TOTNet Training

This script converts ball-tracking-labeler augmented output to TOTNet format.

Usage:
    python prepare_football_data.py --source_dir /path/to/ball-tracking-labeler --output_dir ./data/football_dataset

Output structure:
    data/football_dataset/
    ├── frames/
    │   ├── football-0/
    │   │   ├── img_000000.jpg
    │   │   └── ...
    │   └── ...
    ├── train.json
    └── test.json
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

try:
    from PIL import Image
except ImportError:
    print("Installing Pillow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "--break-system-packages"])
    from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Fall back to no progress bar


def get_video_name_from_augmented_dir(dir_name: str) -> str:
    """Extract video name from augmented directory name.
    
    Examples:
        football-arsenal-0_augmented -> football-arsenal-0
        football-0_augmented -> football-0
    """
    if dir_name.endswith("_augmented"):
        return dir_name[:-11]  # Remove "_augmented"
    return dir_name


def find_augmented_directories(source_dir: str) -> List[Tuple[str, str]]:
    """Find all augmented directories in source.
    
    Returns:
        List of (directory_path, video_name) tuples
    """
    source_path = Path(source_dir)
    augmented_dirs = []
    
    for item in source_path.iterdir():
        if item.is_dir() and item.name.endswith("_augmented"):
            video_name = get_video_name_from_augmented_dir(item.name)
            augmented_dirs.append((str(item), video_name))
    
    # Sort by video name for reproducibility
    augmented_dirs.sort(key=lambda x: x[1])
    return augmented_dirs


def get_sequence_frame_mapping(sequence_dir: Path) -> Dict[int, int]:
    """Map frame indices in sequence to original source frames.
    
    Returns:
        Dict mapping local frame index (0-4) to source frame number
    """
    mapping = {}
    annotations_file = sequence_dir / "annotations.json"
    
    if annotations_file.exists():
        with open(annotations_file, 'r') as f:
            ann = json.load(f)
            source_frames = ann.get("source_frames", [])
            for local_idx, source_frame in enumerate(source_frames):
                mapping[local_idx] = source_frame
    
    # Fallback to sequential if no source_frames info
    if not mapping:
        for i in range(5):  # Assuming 5 frames per sequence
            mapping[i] = i
    
    return mapping


def get_image_resolution(sequence_dir: Path) -> Optional[Tuple[int, int]]:
    """Get resolution from first frame in sequence."""
    for frame_file in sorted(sequence_dir.glob("frame_*.png")):
        try:
            with Image.open(frame_file) as img:
                return img.size  # (width, height)
        except Exception:
            continue
    return None


def process_sequence(
    sequence_dir: Path,
    output_frames_dir: Path,
    video_name: str,
    resize: Optional[Tuple[int, int]] = None,
    copy_files: bool = True
) -> Optional[Dict]:
    """Process a single sequence and return annotation data.
    
    Args:
        sequence_dir: Path to sequence directory (e.g., sequence_00000/)
        output_frames_dir: Base output directory for frames
        video_name: Name of the source video
        resize: Optional (width, height) to resize images
        copy_files: Whether to copy image files (set False for dry run)
    
    Returns:
        Dict with frame annotations or None if invalid
    """
    annotations_file = sequence_dir / "annotations.json"
    
    if not annotations_file.exists():
        print(f"Warning: No annotations.json in {sequence_dir}")
        return None
    
    with open(annotations_file, 'r') as f:
        ann = json.load(f)
    
    # Get source frames mapping
    source_frames = ann.get("source_frames", list(range(5)))
    
    # Process ball positions
    ball_positions = ann.get("ball_pos", [])
    if not ball_positions:
        return None
    
    # Get original resolution
    orig_width, orig_height = None, None
    first_frame = sequence_dir / "frame_0.png"
    if first_frame.exists():
        with Image.open(first_frame) as img:
            orig_width, orig_height = img.size
    
    # Calculate resize scale if needed
    scale_x, scale_y = 1.0, 1.0
    if resize and orig_width and orig_height:
        scale_x = resize[0] / orig_width
        scale_y = resize[1] / orig_height
    
    # Process each frame
    processed_ball_pos = []
    video_frames_dir = output_frames_dir / video_name
    
    for bp in ball_positions:
        local_frame = bp["frame"]
        source_frame = source_frames[local_frame] if local_frame < len(source_frames) else local_frame
        
        # Source image path
        src_img = sequence_dir / f"frame_{local_frame}.png"
        if not src_img.exists():
            continue
        
        # Output image path (using source frame number for continuity)
        dst_img = video_frames_dir / f"img_{source_frame:06d}.jpg"
        
        # Copy/resize image
        if copy_files:
            video_frames_dir.mkdir(parents=True, exist_ok=True)
            if resize:
                with Image.open(src_img) as img:
                    # Convert to RGB (remove alpha channel if present)
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_resized = img.resize(resize, Image.LANCZOS)
                    img_resized.save(dst_img, "JPEG", quality=95)
            else:
                # Just copy and convert to JPEG
                with Image.open(src_img) as img:
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(dst_img, "JPEG", quality=95)
        
        # Scale coordinates if resizing
        ball_x = bp.get("ball_x")
        ball_y = bp.get("ball_y")
        
        if ball_x is not None and ball_y is not None:
            if resize:
                ball_x = ball_x * scale_x
                ball_y = ball_y * scale_y
        
        processed_ball_pos.append({
            "frame": source_frame,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "visibility": bp.get("visibility", 1)
        })
    
    return {
        "sequence_id": ann.get("sequence_id"),
        "source_video": video_name,
        "ball_pos": processed_ball_pos,
        "original_width": orig_width,
        "original_height": orig_height
    }


def process_video_augmented_dir(
    augmented_dir: str,
    video_name: str,
    output_frames_dir: Path,
    resize: Optional[Tuple[int, int]] = None,
    copy_files: bool = True
) -> List[Dict]:
    """Process all sequences in an augmented directory.
    
    Returns:
        List of sequence annotation dicts
    """
    aug_path = Path(augmented_dir)
    sequences = []
    
    # Find all sequence directories
    sequence_dirs = sorted(aug_path.glob("sequence_*"))
    
    print(f"  Processing {video_name}: {len(sequence_dirs)} sequences")
    
    # Use tqdm for progress if available
    iterator = tqdm(sequence_dirs, desc=f"  {video_name}", leave=False) if tqdm else sequence_dirs
    
    for seq_dir in iterator:
        result = process_sequence(
            seq_dir, output_frames_dir, video_name, resize, copy_files
        )
        if result:
            sequences.append(result)
    
    return sequences


def consolidate_ball_positions(sequences: List[Dict]) -> Dict[int, Dict]:
    """Consolidate ball positions across sequences for each video.
    
    Since sequences may overlap (sliding window), we need to merge
    ball positions for the same frame from different sequences.
    
    Returns:
        Dict mapping frame_num -> {ball_x, ball_y, visibility}
    """
    frame_data = {}
    
    for seq in sequences:
        for bp in seq["ball_pos"]:
            frame_num = bp["frame"]
            
            # If frame already exists, prefer higher visibility annotation
            # (visibility 1 > 2 > 3 > 0, where 1 is most visible)
            if frame_num in frame_data:
                existing = frame_data[frame_num]
                # Keep the annotation with better visibility (lower number, except 0)
                existing_vis = existing["visibility"]
                new_vis = bp["visibility"]
                
                # Priority: 1 > 2 > 3 > 0
                def vis_priority(v):
                    if v == 1:
                        return 0
                    elif v == 2:
                        return 1
                    elif v == 3:
                        return 2
                    else:
                        return 3  # visibility 0 (out of frame) is lowest priority
                
                if vis_priority(new_vis) < vis_priority(existing_vis):
                    frame_data[frame_num] = {
                        "ball_x": bp["ball_x"],
                        "ball_y": bp["ball_y"],
                        "visibility": bp["visibility"]
                    }
            else:
                frame_data[frame_num] = {
                    "ball_x": bp["ball_x"],
                    "ball_y": bp["ball_y"],
                    "visibility": bp["visibility"]
                }
    
    return frame_data


def create_video_entry(
    video_name: str,
    frame_data: Dict[int, Dict],
    width: int,
    height: int,
    video_id: int
) -> Dict:
    """Create a TOTNet-compatible video entry.
    
    Format matches TTA dataset format:
    {
        "video": "video_name",
        "id": 1,
        "width": 1280,
        "height": 720,
        "ball_pos": [...]
    }
    """
    ball_pos = []
    for frame_num in sorted(frame_data.keys()):
        data = frame_data[frame_num]
        ball_pos.append({
            "frame": frame_num,
            "ball_x": data["ball_x"],
            "ball_y": data["ball_y"],
            "visibility": data["visibility"]
        })
    
    return {
        "video": video_name,
        "id": video_id,
        "width": width,
        "height": height,
        "ball_pos": ball_pos
    }


def split_videos(
    video_entries: List[Dict],
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split video entries into train and test sets by video.
    
    Args:
        video_entries: List of video entry dicts
        test_ratio: Fraction of videos for testing
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_entries, test_entries)
    """
    random.seed(seed)
    
    # Shuffle video names
    videos = video_entries.copy()
    random.shuffle(videos)
    
    # Split
    split_idx = int(len(videos) * (1 - test_ratio))
    train_videos = videos[:split_idx]
    test_videos = videos[split_idx:]
    
    return train_videos, test_videos


def prepare_football_dataset(
    source_dir: str,
    output_dir: str,
    resize: Optional[Tuple[int, int]] = None,
    test_ratio: float = 0.2,
    seed: int = 42,
    copy_files: bool = True
):
    """Main function to prepare football dataset.
    
    Args:
        source_dir: Path to ball-tracking-labeler directory
        output_dir: Output directory for TOTNet-format dataset
        resize: Optional (width, height) to resize images
        test_ratio: Fraction of videos for testing
        seed: Random seed
        copy_files: Whether to copy files (False for dry run)
    """
    output_path = Path(output_dir)
    frames_dir = output_path / "frames"
    
    print("=" * 60)
    print("TOTNet Football Dataset Preparation")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Resize: {resize if resize else 'No resize'}")
    print(f"Test ratio: {test_ratio}")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    # Find all augmented directories
    augmented_dirs = find_augmented_directories(source_dir)
    
    if not augmented_dirs:
        print("Error: No augmented directories found!")
        return
    
    print(f"\nFound {len(augmented_dirs)} video sources:")
    for dir_path, video_name in augmented_dirs:
        print(f"  - {video_name}")
    
    # Detect resolution from first video
    first_aug_dir = Path(augmented_dirs[0][0])
    first_seq = next(first_aug_dir.glob("sequence_*"), None)
    orig_resolution = None
    if first_seq:
        orig_resolution = get_image_resolution(first_seq)
    
    if orig_resolution:
        print(f"\nDetected resolution: {orig_resolution[0]}x{orig_resolution[1]}")
        if resize:
            print(f"Will resize to: {resize[0]}x{resize[1]}")
            output_width, output_height = resize
        else:
            output_width, output_height = orig_resolution
    else:
        print("Warning: Could not detect resolution, using defaults")
        output_width, output_height = resize if resize else (1280, 720)
    
    # Process each video
    print("\n" + "-" * 60)
    print("Processing videos...")
    print("-" * 60)
    
    video_entries = []
    
    for aug_dir, video_name in augmented_dirs:
        sequences = process_video_augmented_dir(
            aug_dir, video_name, frames_dir, resize, copy_files
        )
        
        if not sequences:
            print(f"  Warning: No valid sequences for {video_name}")
            continue
        
        # Consolidate ball positions
        frame_data = consolidate_ball_positions(sequences)
        
        # Create video entry
        entry = create_video_entry(
            video_name, frame_data, output_width, output_height, len(video_entries) + 1
        )
        video_entries.append(entry)
        
        print(f"  {video_name}: {len(frame_data)} unique frames")
    
    # Split into train/test by video
    print("\n" + "-" * 60)
    print("Splitting train/test by video...")
    print("-" * 60)
    
    train_entries, test_entries = split_videos(video_entries, test_ratio, seed)
    
    print(f"\nTrain videos ({len(train_entries)}):")
    for entry in train_entries:
        print(f"  - {entry['video']}: {len(entry['ball_pos'])} frames")
    
    print(f"\nTest videos ({len(test_entries)}):")
    for entry in test_entries:
        print(f"  - {entry['video']}: {len(entry['ball_pos'])} frames")
    
    # Save JSON files
    if copy_files:
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_json = output_path / "train.json"
        test_json = output_path / "test.json"
        
        with open(train_json, 'w') as f:
            json.dump(train_entries, f, indent=2)
        print(f"\nSaved: {train_json}")
        
        with open(test_json, 'w') as f:
            json.dump(test_entries, f, indent=2)
        print(f"Saved: {test_json}")
    
    # Summary
    total_train_frames = sum(len(e['ball_pos']) for e in train_entries)
    total_test_frames = sum(len(e['ball_pos']) for e in test_entries)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total videos: {len(video_entries)}")
    print(f"Train videos: {len(train_entries)}")
    print(f"Test videos: {len(test_entries)}")
    print(f"Total train frames: {total_train_frames}")
    print(f"Total test frames: {total_test_frames}")
    print(f"Output resolution: {output_width}x{output_height}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare football dataset for TOTNet training"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="../ball-tracking-labeler",
        help="Path to ball-tracking-labeler directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/football_dataset",
        help="Output directory for TOTNet-format dataset"
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[512, 288],
        metavar=("WIDTH", "HEIGHT"),
        help="Resize images to WIDTH HEIGHT (default: 512 288 for TOTNet)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of videos for testing (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't copy files, just show what would be done"
    )
    
    args = parser.parse_args()
    
    # Handle resize argument
    resize = tuple(args.resize) if args.resize else None
    
    prepare_football_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        resize=resize,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_files=not args.dry_run
    )


if __name__ == "__main__":
    main()
