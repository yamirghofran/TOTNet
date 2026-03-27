# TOTNet Custom Dataset Training Guide

This guide covers everything you need to train TOTNet on custom sports datasets. It documents the football dataset implementation and provides a template for adding new sports.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Adding a New Sport](#adding-a-new-sport)
5. [Cloud Training Environments](#cloud-training-environments)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### Architecture

TOTNet requires:
- **Image frames** organized by video/match
- **JSON annotations** with ball positions and visibility
- **Configuration** in `src/config/config.py`
- **Data loader** in `src/data_process/`

### Supported Datasets

| Dataset | Config Value | Description |
|---------|--------------|-------------|
| Table Tennis | `tt` | Original TTNet dataset |
| Tennis | `tennis` | Tennis ball tracking |
| Badminton | `badminton` | Badminton shuttlecock |
| TTA | `tta` | Table tennis alternative format |
| Football | `football` | Custom football dataset (newly added) |

---

## Data Preparation

### Input Format

The data conversion script expects sequences from `ball-tracking-labeler`:

```
ball-tracking-labeler/
├── video-name_augmented/
│   ├── sequence_00000/
│   │   ├── frame_0.png
│   │   ├── frame_1.png
│   │   ├── frame_2.png
│   │   ├── frame_3.png
│   │   ├── frame_4.png
│   │   └── annotations.json
│   ├── sequence_00001/
│   └── ...
└── another-video_augmented/
    └── ...
```

### Annotation Format

Each `annotations.json` should contain:

```json
{
  "sequence_id": 0,
  "source_video": "match_001.mp4",
  "source_frames": [100, 101, 102, 103, 104],
  "ball_pos": [
    {"frame": 0, "ball_x": 905.0, "ball_y": 382.0, "visibility": 3},
    {"frame": 1, "ball_x": 898.0, "ball_y": 380.0, "visibility": 3},
    {"frame": 2, "ball_x": 890.0, "ball_y": 386.0, "visibility": 2},
    {"frame": 3, "ball_x": 886.0, "ball_y": 388.0, "visibility": 1},
    {"frame": 4, "ball_x": 875.0, "ball_y": 385.0, "visibility": 1}
  ]
}
```

**Visibility Levels:**

| Code | Description | Use Case |
|------|-------------|----------|
| 0 | Out of frame | Ball not visible in image |
| 1 | Clearly visible | Ball is fully visible |
| 2 | Partially visible | Ball is partially occluded |
| 3 | Fully occluded | Ball is hidden but position is estimated |

### Running the Conversion Script

```bash
# Basic usage
uv run python src/data_process/prepare_football_data.py \
    --source_dir ../ball-tracking-labeler \
    --output_dir ./data/football_dataset

# With custom options
uv run python src/data_process/prepare_football_data.py \
    --source_dir /path/to/labeled/data \
    --output_dir ./data/football_dataset \
    --resize 512 288 \
    --test_ratio 0.2 \
    --seed 42

# Dry run (no file copying)
uv run python src/data_process/prepare_football_data.py \
    --source_dir ../ball-tracking-labeler \
    --output_dir ./data/football_dataset \
    --dry_run
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source_dir` | `../ball-tracking-labeler` | Path to labeled data |
| `--output_dir` | `./data/football_dataset` | Output directory |
| `--resize` | `512 288` | Target width height |
| `--test_ratio` | `0.2` | Fraction of videos for testing |
| `--seed` | `42` | Random seed for reproducibility |
| `--dry_run` | False | Show what would be done |

### Output Structure

```
data/football_dataset/
├── frames/
│   ├── video-name-1/
│   │   ├── img_000000.jpg
│   │   ├── img_000001.jpg
│   │   └── ...
│   ├── video-name-2/
│   └── ...
├── train.json
└── test.json
```

### Output JSON Format

```json
[
  {
    "video": "video-name-1",
    "id": 1,
    "width": 512,
    "height": 288,
    "ball_pos": [
      {"frame": 0, "ball_x": 905.0, "ball_y": 382.0, "visibility": 3},
      {"frame": 1, "ball_x": 898.0, "ball_y": 380.0, "visibility": 3},
      ...
    ]
  }
]
```

---

## Training

### Local Training

```bash
# Using the training script
./scripts/train_football_cloud.sh --epochs 30 --batch-size 8

# Or directly with Python
uv run python src/main.py \
    --num_epochs 30 \
    --saved_fn 'TOTNet_Football' \
    --num_frames 5 \
    --batch_size 8 \
    --dataset_choice 'football' \
    --model_choice 'TOTNet' \
    --img_size 288 512 \
    --pretrained_path 'weights/TOTNet_Tennis_..._best.pth'
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_epochs` | 30 | Number of training epochs |
| `--batch_size` | 16 | Batch size (reduce if OOM) |
| `--lr` | 5e-4 | Learning rate |
| `--img_size` | 288 512 | Image height width |
| `--num_frames` | 5 | Frames per sequence |
| `--dataset_choice` | football | Dataset to use |
| `--model_choice` | TOTNet | Model architecture |
| `--pretrained_path` | auto | Path to pretrained weights |
| `--val-size` | 0.2 | Validation split ratio |
| `--occluded_prob` | 0.1 | Probability of ball masking |
| `--weighting_list` | 1 2 2 3 | Loss weights for stages |

### Fine-Tuning vs Training from Scratch

```bash
# Fine-tune from Tennis weights (recommended)
./scripts/train_football_cloud.sh --pretrained_path weights/TOTNet_Tennis_..._best.pth

# Train from scratch
./scripts/train_football_cloud.sh --no-pretrained
```

### uv vs pip

The training script uses `uv` by default for dependency management. If `uv` is not installed, it script falls back to `pip`.

**Using uv (recommended):**
```bash
# Install uv first if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Run training - dependencies auto-installed
./scripts/train_football_cloud.sh --batch-size 16 --epochs 30
```

**Using pip:**
```bash
# The script will auto-detect and use pip if uv is not installed
# OR explicitly disable uv:
./scripts/train_football_cloud.sh --batch-size 16 --epochs 30 --no-uv

# Manual pip install
pip install -r requirements.txt
./scripts/train_football_cloud.sh --no-uv
```

### Resuming Training

```bash
./scripts/train_football_cloud.sh --resume checkpoints/TOTNet_Football/model_epoch_10.pth
```

---

## Adding a New Sport

This section describes how to add support for a new sport (e.g., basketball, hockey, cricket).

### Step 1: Prepare Your Data

Option A: If using `ball-tracking-labeler`:
```bash
# Your data should be in:
your-sport-labeler/
├── match-001_augmented/
│   ├── sequence_00000/
│   │   └── annotations.json
│   └── ...
└── match-002_augmented/
```

Option B: If you have data in another format, create a conversion script.

### Step 2: Add Dataset Configuration

Edit `src/config/config.py`:

```python
# Add after existing dataset configs (around line 230)
###################################################################
##############          Basketball dataset
####################
configs.basketball_dataset_dir = os.path.join(configs.working_dir, 'data', 'basketball_dataset')
```

### Step 3: Add Data Loading Function

Edit `src/data_process/data_utils.py`:

Add a new function based on the football implementation:

```python
def get_all_detection_infor_basketball(dataset_dir, dataset_type, num_frames=5, resize=None, bidirect=False):
    """Load basketball dataset annotations in TOTNet format.
    
    Args:
        dataset_dir: Path to basketball_dataset directory
        dataset_type: 'train' or 'test'
        num_frames: Number of frames per sequence (default: 5)
        resize: Optional (height, width) tuple for coordinate scaling
        bidirect: If True, target frame is middle; if False, target is last
    
    Returns:
        events_infor: List of image path lists
        events_labels: List of [ball_position, visibility, status]
    """
    if dataset_type == 'train':
        annos_file = os.path.join(dataset_dir, 'train.json')
    else:
        annos_file = os.path.join(dataset_dir, 'test.json')
    
    if not os.path.exists(annos_file):
        raise FileNotFoundError(f"Annotation file not found: {annos_file}")
    
    with open(annos_file, 'r') as f:
        annos = json.load(f)
    
    status = 1
    events_infor = []
    events_labels = []
    
    for video in annos:
        video_name = video['video']
        img_width = video['width']
        img_height = video['height']
        images_dir = os.path.join(dataset_dir, 'frames', video_name)
        
        frame_to_ball = {}
        for bp in video['ball_pos']:
            frame_to_ball[bp['frame']] = bp
        
        all_frames = sorted(frame_to_ball.keys())
        
        for i in range(len(all_frames) - num_frames + 1):
            sequence_frames = all_frames[i:i + num_frames]
            
            img_path_list = []
            for frame_num in sequence_frames:
                img_path = os.path.join(images_dir, f'img_{frame_num:06d}.jpg')
                img_path_list.append(img_path)
            
            if bidirect:
                target_idx = num_frames // 2
            else:
                target_idx = num_frames - 1
            
            target_frame = sequence_frames[target_idx]
            ball_info = frame_to_ball.get(target_frame)
            
            if ball_info is None:
                continue
            
            x = ball_info['ball_x']
            y = ball_info['ball_y']
            
            if x is not None and y is not None and resize is not None:
                x = int(x * (resize[1] / img_width))
                y = int(y * (resize[0] / img_height))
            elif x is not None and y is not None:
                x = int(x)
                y = int(y)
            
            ball_position = np.array([x, y], dtype=int) if x is not None else np.array([-1, -1], dtype=int)
            visibility = ball_info.get('visibility', 1)
            
            events_infor.append(img_path_list)
            events_labels.append([ball_position, visibility, status])
    
    return events_infor, events_labels
```

### Step 4: Update Train/Val Splitting

Add to `train_val_data_separation()` in `src/data_process/data_utils.py`:

```python
elif configs.dataset_choice == 'basketball':
    events_infor, events_labels = get_all_detection_infor_basketball(
        configs.basketball_dataset_dir, 'train', 
        num_frames=configs.num_frames, 
        resize=configs.resize, 
        bidirect=configs.bidirect
    )
    if configs.no_val:
        train_events_infor = events_infor
        train_events_labels = events_labels
        val_events_infor = None
        val_events_labels = None
    else:
        train_events_infor, val_events_infor, train_events_labels, val_events_labels = train_test_split(
            events_infor, events_labels,
            shuffle=True,
            test_size=configs.val_size,
            random_state=configs.seed,
        )
```

### Step 5: Update Dataloader

Edit `src/data_process/dataloader.py`:

**Add import:**
```python
from data_process.data_utils import (
    # ... existing imports ...
    get_all_detection_infor_basketball
)
```

**Add to `create_occlusion_train_val_dataloader()`:**
```python
elif configs.dataset_choice == 'basketball':
    train_dataset = TTA_Dataset(train_events_infor, train_events_label, transform=train_transform,
                                num_samples=configs.num_samples)
```

**Add validation dataset:**
```python
elif configs.dataset_choice == 'basketball':
    val_dataset = TTA_Dataset(val_events_infor, val_events_label, transform=val_transform,
                              num_samples=configs.num_samples)
```

**Add to `create_occlusion_test_dataloader()`:**
```python
elif configs.dataset_choice == 'basketball':
    test_events_infor, test_events_labels = get_all_detection_infor_basketball(
        configs.basketball_dataset_dir, 'test', 
        num_frames=configs.num_frames, 
        resize=configs.resize, 
        bidirect=configs.bidirect
    )
    test_dataset = TTA_Dataset(test_events_infor, test_events_labels, transform=test_transform,
                            num_samples=configs.num_samples)
```

### Step 6: Create Data Preparation Script

Copy and modify `prepare_football_data.py`:

```bash
cp src/data_process/prepare_football_data.py src/data_process/prepare_basketball_data.py
```

Modify the script:
1. Change function/class names
2. Update default paths
3. Adjust any sport-specific logic

### Summary: Files to Modify

| File | Change |
|------|--------|
| `src/config/config.py` | Add `configs.<sport>_dataset_dir` |
| `src/data_process/data_utils.py` | Add `get_all_detection_infor_<sport>()` and update `train_val_data_separation()` |
| `src/data_process/dataloader.py` | Add import and dataset cases |
| `src/data_process/prepare_<sport>_data.py` | Create data preparation script (optional) |

---

## Cloud Training Environments

### Google Colab

**Setup:**
1. Create ZIP files locally:
   ```bash
   # Zip dataset
   cd data && zip -r ../football_dataset.zip football_dataset/
   
   # Zip TOTNet code
   cd .. && zip -r TOTNet.zip TOTNet -x "TOTNet/.git/*" "TOTNet/.venv/*"
   ```

2. Upload to Google Drive

3. Open `notebooks/train_football_colab.ipynb` in Colab

4. Set GPU runtime: Runtime → Change runtime type → T4 GPU

5. Run all cells

**Colab Notes:**
- Free tier: T4 GPU, ~12GB VRAM (batch_size 8-12)
- Pro tier: A100 GPU, ~40GB VRAM (batch_size 16-24)
- Sessions timeout after 12 hours (free) or 24 hours (pro)

### RunPod / Lambda Labs

**Setup:**
1. Launch GPU instance (RTX 4090, A100, etc.)

2. Connect via SSH

3. Install dependencies:
   ```bash
   # Using conda
   conda create -n totNet python=3.10
   conda activate totNet
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   ```

4. Clone/upload code and data

5. Run training:
   ```bash
   ./scripts/train_football_cloud.sh --batch-size 24 --epochs 30
   ```

**Multi-GPU Training:**
```bash
./scripts/train_football_cloud.sh --gpus 2 --batch-size 32
```

### AWS / GCP / Azure

**Instance Requirements:**
- GPU: NVIDIA T4, V100, A100, or H100
- CUDA: 11.8+
- Python: 3.10+
- Storage: 50GB+ (for data and checkpoints)

**AWS Example (p3.2xlarge - V100):**
```bash
# Launch instance with Deep Learning AMI
ssh ubuntu@<instance-ip>

# Setup
source activate pytorch
git clone <your-repo>
cd TOTNet

# Upload data (using scp or S3)
aws s3 cp s3://your-bucket/football_dataset.zip .
unzip football_dataset.zip -d data/

# Train
./scripts/train_football_cloud.sh --batch-size 16
```

### Docker (Any Cloud)

Create `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["./scripts/train_football_cloud.sh"]
```

Build and run:
```bash
docker build -t totNet:latest .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints totNet:latest
```

---

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size
```bash
./scripts/train_football_cloud.sh --batch-size 8  # or 4
```

#### Data Loading Errors
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solution:** Check paths in train.json/test.json match actual file locations
```bash
# Verify paths
head -20 data/football_dataset/train.json
ls data/football_dataset/frames/video-name/
```

#### Pretrained Weights Not Found
```
FileNotFoundError: pretrained weights not found
```
**Solution:** Specify path explicitly or train from scratch
```bash
./scripts/train_football_cloud.sh --pretrained_path weights/TOTNet_Tennis_.../best.pth
# or
./scripts/train_football_cloud.sh --no-pretrained
```

#### Slow Data Loading
**Solution:** Increase num_workers or use SSD storage
```bash
# In training command, add:
--num_workers 4
```

#### Image Resolution Mismatch
```
Coordinate scaling issues
```
**Solution:** Ensure resize matches img_size
```bash
# Data preparation
--resize 512 288

# Training
--img_size 288 512  # Note: height, width order
```

### Verification Commands

```bash
# Check dataset integrity
python -c "
import json
with open('data/football_dataset/train.json') as f:
    data = json.load(f)
print(f'Videos: {len(data)}')
print(f'Total frames: {sum(len(v[\"ball_pos\"]) for v in data)}')
"

# Check GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"

# Test dataloader
uv run python -c "
from src.data_process.dataloader import create_occlusion_train_val_dataloader
from src.config.config import parse_configs
configs = parse_configs()
configs.dataset_choice = 'football'
configs.batch_size = 1
configs.distributed = False
train_dl, val_dl, _ = create_occlusion_train_val_dataloader(configs)
print(f'Train batches: {len(train_dl)}, Val batches: {len(val_dl)}')
"
```

---

## Reference: File Locations

```
TOTNet/
├── src/
│   ├── main.py                          # Training entry point
│   ├── config/
│   │   └── config.py                    # Dataset paths
│   └── data_process/
│       ├── prepare_football_data.py     # Data conversion
│       ├── data_utils.py                # Data loading functions
│       ├── dataloader.py                # PyTorch dataloaders
│       └── dataset.py                   # Dataset classes
├── scripts/
│   └── train_football_cloud.sh          # Training script
├── notebooks/
│   └── train_football_colab.ipynb       # Colab notebook
├── weights/
│   └── TOTNet_Tennis_.../               # Pretrained weights
├── data/
│   └── football_dataset/
│       ├── frames/
│       ├── train.json
│       └── test.json
├── checkpoints/
│   └── TOTNet_Football/                 # Saved models
├── docs/
│   └── TRAINING_FOOTBALL.md             # This file
├── requirements.txt
└── pyproject.toml
```

---

## Changelog

### 2025-03-25
- Added football dataset support
- Created `prepare_football_data.py` conversion script
- Added `get_all_detection_infor_football()` data loader
- Updated dataloader.py for football dataset
- Created cloud training script and Colab notebook
- Added comprehensive documentation

---

## License

MIT License - See LICENSE file for details.
