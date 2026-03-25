# Minimal Ball Annotation Tool

**Purpose:** Quick throwaway tool for personal use - annotate ball positions in videos for TOTNet training data.

---

## UI Mockup (Single Window)

```
┌─────────────────────────────────────────────────────────────────┐
│  Ball Annotator                              [Load] [Export]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                                                                 │
│                    ┌─────────────────┐                          │
│                    │                 │                          │
│                    │   VIDEO FRAME   │                          │
│                    │                 │                          │
│                    │      [●]        │  ← click to place ball   │
│                    │                 │                          │
│                    │                 │                          │
│                    └─────────────────┘                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ◄ [||||||||||||||||||||||||||||||░░░░░░░░░░░░] ►  Frame 142/5420│
├─────────────────────────────────────────────────────────────────┤
│  Position: (899, 234)  │  Visibility: [V1]  │  Annotated: 892   │
└─────────────────────────────────────────────────────────────────┘
```

---

## JSON Output Format

```json
[
  {
    "video": "match_001.mp4",
    "id": 1,
    "width": 1280,
    "height": 720,
    "ball_pos": [
      {"frame": 0, "ball_x": 640.0, "ball_y": 360.0, "visibility": "V1"},
      {"frame": 1, "ball_x": 642.5, "ball_y": 358.2, "visibility": "V1"},
      {"frame": 2, "ball_x": null, "ball_y": null, "visibility": "V3"}
    ]
  }
]
```

**Visibility codes:**
- `V1` - Ball clearly visible
- `V2` - Ball partially occluded  
- `V3` - Ball fully hidden/out of frame (set ball_x/ball_y to null)

---

## Keyboard Shortcuts (10 max)

| Key | Action |
|-----|--------|
| `←` / `→` | Previous/Next frame |
| `1` | Set visibility V1 (visible) |
| `2` | Set visibility V2 (partial) |
| `3` | Set visibility V3 (hidden) |
| `Delete` | Remove annotation |
| `Space` | Play/Pause |
| `O` | Open video |
| `E` | Export JSON |
| `Q` | Quit |

---

## Tech Stack

```
Python 3.x + OpenCV (cv2)
```

**Why:** OpenCV handles video loading, frame display, and mouse callbacks in ~100 lines. No GUI framework needed.

---

## Core Implementation Outline

```python
import cv2
import json

class BallAnnotator:
    def __init__(self):
        self.video_path = None
        self.cap = None
        self.annotations = {}  # {frame_num: {"x": float, "y": float, "visibility": str}}
        self.current_frame = 0
        self.total_frames = 0
        self.visibility = "V1"
        
    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_path = path
        
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.annotations[self.current_frame] = {"x": x, "y": y, "visibility": self.visibility}
            
    def export(self, output_path):
        ball_pos = [
            {"frame": f, "ball_x": a["x"], "ball_y": a["y"], "visibility": a["visibility"]}
            for f, a in sorted(self.annotations.items())
        ]
        data = [{"video": self.video_path, "id": 1, "width": self.width, "height": self.height, "ball_pos": ball_pos}]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
```

---

## Features (All 6)

1. **Load video** - Press `O` or button, select MP4/AVI/MOV
2. **Navigate** - Arrow keys or slider click/drag
3. **Mark ball** - Click on frame to place marker
4. **Set visibility** - Press 1/2/3 (marker color changes: green/yellow/red)
5. **Delete** - Press Delete to remove current frame's annotation
6. **Export** - Press `E` to save TOTNet JSON

---

## Coordinate System

```
(0,0) ────────────► X
  │
  │
  ▼
  Y
```

Origin at top-left, pixels as floats.

---

## File Structure

```
ball_annotator.py    # Single file, ~150 lines
output.json          # Exported annotations
```

---

## Build Time: ~2-3 hours

1. Video loading + display (30 min)
2. Frame navigation + slider (30 min)
3. Mouse click annotation (20 min)
4. Keyboard handlers (30 min)
5. JSON export (20 min)
6. Polish + testing (30 min)

---

## Model-Assisted Workflow (Extension)

**Use case:** Import pre-generated ball positions from TOTNet model, then manually correct errors. Same JSON format works for both input and output.

### Additional Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `I` | Import annotations JSON |
| `D` | Toggle drag mode (edit existing markers) |
| `M` | Mark current frame as missed (add new annotation) |

### Import Function

```python
def import_annotations(self, json_path):
    with open(json_path) as f:
        data = json.load(f)
    for entry in data[0]["ball_pos"]:
        if entry["ball_x"] is not None:
            self.annotations[entry["frame"]] = {
                "x": entry["ball_x"],
                "y": entry["ball_y"],
                "visibility": entry["visibility"]
            }
```

### Updated Mouse Handler (Drag Support)

```python
def on_mouse(self, event, x, y, flags, param):
    ann = self.annotations.get(self.current_frame)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if self.drag_mode and ann:
            dist = ((x - ann["x"])**2 + (y - ann["y"])**2)**0.5
            if dist < 15:
                self.dragging = True
        if not self.dragging:
            self.annotations[self.current_frame] = {"x": x, "y": y, "visibility": self.visibility}
            
    elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
        self.annotations[self.current_frame]["x"] = x
        self.annotations[self.current_frame]["y"] = y
        
    elif event == cv2.EVENT_LBUTTONUP:
        self.dragging = False
```

### Workflow

1. `O` Load video → `I` Import predictions → `←`/`→` Review frames
2. `D` Toggle drag mode to reposition markers
3. `M` Add missed annotations; `1`/`2`/`3` Adjust visibility
4. `E` Export corrected JSON
