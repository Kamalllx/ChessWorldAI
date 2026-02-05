# Legacy Trackers - Original Implementations

This directory contains your original tracker implementations developed iteratively during the project.

## Files

### 1. run_camerachess.py
**Approach:** CameraChessWeb-inspired state matrix tracker

**How It Works:**
- Uses 64×12 state matrix (confidence per piece type per square)
- Decay factor (0.7) fades old observations
- Scores moves based on confidence changes at source/destination
- Requires both "from empty" and "to filled" conditions

**Key Features:**
- Can work mid-game (no starting position needed)
- Temporal smoothing through decay
- Model classification-based

**Run:**
```bash
python legacy/run_camerachess.py videos/game_4.mp4 --debug
```

**Dependencies:**
- ultralytics (YOLO)
- opencv-python
- numpy
- python-chess

**Detection Pipeline:**
```
Frame → YOLO → Map to state matrix → Apply decay → 
Add new detections → Score moves → Confirm → Execute
```

---

### 2. run_tracker_v2.py
**Approach:** Fast voting tracker with PieceDetector wrapper

**How It Works:**
- Uses custom PieceDetector class
- Voting system: accumulates detections over 2 frames
- Very fast response (window=2, cooldown=4)
- Validates piece types against expectations

**Key Features:**
- Fastest response time
- Piece type validation
- Simple voting logic

**Run:**
```bash
python legacy/run_tracker_v2.py videos/game_4.mp4 --speed 2.0
```

**Dependencies:**
- Custom PieceDetector (src/piece_detection.py)
- opencv-python
- numpy
- python-chess

**Detection Pipeline:**
```
Frame → PieceDetector → Map to squares → Vote (2 frames) → 
Analyze vacated/appeared → Score with type validation → Execute
```

---

### 3. run_tracker_v3.py
**Approach:** Direct YOLO with better thresholds

**How It Works:**
- Loads YOLO directly (no wrapper)
- Higher confidence threshold (0.3)
- 5-frame voting window (more robust than V2)
- Stricter vote threshold (60%)

**Key Features:**
- No wrapper dependency
- Better quality detections
- More robust to noise

**Run:**
```bash
python legacy/run_tracker_v3.py videos/game_4.mp4 --speed 2.0
```

**Dependencies:**
- ultralytics (YOLO)
- opencv-python
- numpy
- python-chess

**Detection Pipeline:**
```
Frame → YOLO (conf=0.3) → Map to squares → Vote (5 frames, 60%) → 
Analyze → Score → Confirm → Execute
```

---

### 4. run_tracker_v4.py
**Approach:** Ensemble detection with multiple models

**How It Works:**
- Loads multiple YOLO models
- Runs all models on each frame
- Clusters detections within 35px
- Consensus-based decisions

**Key Features:**
- Most robust (ensemble voting)
- Self-correcting through consensus
- Can combine different model architectures

**Run:**
```bash
python legacy/run_tracker_v4.py videos/game_4.mp4 --speed 2.0
```

**Dependencies:**
- ultralytics (YOLO) - multiple models
- opencv-python
- numpy
- python-chess

**Detection Pipeline:**
```
Frame → Model 1 + Model 2 + ... → Cluster (35px) → 
Vote (5 frames) → Analyze → Score → Execute
```

**Note:** Currently configured for single model but designed for multiple.

---

### 5. run_tracker_v5.py
**Approach:** Phase-based detect-lock-track

**How It Works:**
- State machine: CALIBRATE → DETECT → LOCKED → TRACKING
- Waits to detect 30+ pieces before tracking
- 5-frame accumulation per phase
- Explicit phase transitions

**Key Features:**
- Clear phase separation
- Ensures board ready before tracking
- Good for automated scenarios

**Run:**
```bash
python legacy/run_tracker_v5.py videos/game_4.mp4 --speed 2.0
```

**Dependencies:**
- ultralytics (YOLO)
- opencv-python
- numpy
- python-chess

**Detection Pipeline:**
```
CALIBRATE (corners) → DETECT (wait for 30+ pieces) → 
TRACKING (accumulate 5 frames → analyze → execute) → Repeat
```

---

### 6. run_tracker_v2_enhanced.py
**Approach:** Enhanced V2 with clustering and memory

**How It Works:**
- Direct YOLO loading
- Tight clustering (30px) to avoid merging adjacent pieces
- Detection memory: remembers pieces for 45 frames
- 4-frame voting with confidence weighting

**Key Features:**
- Handles occlusion well (memory)
- Better clustering than V4
- Confidence-weighted voting

**Run:**
```bash
python legacy/run_tracker_v2_enhanced.py videos/game_4.mp4 --speed 2.0 --debug
```

**Dependencies:**
- ultralytics (YOLO)
- opencv-python
- numpy
- python-chess

**Detection Pipeline:**
```
Frame → YOLO → Cluster (30px) → Update memory → 
Vote (4 frames) → Use memory if needed → Score → Execute
```

---

## Comparison

| Version | Speed | Accuracy | Complexity | Best For |
|---------|-------|----------|------------|----------|
| CameraChess | Medium | High | High | Research/Learning |
| V2 | FAST | Low | Low | Real-time apps |
| V3 | Medium | Medium | Low | Clean codebase |
| V4 | SLOW | HIGH | High | Production (ensemble) |
| V5 | Medium | Medium | Medium | Auto-start scenarios |
| V2E | Medium | High | Medium | Production (single model) |

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install ultralytics opencv-python numpy python-chess
```

## Common Parameters

All trackers support:
- `--speed`: Playback speed multiplier (default: 1.0)
- `--crop`: Crop ratio from top (default: 0.45)
- `--output`: Output PGN file path
- `--model`: YOLO model path (default: models/pieces.pt)
- `--debug`: Enable debug output

## Models

Place YOLO models in `models/`:
- `pieces.pt` - Original Roboflow model
- `pieces_trained.pt` - Your trained model
- `pieces_enhanced.pt` - Enhanced trained model

## Output

All trackers generate PGN files in `output/pgn/`:
```
output/pgn/game_4_camerachess.pgn
output/pgn/game_4_v2.pgn
output/pgn/game_4_v3.pgn
...
```

## Troubleshooting

### Detection Issues
- Lower confidence threshold: `--conf 0.15`
- Increase vote window in code
- Check board corners calibration

### Speed Issues
- Increase `--speed` parameter
- Reduce vote window in code
- Use faster model

### Accuracy Issues
- Use ensemble (V4)
- Use memory system (V2E)
- Adjust thresholds in code
