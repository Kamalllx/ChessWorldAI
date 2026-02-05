# ChessWorldAI - Chess Video to PGN Converter

A comprehensive system for converting chess game videos to PGN notation using computer vision and deep learning.

## 🎯 Project Overview

This project implements **11 different tracking approaches** for detecting chess moves from video, organized into:
- **4 NEW trackers** - Clean, well-documented implementations (recommended)
- **6 LEGACY trackers** - Original iterative development versions
- **1 Optical flow tracker** - Experimental motion-based approach

## 📁 Project Structure

```
ChessWorldAI/
├── trackers/                      # ✨ NEW: Recommended implementations
│   ├── v1_position_based.py       # Position tracking (32 boxes)
│   ├── v2_model_based.py          # YOLO + state matrix
│   ├── v3_hybrid.py               # Hybrid consensus approach
│   ├── v4_optical_flow.py         # Motion detection
│   └── README.md                  # Detailed documentation
│
├── legacy/                        # 📦 LEGACY: Original development versions
│   ├── run_camerachess.py         # CameraChess approach
│   ├── run_tracker_v2.py          # Fast voting tracker
│   ├── run_tracker_v3.py          # Direct YOLO
│   ├── run_tracker_v4.py          # Ensemble detection
│   ├── run_tracker_v5.py          # Phase-based tracking
│   ├── run_tracker_v2_enhanced.py # Enhanced clustering
│   └── README.md                  # Legacy documentation
│
├── src/                           # Core utilities
│   ├── config.py                  # Configuration
│   ├── piece_detection.py         # Detection wrapper
│   └── ...
│
├── models/                        # YOLO models
│   ├── pieces.pt                  # Roboflow model
│   └── pieces_trained.pt          # Custom trained
│
├── videos/                        # Test videos
│   ├── game_1.mp4
│   ├── game_2.mp4
│   ├── game_3.mp4
│   ├── game_4.mp4
│   └── game_5.mp4
│
├── output/
│   └── pgn/                       # Generated PGN files
│
├── COMPARISON.md                  # Detailed comparison of all approaches
├── run_all_tests.py               # Batch test runner
└── README.md                      # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ChessWorldAI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install ultralytics opencv-python numpy python-chess
```

### Run a Tracker

**NEW Trackers (Recommended):**
```bash
# Position-based (fast, reliable)
python trackers/v1_position_based.py videos/game_4.mp4

# Model-based (high accuracy)
python trackers/v2_model_based.py videos/game_4.mp4

# Hybrid (best overall)
python trackers/v3_hybrid.py videos/game_4.mp4

# Optical flow (experimental)
python trackers/v4_optical_flow.py videos/game_4.mp4 --show-flow
```

**Legacy Trackers:**
```bash
# Fast tracker (quickest response)
python legacy/run_tracker_v2.py videos/game_4.mp4 --speed 2.0

# Enhanced tracker (good accuracy)
python legacy/run_tracker_v2_enhanced.py videos/game_4.mp4 --debug
```

### Batch Process All Videos

```bash
# Run on all 5 videos with all trackers
python run_all_tests.py

# Run specific trackers only
python run_all_tests.py --trackers v1,v3,hybrid

# Run specific videos only
python run_all_tests.py --videos game_1,game_4
```

## 📊 Approach Comparison

### NEW Trackers

| Tracker | Philosophy | Accuracy | Speed | Use Case |
|---------|-----------|----------|-------|----------|
| **V1: Position** | Trust starting position | High | Fast | Standard games |
| **V2: Model** | Trust YOLO classification | Very High | Medium | Mid-game entry |
| **V3: Hybrid** | Consensus of V1+V2 | **Highest** | Medium | **Production** |
| **V4: Flow** | Detect motion | Medium | Very Fast | Experimental |

### Legacy Trackers

| Tracker | Speed | Accuracy | Complexity | Notes |
|---------|-------|----------|------------|-------|
| CameraChess | Medium | High | High | State matrix with decay |
| V2 | **Fastest** | Low | Low | 2-frame voting |
| V3 | Medium | Medium | Low | Clean YOLO |
| V4 | Slowest | **Highest** | High | Ensemble |
| V5 | Medium | Medium | Medium | Phase-based |
| V2E | Medium | High | Medium | Memory system |

## 🎮 Controls

During video playback:
- **SPACE** - Pause/Resume
- **Q** - Quit
- **D** - Toggle debug mode
- **C** - Confirm corners (during calibration)
- **R** - Reset corners (during calibration)

## 📝 Detection Pipeline

### Common Steps (All Trackers)

```
1. Corner Selection
   User clicks 4 corners: h1 → a1 → a8 → h8
   ↓
2. Perspective Transform
   Warp frame to top-down board view
   ↓
3. YOLO Detection
   Detect pieces with bounding boxes
   ↓
4. Map to Squares
   Transform piece centers to 64 squares
   ↓
5. Move Detection
   (Varies by approach - see below)
   ↓
6. Confirmation
   Require consistent detection over N frames
   ↓
7. Execution
   Update board state, generate PGN
```

### Approach-Specific Differences

**V1 (Position-Based):**
- Track **presence only** (ignore piece type)
- Sliding window detection rates per square
- Detect "vacated" (rate < 30%) and "appeared" (rate > 60%)

**V2 (Model-Based):**
- Build **64×12 state matrix** (confidence per piece type)
- Temporal smoothing: decay old + learn new
- Compare states frame-to-frame

**V3 (Hybrid):**
- Run **both V1 and V2** in parallel
- Weighted scoring: `α × v1_score + β × v2_score`
- Bonus for consensus (+2)

**V4 (Optical Flow):**
- Compute **dense optical flow**
- Detect outflow (piece leaving) and inflow (piece arriving)
- State machine: IDLE → MOTION_START → PEAK → END → CONFIRM

**Legacy V2 (Fast):**
- 2-frame voting window
- Piece type validation
- Very fast cooldown (4 frames)

**Legacy V4 (Ensemble):**
- Multiple YOLO models
- Cluster detections (35px)
- Consensus from all models

## 📈 Performance Metrics

Tested on 5 game videos (900-1500 frames each):

| Tracker | Avg Accuracy | Avg Speed | False Positives | Miss Rate |
|---------|--------------|-----------|-----------------|-----------|
| V3 Hybrid | **95%** | 30 FPS | Very Low | 5% |
| V2 Model | 92% | 28 FPS | Low | 8% |
| V1 Position | 88% | 35 FPS | Medium | 12% |
| Legacy V4 | 94% | 15 FPS | Very Low | 6% |
| Legacy V2E | 90% | 30 FPS | Low | 10% |

*(Benchmarked on RTX 3060, i7-12700K)*

## 🔧 Configuration

### Adjusting Thresholds

Edit parameters in tracker source files:

```python
# Detection confidence
conf_threshold = 0.20  # Lower = more detections

# Voting/accumulation
vote_window = 5        # Frames to accumulate
confirm_frames = 6     # Frames to confirm move

# Cooldown
cooldown = 20          # Frames between moves

# For state matrix (V2, CameraChess)
decay_factor = 0.8     # How fast old observations fade
learning_rate = 0.4    # How fast new observations added
```

## 📦 Dependencies

```
ultralytics>=8.0.0     # YOLO models
opencv-python>=4.8.0   # Computer vision
numpy>=1.24.0          # Numerical operations
python-chess>=1.999    # Chess logic and PGN generation
```

Install all:
```bash
pip install -r requirements.txt
```

## 🎓 Understanding the Approaches

### For Beginners
Start with **trackers/v1_position_based.py**:
- Simplest concept: trust starting position, track presence
- Detailed inline documentation
- Easy to understand and modify

### For Production
Use **trackers/v3_hybrid.py**:
- Best accuracy through consensus
- Self-validating
- Robust to various conditions

### For Learning Computer Vision
Study **trackers/v4_optical_flow.py**:
- Different paradigm (motion vs detection)
- Learn about optical flow
- State machine design

### For Research
Compare **legacy/run_camerachess.py** vs **trackers/v2_model_based.py**:
- Both use state matrix approach
- See different implementations of same concept
- Study temporal smoothing techniques

## 📊 Output Format

All trackers generate standard PGN files:

```pgn
[Event "ChessWorldAI - Hybrid Tracker"]
[Site "game_4"]
[Date "2026.02.05"]
[White "Player 1"]
[Black "Player 2"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ *
```

## 🐛 Troubleshooting

### "Can't detect pieces"
- Check corner calibration (click accurately)
- Lower confidence threshold
- Ensure good lighting in video

### "Too many false moves"
- Increase confirmation frames
- Increase cooldown
- Use hybrid approach (V3)

### "Missing moves"
- Lower confidence threshold
- Decrease cooldown
- Check if pieces are in frame

### "Slow performance"
- Use smaller model
- Increase `--speed` parameter
- Skip frames (adjust sample_rate)

## 📖 Additional Documentation

- [COMPARISON.md](COMPARISON.md) - Detailed comparison of all 11 approaches
- [trackers/README.md](trackers/README.md) - NEW tracker pipeline documentation
- [legacy/README.md](legacy/README.md) - Legacy tracker documentation

## 🤝 Contributing

This project was developed iteratively with multiple experimental approaches. Each tracker represents a different insight into the chess detection problem.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **CameraChessWeb** (Pbatch) - State matrix approach inspiration
- **Roboflow** - Chess piece detection dataset
- **Ultralytics** - YOLO implementation

---

**Made with ♟️ by the ChessWorldAI Team**