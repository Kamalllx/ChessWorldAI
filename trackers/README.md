# ChessWorldAI Tracker Versions

This document provides detailed documentation of the 4 different tracking approaches implemented for chess video to PGN conversion.

---

## Quick Comparison

| Version | Approach | Depends On | Best For |
|---------|----------|------------|----------|
| **V1** | Position-Based | Starting position + presence detection | Standard games, any piece style |
| **V2** | Model-Based | YOLO classification | Mid-game entry, piece identification |
| **V3** | Hybrid | Both V1 + V2 | Maximum accuracy, production use |
| **V4** | Optical Flow | Motion detection | Fast games, unknown piece styles |

---

## Version 1: Position-Based Tracker

**File:** `trackers/v1_position_based.py`

### Philosophy
> "We KNOW where pieces start. Just detect when they move."

### Pipeline Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   VIDEO     │────▶│   CORNERS   │────▶│  INIT 32    │
│   INPUT     │     │  SELECTION  │     │  PIECES     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    YOLO     │◀────│   WARP TO   │◀────│   FRAME     │
│  DETECTION  │     │   BOARD     │     │   LOOP      │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼ (ignore classes - only use presence)
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  OCCUPIED   │────▶│  SLIDING    │────▶│  DETECTION  │
│  SQUARES    │     │  WINDOW     │     │   RATES     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  VACATED    │────▶│  SCORE      │────▶│  CONFIRM    │
│  APPEARED   │     │  MOVES      │     │  EXECUTE    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Key Concepts

1. **32 Boxes**: We initialize with exact knowledge of where all 32 pieces are
2. **Presence Detection**: YOLO finds "something" at a square (class label ignored)
3. **Detection Rate**: Sliding window tracks how often each square is detected
4. **Vacated**: Detection rate drops below 30% → piece left
5. **Appeared**: Previously empty square now consistently detected → piece arrived

### Parameters
```python
window_size = 10          # Frames for sliding window
vacated_threshold = 0.3   # Below this = piece left
appeared_threshold = 0.6  # Above this = piece arrived
confirm_frames = 5        # Frames to confirm move
cooldown = 20             # Frames between moves
```

### Usage
```bash
python trackers/v1_position_based.py videos/game.mp4 --debug
```

---

## Version 2: Model-Based Detector

**File:** `trackers/v2_model_based.py`

### Philosophy
> "Trust the model. It knows what each piece looks like."

### Pipeline Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   VIDEO     │────▶│   CORNERS   │────▶│   YOLO      │
│   INPUT     │     │  SELECTION  │     │  DETECTION  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌───────────────────────────────────────────────────────┐
│                 STATE MATRIX (64 × 12)                │
│                                                       │
│  Square →  a1  a2  a3  ...  h8                       │
│  ──────────────────────────────────                  │
│  b (black bishop)  [0.0, 0.0, ...]                   │
│  k (black king)    [0.0, 0.0, ...]                   │
│  n (black knight)  [0.0, 0.0, ...]                   │
│  p (black pawn)    [0.7, 0.0, ...]  ← pawn on a2    │
│  ...                                                  │
│  R (white rook)    [0.9, 0.0, ...]  ← rook on a1    │
└───────────────────────────────────────────────────────┘
       │
       ▼ (temporal smoothing: decay + learning)
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  EXTRACT    │────▶│  COMPARE    │────▶│  FIND       │
│  PIECES     │     │  STATES     │     │  CHANGES    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  REMOVED    │────▶│  SCORE      │────▶│  CONFIRM    │
│  ADDED      │     │  MOVES      │     │  EXECUTE    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Key Concepts

1. **State Matrix**: 64 squares × 12 piece types, each cell is confidence
2. **Class Mapping**: 
   - YOLO classes 1-6 → Black pieces (b, k, n, p, q, r)
   - YOLO classes 7-12 → White pieces (B, K, N, P, Q, R)
3. **Temporal Smoothing**:
   - `state = state × decay` (old observations fade)
   - `state = max(state, new × learning_rate)` (new observations add)
4. **Piece Extraction**: For each square, find max confidence → that's the piece

### Parameters
```python
decay_factor = 0.8        # How fast old observations fade
learning_rate = 0.4       # How fast new observations are incorporated  
occupancy_threshold = 0.5 # Minimum confidence to count as occupied
conf_threshold = 0.25     # YOLO confidence threshold
confirm_frames = 8        # Frames to confirm move
```

### Usage
```bash
python trackers/v2_model_based.py videos/game.mp4 --debug
```

---

## Version 3: Hybrid Tracker

**File:** `trackers/v3_hybrid.py`

### Philosophy
> "Use position tracking for stability, model for validation. Trust consensus."

### Pipeline Diagram

```
                    ┌─────────────┐
                    │   SHARED    │
                    │   YOLO      │
                    │   DETECT    │
                    └─────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│    V1 SUBSYSTEM     │     │    V2 SUBSYSTEM     │
│  (Position-Based)   │     │   (Model-Based)     │
│                     │     │                     │
│  • Detection history│     │  • State matrix     │
│  • Detection rates  │     │  • Piece extraction │
│  • Vacated/Appeared │     │  • Removed/Added    │
└─────────────────────┘     └─────────────────────┘
            │                           │
            ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│   score_v1(move)    │     │   score_v2(move)    │
└─────────────────────┘     └─────────────────────┘
            │                           │
            └───────────┬───────────────┘
                        ▼
              ┌─────────────────┐
              │  WEIGHTED SUM   │
              │                 │
              │  combined =     │
              │  α × score_v1 + │
              │  β × score_v2   │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │   CONSENSUS     │
              │   BONUS/PENALTY │
              │                 │
              │  Both agree: +2 │
              │  Disagree: -2   │
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │   ADAPTIVE      │
              │   CONFIRMATION  │
              │                 │
              │  High conf: 4fr │
              │  Med conf: 6fr  │
              │  Low conf: 10fr │
              └─────────────────┘
```

### Key Concepts

1. **Dual State**: Maintains both V1 (position) and V2 (model) states in parallel
2. **Shared Detection**: Single YOLO call, results fed to both subsystems
3. **Weighted Scoring**: `combined = trust_position × v1_score + trust_model × v2_score`
4. **Consensus Bonus**: Agreement between systems boosts confidence
5. **Adaptive Confirmation**: Higher confidence = faster confirmation

### Parameters
```python
trust_position = 0.6      # Weight for V1 evidence
trust_model = 0.4         # Weight for V2 evidence
base_confirm = 6          # Base confirmation frames
# Adjusted based on confidence:
# High agreement: base - 2 = 4 frames
# Low agreement: base + 2 = 8 frames
```

### Usage
```bash
python trackers/v3_hybrid.py videos/game.mp4 --debug
```

---

## Version 4: Optical Flow Tracker

**File:** `trackers/v4_optical_flow.py`

### Philosophy
> "Detect motion, not pieces. When something moves, track where it goes."

### Pipeline Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  FRAME N-1  │     │  FRAME N    │     │   WARP TO   │
│  (prev)     │     │  (curr)     │────▶│  TOP-DOWN   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       └───────────┬───────┘                   │
                   ▼                           │
         ┌─────────────────┐                   │
         │  OPTICAL FLOW   │◀──────────────────┘
         │  (Farneback)    │
         │                 │
         │  flow[y,x] =    │
         │   (dx, dy)      │
         └─────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  PER-SQUARE     │
         │  ANALYSIS       │
         │                 │
         │  For each sq:   │
         │  • motion_count │
         │  • avg_flow     │
         │  • direction    │
         └─────────────────┘
                   │
         ┌────────┴────────┐
         ▼                 ▼
┌─────────────┐     ┌─────────────┐
│  OUTFLOW    │     │  INFLOW     │
│  (leaving)  │     │  (arriving) │
│             │     │             │
│  Vectors    │     │  Vectors    │
│  point OUT  │     │  point IN   │
└─────────────┘     └─────────────┘
         │                 │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  STATE MACHINE  │
         │                 │
         │  IDLE           │
         │    ↓            │
         │  MOTION_START   │
         │    ↓            │
         │  MOTION_PEAK    │
         │    ↓            │
         │  MOTION_END     │
         │    ↓            │
         │  CONFIRM        │
         └─────────────────┘
```

### Key Concepts

1. **Optical Flow**: Computes motion vectors between consecutive frames
2. **Warped View**: 480×480 top-down view of board (60px per square)
3. **Outflow Detection**: Vectors pointing away from square center = piece leaving
4. **Inflow Detection**: Vectors pointing toward square center = piece arriving
5. **Motion State Machine**: Tracks the lifecycle of a piece movement
6. **Direction Matching**: Flow direction should match source→destination line

### State Machine
```
IDLE ──(motion starts)──▶ MOTION_START
                               │
                    (motion increases)
                               ▼
                         MOTION_PEAK
                               │
                    (motion decreases)
                               ▼
                         MOTION_END
                               │
                       (motion stops)
                               ▼
                           CONFIRM ──(valid move)──▶ EXECUTE
```

### Parameters
```python
motion_threshold = 2.0     # Minimum flow magnitude to count
outflow_threshold = 200    # Motion pixels for outflow
inflow_threshold = 200     # Motion pixels for inflow
cooldown = 30              # Frames between moves
```

### Usage
```bash
python trackers/v4_optical_flow.py videos/game.mp4 --debug --show-flow
```

---

## Recommendations

### For Production Use
**Use V3 (Hybrid)** - Combines the best of both approaches with self-validation.

### For Fast/Simple Games
**Use V1 (Position-Based)** - Minimal model dependency, very fast.

### For Mid-Game Entry
**Use V2 (Model-Based)** - Can start from any position if model is accurate.

### For Unknown Piece Styles
**Use V4 (Optical Flow)** - Works without piece classification.

---

## Common Issues & Solutions

| Issue | V1 Solution | V2 Solution | V3 Solution | V4 Solution |
|-------|-------------|-------------|-------------|-------------|
| Model misclassifies | N/A (no classification) | Increase threshold | V1 compensates | N/A |
| Piece not detected | Lower conf_threshold | Lower conf_threshold | Either system can detect | Use motion |
| Hand blocks view | Window smoothing | Decay handles occlusion | Either recovers | Wait for motion end |
| Camera shake | Cooldown | Cooldown | Cooldown | Threshold motion |
| Wrong move detected | Confirm threshold | Confirm threshold | Consensus required | State machine |

---

## Running All Versions

```bash
# V1: Position-Based
python trackers/v1_position_based.py videos/game.mp4 -o output/game_v1.pgn

# V2: Model-Based  
python trackers/v2_model_based.py videos/game.mp4 -o output/game_v2.pgn

# V3: Hybrid
python trackers/v3_hybrid.py videos/game.mp4 -o output/game_v3.pgn

# V4: Optical Flow
python trackers/v4_optical_flow.py videos/game.mp4 -o output/game_v4.pgn
```

---

## Architecture Summary

```
ChessWorldAI/
├── trackers/
│   ├── __init__.py
│   ├── v1_position_based.py   # 32 boxes + presence tracking
│   ├── v2_model_based.py      # YOLO classification + state matrix
│   ├── v3_hybrid.py           # V1 + V2 with consensus
│   └── v4_optical_flow.py     # Motion-based detection
├── models/
│   └── pieces.pt              # YOLO model (for V1, V2, V3)
└── output/
    └── pgn/                   # Generated PGN files
```
