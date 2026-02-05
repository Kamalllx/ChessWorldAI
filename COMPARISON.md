# ChessWorldAI - Complete Tracker Comparison

## Directory Structure

```
ChessWorldAI/
├── trackers/                      # NEW: Clean, documented versions
│   ├── v1_position_based.py       # Position tracking (32 boxes)
│   ├── v2_model_based.py          # YOLO classification + state matrix
│   ├── v3_hybrid.py               # Hybrid (V1 + V2 consensus)
│   ├── v4_optical_flow.py         # Motion-based detection
│   └── README.md                  # Documentation for new versions
│
├── legacy/                        # OLD: Your original implementations
│   ├── run_camerachess.py         # CameraChess approach (state matrix)
│   ├── run_tracker_v2.py          # Fast tracker with PieceDetector
│   ├── run_tracker_v3.py          # Direct YOLO, better thresholds
│   ├── run_tracker_v4.py          # Ensemble detection
│   ├── run_tracker_v5.py          # Detect-Lock-Track phases
│   ├── run_tracker_v2_enhanced.py # Enhanced clustering
│   └── README.md                  # Documentation for legacy versions
│
├── videos/                        # Test videos
│   ├── game_1.mp4
│   ├── game_2.mp4
│   ├── game_3.mp4
│   ├── game_4.mp4
│   └── game_5.mp4
│
└── output/
    └── pgn/                       # Generated PGN files
```

---

## Code Explanations

### 1. run_camerachess.py - CameraChess State Matrix Approach

**Based on:** Pbatch/CameraChessWeb GitHub repository

**Core Concept:**
- Uses 64×12 **state matrix** (64 squares × 12 piece types)
- Each cell contains confidence score (0.0 to 1.0)
- **Decay system**: Old observations fade over time
- **Move scoring**: Compares from/to confidence changes

**Pipeline:**
```
1. YOLO Detection
   ↓
2. Map detections to state matrix
   state[square][piece_type] = confidence
   ↓
3. Apply decay to all squares
   state *= decay_factor (0.7)
   ↓
4. Add new detections
   state[sq][piece] = max(state[sq][piece], new_conf)
   ↓
5. Score moves by confidence changes
   score = (from_square dropped?) + (to_square increased?)
   ↓
6. Confirm over multiple frames
```

**Key Parameters:**
```python
decay_factor = 0.7          # How fast old observations fade
from_threshold = 0.3        # Source must be below this
to_threshold = 0.5          # Destination must be above this
confirm_frames = 8          # Frames to confirm
cooldown = 15               # Frames between moves
```

**Strengths:**
- Smooth temporal handling (decay)
- Can work mid-game (no starting position assumed)
- Model-based validation

**Weaknesses:**
- Heavy on YOLO classification accuracy
- Can drift if model consistently wrong
- Complex state management

---

### 2. run_tracker_v2.py - Fast Tracker with PieceDetector Wrapper

**Core Concept:**
- Uses **PieceDetector wrapper class** (your custom detector)
- **Voting system**: Accumulates detections over 2 frames
- Very **fast response** (small windows, quick cooldown)
- **Piece type validation**: Matches detected type to expected piece

**Pipeline:**
```
1. PieceDetector.detect(frame, corners)
   ↓
2. Map centers to squares
   ↓
3. Vote accumulation (2 frames)
   detection_votes[square][piece_type] += 1
   ↓
4. Analyze every 2 frames
   - Get most-voted piece per square (>20% threshold)
   - Compare to expected board_state
   - Find vacated/appeared squares
   ↓
5. Score legal moves
   - Basic: vacated source (+3), appeared dest (+3)
   - Type match: detected piece matches expected (+2)
   - Castling: both king and rook moved
   ↓
6. Confirm and execute
```

**Key Parameters:**
```python
vote_window = 2             # Very fast accumulation
cooldown = 4                # Very fast cooldown
vote_threshold = 0.2        # Only 20% votes needed
```

**Strengths:**
- FASTEST response time
- Uses piece type for validation
- Simple, clean code

**Weaknesses:**
- Depends on PieceDetector wrapper
- Low threshold can cause false positives
- No temporal smoothing

---

### 3. run_tracker_v3.py - Direct YOLO Loading

**Core Concept:**
- Loads **YOLO directly** (no wrapper)
- Better **confidence thresholds** (0.3)
- Improved piece type matching
- Still uses voting but with higher quality

**Pipeline:**
```
1. Direct YOLO inference
   results = model(frame, conf=0.3)
   ↓
2. Parse results manually
   - Extract boxes, classes, confidences
   - Map to squares
   ↓
3. Vote accumulation (5 frames)
   detection_votes[square][piece_class] += 1
   ↓
4. Analyze every 5 frames
   - Require >60% votes (stricter than V2)
   - Compare to board_state
   ↓
5. Score with piece type validation
   - Same as V2 but stricter thresholds
   ↓
6. Execute with cooldown=8
```

**Key Parameters:**
```python
conf_threshold = 0.3        # Higher than V2
vote_window = 5             # Longer window
vote_threshold = 0.6        # Stricter (60%)
cooldown = 8                # Moderate cooldown
```

**Strengths:**
- No wrapper dependency
- Better quality detection
- More robust to noise

**Weaknesses:**
- Slightly slower than V2
- Still basic voting logic

---

### 4. run_tracker_v4.py - Ensemble Detection

**Core Concept:**
- **Multiple YOLO models** run in parallel
- Combines predictions from all models
- **Clustering** to merge close detections
- Consensus-based decisions

**Pipeline:**
```
1. Load multiple models
   models = [YOLO(path1), YOLO(path2), ...]
   ↓
2. Run all models on frame
   for model in models:
       detections += model.detect()
   ↓
3. Cluster detections (dist < 35px)
   - Keep highest confidence in each cluster
   ↓
4. Vote accumulation (5 frames)
   ↓
5. Standard analysis
   ↓
6. Execute
```

**Key Parameters:**
```python
cluster_distance = 35       # Max distance to merge
vote_window = 5
cooldown = 8
```

**Strengths:**
- Most robust (ensemble voting)
- Can use different model architectures
- Self-correcting

**Weaknesses:**
- SLOWEST (multiple models)
- Needs multiple trained models
- Complex setup

---

### 5. run_tracker_v5.py - Detect-Lock-Track Phases

**Core Concept:**
- **State machine** with distinct phases
- CALIBRATE → DETECT → LOCKED → TRACKING
- Explicitly waits to detect 30+ pieces before tracking
- More structured approach

**Pipeline:**
```
Phase 1: CALIBRATE
   User clicks corners
   ↓
Phase 2: DETECT
   Accumulate detections (5 frames)
   Wait until 30+ pieces found
   ↓
Phase 3: LOCKED
   Board state locked
   Start tracking moves
   ↓
Phase 4: TRACKING
   Accumulate detections (5 frames)
   Analyze vacated/appeared
   Score moves
   Confirm and execute
   ↓
   Repeat Phase 4
```

**Key Parameters:**
```python
detect_window = 5           # Frames to accumulate
min_pieces = 30             # Minimum to start tracking
conf_threshold = 0.25
cluster_dist = 40
cooldown = 6
```

**Strengths:**
- Clear phase separation
- Ensures board is ready before tracking
- Good for auto-start scenarios

**Weaknesses:**
- Can get stuck in DETECT phase
- More complex state management

---

### 6. run_tracker_v2_enhanced.py - Enhanced Clustering

**Core Concept:**
- Enhanced version of V2
- Better **clustering** (smaller distance = 30px)
- **Detection memory**: Remembers pieces for 45 frames
- Confidence weighting in votes

**Pipeline:**
```
1. YOLO detection (direct)
   ↓
2. Cluster detections (dist < 30px)
   - Avoid merging adjacent pieces
   ↓
3. Detection memory
   - Remember last seen (class, conf, frame)
   - Decay after 45 frames
   ↓
4. Vote accumulation (4 frames)
   - Store (class, confidence) tuples
   ↓
5. Analyze with memory
   - Use memory if square not detected recently
   ↓
6. Score with type matching
   ↓
7. Execute with cooldown=8
```

**Key Parameters:**
```python
cluster_distance = 30       # Tighter clustering
vote_window = 4
memory_decay_frames = 45    # Long memory
confirm_needed = 2          # Fast confirmation
cooldown = 8
```

**Strengths:**
- Best of V2 + improvements
- Handles occlusion well (memory)
- Tight clustering avoids merges

**Weaknesses:**
- Memory can hold stale data
- More complex than base V2

---

## Comparison Matrix

| Feature | CameraChess | V2 | V3 | V4 | V5 | V2E |
|---------|-------------|----|----|----|----|-----|
| **Detection Method** | YOLO | PieceDetector | YOLO | Multi-YOLO | YOLO | YOLO |
| **Temporal Strategy** | Decay | Voting | Voting | Voting | Voting | Voting+Memory |
| **Window Size** | N/A | 2 | 5 | 5 | 5 | 4 |
| **Cooldown** | 15 | 4 | 8 | 8 | 6 | 8 |
| **Clustering** | No | No | No | Yes | Yes | Yes |
| **Type Validation** | Matrix | Yes | Yes | Yes | Yes | Yes |
| **Starting Position** | No | Yes | Yes | Yes | Detect | Yes |
| **Speed** | Medium | FAST | Medium | SLOW | Medium | Medium |
| **Robustness** | High | Low | Medium | HIGH | Medium | High |
| **Complexity** | High | Low | Low | High | Medium | Medium |

---

## Performance Characteristics

### Speed Ranking (Fastest to Slowest)
1. **V2** - 2-frame window, 4-frame cooldown
2. **V5** - 5-frame window, 6-frame cooldown
3. **V2E** - 4-frame window, 8-frame cooldown
4. **V3** - 5-frame window, 8-frame cooldown
5. **CameraChess** - Decay-based, 15-frame cooldown
6. **V4** - Ensemble, 5-frame window, 8-frame cooldown

### Accuracy Ranking (Most Accurate to Least)
1. **V4** - Ensemble consensus
2. **V2E** - Memory + clustering
3. **CameraChess** - State matrix with decay
4. **V3** - Strict thresholds
5. **V5** - Phase-based
6. **V2** - Fast but less robust

### Complexity Ranking (Simplest to Most Complex)
1. **V2** - Basic voting
2. **V3** - Direct YOLO + voting
3. **V5** - Phase machine
4. **V2E** - Memory system
5. **CameraChess** - State matrix + decay
6. **V4** - Ensemble + clustering

---

## Which Version to Use?

### For Production
**Recommended: V4 (Ensemble)** or **V2E (Enhanced)**
- Best accuracy through consensus
- Robust to noise and occlusion

### For Speed
**Recommended: V2 (Fast Tracker)**
- Fastest response time
- Good for real-time applications

### For Research/Learning
**Recommended: CameraChess**
- Implements state-of-the-art decay system
- Good foundation for improvements

### For Clean Codebase
**Recommended: V3**
- Simple, direct approach
- Easy to understand and modify

---

## Migration to New Trackers

Your **NEW trackers/** directory contains:
1. **v1_position_based.py** - Similar to V2 but presence-only (no classification)
2. **v2_model_based.py** - Similar to CameraChess (state matrix approach)
3. **v3_hybrid.py** - Combines V1 + V2 (best of both worlds)
4. **v4_optical_flow.py** - Completely new approach (motion detection)

**Recommendation:** Use the new trackers for future work, keep legacy for comparison.
