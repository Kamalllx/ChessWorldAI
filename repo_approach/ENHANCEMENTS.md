# ChessVision Pro - Enhanced with Position-Based Tracking

## 🎯 What's New

This is an **enhanced version** of the original CameraChessWeb repository with our custom **Position-Based Tracking** algorithm integrated!

### Original CameraChessWeb
- **Score-Based Approach**: Uses detection confidence scores and move scoring
- **Works on**: Any chess position mid-game
- **Accuracy**: Good but can have false positives

### Our Position-Based Tracking (NEW! 🎯)
- **Tracking Approach**: Starts with known initial position, tracks piece movement
- **Detection Method**: Presence/absence detection with sliding window (15 frames)
- **Works best on**: Complete games from starting position
- **Accuracy**: More accurate and consistent for full games!

---

## 🚀 How It Works

### Position-Based Tracking Algorithm

```
1. Initialize: Start with known 32 pieces in starting position
2. Detect Presence: Track each square's occupancy over 15-frame window
3. Find Changes:
   - VACATED: Square that was occupied is now empty (< 0.3 confidence)
   - ARRIVED: Square that was empty is now occupied (> 0.6 confidence)
4. Match Moves: Find legal move matching the VACATED → ARRIVED pattern
5. Execute: Update board state and expected position
6. Repeat: Continue tracking from new position
```

### Key Thresholds

```typescript
DETECTION_THRESHOLD = 0.15   // Minimum to consider detection valid
VACATED_THRESHOLD = 0.3      // Square considered empty
ARRIVED_THRESHOLD = 0.6      // Square considered occupied
CONFIRM_FRAMES = 4           // Frames needed to confirm change
COOLDOWN_FRAMES = 15         // Cooldown after move
WINDOW_SIZE = 15             // Sliding window size
```

---

## 🎮 Using the App

### Step 1: Upload Video
Click **"Choose Video"** and select your chess game recording.

### Step 2: Toggle Tracking Mode
- **🎯 Position Tracking** (Default) - Our new approach
- **📊 Score-Based** - Original algorithm

### Step 3: Find Corners
- Auto-detect with **"Find Corners"** button
- Or **drag the red markers** (a1, h1, h8, a8) manually

### Step 4: Detect Position
Click **"Find FEN"** to detect the starting position.

### Step 5: Play Analysis
Hit **Play** and watch the moves being detected!

---

## 🏗️ Architecture Changes

### New Files Added

```
src/
├── utils/
│   ├── positionTracker.tsx      # Position-based tracking logic
│   └── findPiecesHybrid.tsx     # Hybrid detection (original + position)
└── slices/
    └── settingsSlice.tsx         # Settings state (tracking mode toggle)
└── components/
    └── common/
        └── trackingModeButton.tsx # UI toggle button
```

### Modified Files

```
src/
├── store.tsx                     # Added settingsReducer
├── slices/index.tsx             # Export settingsReducer
├── components/
│   ├── common/
│   │   ├── video.tsx            # Use findPiecesHybrid
│   │   ├── index.tsx            # Export TrackingModeButton
│   │   ├── sidebar.tsx          # Custom branding
│   │   ├── marker.tsx           # Improved dragging
│   │   └── container.tsx        # Modern gradient background
│   └── upload/
│       └── uploadSidebar.tsx    # Added mode toggle + explanation
└── index.html                   # Custom title
```

---

## 🔄 Comparison: Both Approaches

| Feature | Position Tracking 🎯 | Score-Based 📊 |
|---------|---------------------|----------------|
| **Starting Position** | Required | Not required |
| **Accuracy** | Higher for full games | Good for any position |
| **False Positives** | Very low | Moderate |
| **Speed** | Fast | Fast |
| **Best For** | Recording full games | Mid-game analysis |

---

## 📊 Position Tracking Details

### How It Tracks Each Square

```typescript
class SquareTracker {
  // Each square maintains:
  - history: [0.1, 0.3, 0.7, 0.8, 0.9]  // 15-frame window
  - lastChangeFrame: 42                   // When last changed
  - confirmedEmpty: false                 // Confirmed state
  - confirmedOccupied: true
}
```

### Move Detection Logic

```typescript
1. Scan all 64 squares
2. Find vacated squares (expected occupied → now empty)
3. Find arrived squares (expected empty → now occupied)
4. Match to legal moves:
   - Score = (1 - from_avg) * 3 + to_avg * 3 + bonus
5. Execute highest scoring legal move
6. Update expected position
```

### Special Move Handling

- **Castling**: Detects king+rook movement patterns
- **En Passant**: Removes captured pawn from different square
- **Promotion**: Handles pawn reaching 8th rank
- **Captures**: Updates expected position

---

## 🎨 UI Improvements

### Modern Design
- **Gradient Background**: Blue gradient instead of plain dark
- **Custom Branding**: "ChessVision Pro" instead of "ChessCam"
- **Draggable Markers**: Improved styling with better visibility
- **Instructions Panel**: Step-by-step guide in sidebar
- **Mode Explanation**: Shows which algorithm is active

### Direct-to-Upload
- No more login/authentication required
- No home page - goes straight to upload
- Simplified routing (upload, export, FAQ only)

---

## 🧪 Testing Position Tracking

1. **Record or find a chess game video** from starting position
2. Load it in the app
3. Toggle to **🎯 Position Tracking** mode
4. Find corners (auto or manual)
5. Play the video
6. Watch the status: "Frames since move: X"
7. Moves should be detected accurately!

---

## 💡 Technical Insights

### Why Position Tracking Works Better

**Original Score-Based:**
```
Problem: Piece misclassification (knight vs bishop)
Result: Wrong moves or false positives
```

**Position Tracking:**
```
Solution: Only care about PRESENCE, not piece type
Result: More reliable move detection
```

### The Sliding Window Advantage

```
Without sliding window:
Frame 1: 0.9 (occupied) ← noise spike!
Frame 2: 0.1 (empty)
Result: False move detection

With sliding window (15 frames):
Average: [0.9, 0.8, 0.2, 0.1, 0.1...] → 0.3
Result: Confirmed vacancy, reliable detection
```

---

## 🛠️ Development

```bash
# Install dependencies
npm install

# Run development server
npm run start

# Build for production
npm run build

# Deploy to GitHub Pages
npm run deploy
```

---

## 📝 Future Improvements

- [ ] Mid-game position entry for Position Tracking
- [ ] Auto-detect if game starts from initial position
- [ ] Combine both approaches (hybrid scoring)
- [ ] Mobile optimization
- [ ] Batch video processing

---

## 🙏 Credits

- **Original CameraChessWeb**: [Pbatch/CameraChessWeb](https://github.com/Pbatch/CameraChessWeb)
- **Position-Based Tracking**: Our custom enhancement
- **YOLO Models**: LeYOLO architecture optimized for chess pieces

---

## 📄 License

MIT License (same as original CameraChessWeb)

---

## 🚀 Live Demo

Try it now: **http://localhost:5173**

Toggle between modes and see the difference!
