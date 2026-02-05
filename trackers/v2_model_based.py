#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    VERSION 2: MODEL-BASED DETECTOR                           ║
║                    "Pure YOLO Classification" Approach                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This detector uses YOLO to classify every piece every frame.               ║
║  Moves are detected by comparing consecutive board states.                   ║
║  No assumptions about starting position - works mid-game too.               ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
                              PIPELINE DOCUMENTATION
================================================================================

OVERVIEW:
---------
This approach treats every frame as an independent observation. YOLO detects
and classifies all pieces, building a complete board state. Moves are inferred
by comparing states across frames.

The key insight: The model knows WHAT each piece is. Trust the classification.

PIPELINE STAGES:
----------------

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INITIALIZATION                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1.1. Load YOLO model trained on chess pieces                             │
│        Classes: 0=bishop, 1-6=black pieces, 7-12=white pieces              │
│                                                                             │
│   1.2. User clicks 4 corners: h1 → a1 → a8 → h8                            │
│                                                                             │
│   1.3. Compute perspective transform                                        │
│                                                                             │
│   1.4. Initialize STATE MATRIX: 64 squares × 12 piece types                │
│        Each cell is a confidence score (0.0 to 1.0)                        │
│                                                                             │
│        Example for one square:                                              │
│        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  <- Black: b,k,n,p,q,r               │
│         0.9, 0.0, 0.0, 0.0, 0.0, 0.0]  <- White: B,K,N,P,Q,R               │
│        (This square likely has a white bishop)                             │
│                                                                             │
│   1.5. Wait until we detect 30+ pieces to start (game ready)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: PER-FRAME DETECTION (runs every frame)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   2.1. Run YOLO model on frame                                             │
│        Output: list of (bounding_box, class_id, confidence)                │
│                                                                             │
│   2.2. For each detection:                                                  │
│        - Filter by confidence threshold (0.25)                             │
│        - Calculate center point                                            │
│        - Check if inside board polygon                                     │
│        - Transform to board coordinates                                    │
│        - Map to square index (0-63)                                        │
│        - Map class_id to piece type index (0-11)                           │
│                                                                             │
│   2.3. Build raw_state matrix: 64 × 12                                     │
│        For each detection, set:                                            │
│            raw_state[square][piece_type] = max(current, confidence)        │
│                                                                             │
│   Class mapping:                                                            │
│   YOLO 1→black bishop(0), 2→black king(1), 3→black knight(2)              │
│   YOLO 4→black pawn(3), 5→black queen(4), 6→black rook(5)                 │
│   YOLO 7→white Bishop(6), 8→white King(7), 9→white Knight(8)              │
│   YOLO 10→white Pawn(9), 11→white Queen(10), 12→white Rook(11)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: STATE SMOOTHING (temporal filtering)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   3.1. Apply decay to existing state:                                       │
│        state = state × decay_factor                                        │
│        (decay_factor = 0.8, so old observations fade)                      │
│                                                                             │
│   3.2. Add new observations:                                                │
│        state = max(state, raw_state × learning_rate)                       │
│        (learning_rate = 0.4)                                               │
│                                                                             │
│   3.3. Result: state matrix smoothed over time                             │
│        - Pieces that stay put accumulate high confidence                   │
│        - Pieces that move show decreasing source, increasing destination   │
│        - Noise gets averaged out                                           │
│                                                                             │
│   Why? Single-frame detection is noisy. Temporal smoothing provides:       │
│   - Robustness to occlusion (hand passing over piece)                      │
│   - Robustness to false positives (briefly detected wrong class)           │
│   - Gradual transition during moves (not instant flip)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: BOARD STATE EXTRACTION                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   4.1. For each square (0-63):                                             │
│        - Find max confidence across all 12 piece types                     │
│        - If max > occupancy_threshold (0.5): square is occupied            │
│        - piece_type = argmax(state[square])                                │
│                                                                             │
│   4.2. Build current_pieces: Dict[square] -> (symbol, color)               │
│        Example: {0: ('R', 'white'), 4: ('K', 'white'), ...}                │
│                                                                             │
│   4.3. Compare with previous_pieces to detect changes:                     │
│        - Removed: squares in previous but not in current                   │
│        - Added: squares in current but not in previous                     │
│        - Changed: same square but different piece type                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: MOVE INFERENCE                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   5.1. Analyze changes to find move candidates:                            │
│                                                                             │
│        Normal move: 1 removed + 1 added (same color)                       │
│        Capture: 1 removed (mover) + 1 removed (captured) + 1 added         │
│        Castling: 2 removed + 2 added (king and rook)                       │
│        En passant: 1 removed + 1 added + 1 removed (different rank)        │
│                                                                             │
│   5.2. For each candidate, check if legal:                                 │
│        move = chess.Move(from_square, to_square)                           │
│        if move in board.legal_moves: valid                                 │
│                                                                             │
│   5.3. Score moves based on state confidence:                              │
│        score = source_decay + destination_growth                           │
│        - source_decay: how much confidence dropped at from_square          │
│        - destination_growth: how much confidence grew at to_square         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: MOVE CONFIRMATION                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   6.1. Best scoring move becomes "pending"                                 │
│   6.2. Track pending move across frames                                    │
│   6.3. If same move is best for N consecutive frames (N=8):               │
│        - Execute move                                                       │
│        - Update python-chess board                                         │
│        - Start cooldown                                                     │
│                                                                             │
│   6.4. Cooldown prevents detecting same move multiple times                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

ADVANTAGES:
-----------
✓ Can start mid-game (doesn't assume starting position)
✓ Self-correcting - if state drifts, detection brings it back
✓ Provides rich information (piece types, not just presence)
✓ Can detect illegal moves or piece rearrangements

DISADVANTAGES:
--------------
✗ Heavily dependent on model accuracy
✗ Confused by similar pieces (bishop vs queen, especially at angles)
✗ Slower due to classification overhead
✗ Can hallucinate piece type changes

DATA STRUCTURES:
----------------
state: np.ndarray (64, 12)
    State matrix storing confidence for each piece type at each square
    Indices 0-5: black pieces (b, k, n, p, q, r)
    Indices 6-11: white pieces (B, K, N, P, Q, R)

current_pieces: Dict[int, Tuple[str, str]]
    Current board state extracted from state matrix
    Maps square index to (piece_symbol, color)

previous_pieces: Dict[int, Tuple[str, str]]
    Board state from previous frame, for comparison

CLASS MAPPING:
--------------
YOLO Class | Piece | State Index
-----------|-------|------------
    0      | (generic, ignored)
    1      | black bishop | 0
    2      | black king   | 1
    3      | black knight | 2
    4      | black pawn   | 3
    5      | black queen  | 4
    6      | black rook   | 5
    7      | white Bishop | 6
    8      | white King   | 7
    9      | white Knight | 8
    10     | white Pawn   | 9
    11     | white Queen  | 10
    12     | white Rook   | 11

================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
import sys
import argparse
import chess
import chess.pgn
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BOARD_SIZE, SQUARE_SIZE

# =============================================================================
# CONSTANTS
# =============================================================================

SQUARE_NAMES = [f"{chr(ord('a')+f)}{r+1}" for r in range(8) for f in range(8)]

# YOLO class -> (state_index, symbol, color)
# State indices: 0-5 black (b,k,n,p,q,r), 6-11 white (B,K,N,P,Q,R)
YOLO_TO_STATE = {
    1: (0, 'b', 'black'),   # black bishop
    2: (1, 'k', 'black'),   # black king
    3: (2, 'n', 'black'),   # black knight
    4: (3, 'p', 'black'),   # black pawn
    5: (4, 'q', 'black'),   # black queen
    6: (5, 'r', 'black'),   # black rook
    7: (6, 'B', 'white'),   # white Bishop
    8: (7, 'K', 'white'),   # white King
    9: (8, 'N', 'white'),   # white Knight
    10: (9, 'P', 'white'),  # white Pawn
    11: (10, 'Q', 'white'), # white Queen
    12: (11, 'R', 'white'), # white Rook
}

STATE_TO_PIECE = {
    0: ('b', 'black'), 1: ('k', 'black'), 2: ('n', 'black'),
    3: ('p', 'black'), 4: ('q', 'black'), 5: ('r', 'black'),
    6: ('B', 'white'), 7: ('K', 'white'), 8: ('N', 'white'),
    9: ('P', 'white'), 10: ('Q', 'white'), 11: ('R', 'white'),
}


def get_square_centers() -> np.ndarray:
    """Pre-compute center coordinates for all 64 squares."""
    centers = []
    for sq in range(64):
        f, r = sq % 8, sq // 8
        x = (f + 0.5) * SQUARE_SIZE
        y = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
        centers.append([x, y])
    return np.array(centers)


def interactive_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """Interactive corner selection."""
    corners = []
    display = frame.copy()
    names = ["h1 (bottom-right)", "a1 (bottom-left)", "a8 (top-left)", "h8 (top-right)"]
    
    def mouse_cb(event, x, y, flags, param):
        nonlocal corners, display
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([x, y])
            cv2.circle(display, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(display, names[len(corners)-1].split()[0], (x+15, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if len(corners) > 1:
                cv2.line(display, tuple(corners[-2]), tuple(corners[-1]), (0, 255, 0), 2)
            if len(corners) == 4:
                cv2.line(display, tuple(corners[3]), tuple(corners[0]), (0, 255, 0), 2)
                
    cv2.namedWindow("Click 4 Corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click 4 Corners", mouse_cb)
    
    print("\n" + "="*60)
    print("CORNER SELECTION")
    print("="*60)
    print("Click: h1 → a1 → a8 → h8")
    print("Controls: 'c' = confirm, 'r' = reset, 'q' = quit")
    
    while True:
        temp = display.copy()
        if len(corners) < 4:
            cv2.putText(temp, f"Click: {names[len(corners)]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            cv2.putText(temp, "Press 'c' to confirm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Click 4 Corners", temp)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('c') and len(corners) == 4:
            cv2.destroyWindow("Click 4 Corners")
            return np.array(corners, dtype=np.float32)
        elif key == ord('r'):
            corners = []
            display = frame.copy()
        elif key == ord('q'):
            cv2.destroyWindow("Click 4 Corners")
            return None


class ModelBasedDetector:
    """
    VERSION 2: Model-Based Detector
    
    Philosophy: Trust the model to classify pieces correctly.
    Build complete board state from YOLO detections each frame.
    Infer moves from state changes.
    """
    
    def __init__(self, model_path: str = 'models/pieces.pt'):
        """Initialize with YOLO model."""
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        print(f"[INIT] Model loaded: {model_path}")
        
        # ─────────────────────────────────────────────────────────────────────
        # GEOMETRY
        # ─────────────────────────────────────────────────────────────────────
        self.corners: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.inv_transform: Optional[np.ndarray] = None
        self.square_centers = get_square_centers()
        self.board_polygon: Optional[np.ndarray] = None
        
        # ─────────────────────────────────────────────────────────────────────
        # STATE MATRIX: 64 squares × 12 piece types
        # ─────────────────────────────────────────────────────────────────────
        self.state = np.zeros((64, 12), dtype=np.float32)
        
        # Temporal smoothing parameters
        self.decay_factor = 0.8    # How fast old observations fade
        self.learning_rate = 0.4   # How fast new observations are incorporated
        self.occupancy_threshold = 0.5  # Minimum confidence to consider occupied
        
        # ─────────────────────────────────────────────────────────────────────
        # EXTRACTED BOARD STATE
        # ─────────────────────────────────────────────────────────────────────
        self.current_pieces: Dict[int, Tuple[str, str]] = {}
        self.previous_pieces: Dict[int, Tuple[str, str]] = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # CHESS ENGINE
        # ─────────────────────────────────────────────────────────────────────
        self.board = chess.Board()
        self.moves: List[str] = []
        
        # ─────────────────────────────────────────────────────────────────────
        # TRACKING STATE
        # ─────────────────────────────────────────────────────────────────────
        self.is_tracking = False
        self.frame_count = 0
        self.last_move_frame = 0
        self.cooldown = 25
        
        # ─────────────────────────────────────────────────────────────────────
        # MOVE CONFIRMATION
        # ─────────────────────────────────────────────────────────────────────
        self.pending_move: Optional[chess.Move] = None
        self.pending_count = 0
        self.confirm_threshold = 8
        
        # ─────────────────────────────────────────────────────────────────────
        # DETECTION SETTINGS
        # ─────────────────────────────────────────────────────────────────────
        self.conf_threshold = 0.25  # Higher threshold for classification
        self.debug = False
        self.last_raw_detections: List[dict] = []
        
    def calibrate(self, corners: np.ndarray):
        """Compute perspective transform from clicked corners."""
        self.corners = corners.astype(np.float32)
        dst = np.array([
            [BOARD_SIZE, BOARD_SIZE],
            [0, BOARD_SIZE],
            [0, 0],
            [BOARD_SIZE, 0],
        ], dtype=np.float32)
        self.transform = cv2.getPerspectiveTransform(self.corners, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, self.corners)
        self.board_polygon = self.corners.reshape(-1, 1, 2).astype(np.int32)
        print("[CALIBRATE] Perspective transform computed")
        
    def _is_inside_board(self, x: float, y: float) -> bool:
        if self.board_polygon is None:
            return True
        return cv2.pointPolygonTest(self.board_polygon, (x, y), False) >= 0
    
    def _point_to_square(self, x: float, y: float) -> Optional[int]:
        if not self._is_inside_board(x, y):
            return None
        pt = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.transform)[0, 0]
        if 0 <= warped[0] < BOARD_SIZE and 0 <= warped[1] < BOARD_SIZE:
            dists = np.linalg.norm(self.square_centers - warped, axis=1)
            return int(np.argmin(dists))
        return None
    
    def start(self):
        """Start tracking (wait for pieces to be detected)."""
        self.board = chess.Board()
        self.moves = []
        self.state = np.zeros((64, 12), dtype=np.float32)
        self.current_pieces = {}
        self.previous_pieces = {}
        self.is_tracking = True
        self.frame_count = 0
        
        print("\n" + "="*60)
        print("WAITING FOR BOARD DETECTION...")
        print("="*60)
        print("Will start when 30+ pieces are detected")
        
    def _detect_and_update_state(self, frame: np.ndarray):
        """
        STAGE 2-3: Run YOLO, build raw state, apply temporal smoothing.
        """
        # STAGE 2: Run YOLO
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        # Build raw state matrix
        raw_state = np.zeros((64, 12), dtype=np.float32)
        self.last_raw_detections = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                
                # Skip generic class
                if cls not in YOLO_TO_STATE:
                    continue
                
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                
                if not self._is_inside_board(cx, cy):
                    continue
                
                sq = self._point_to_square(cx, cy)
                if sq is None:
                    continue
                
                state_idx, symbol, color = YOLO_TO_STATE[cls]
                
                # Update raw state with max confidence
                raw_state[sq, state_idx] = max(raw_state[sq, state_idx], conf)
                
                self.last_raw_detections.append({
                    'center': (cx, cy),
                    'square': sq,
                    'symbol': symbol,
                    'color': color,
                    'conf': conf,
                    'state_idx': state_idx,
                })
        
        # STAGE 3: Temporal smoothing
        # Decay old state
        self.state *= self.decay_factor
        
        # Add new observations
        self.state = np.maximum(self.state, raw_state * self.learning_rate)
    
    def _extract_board_state(self) -> Dict[int, Tuple[str, str]]:
        """
        STAGE 4: Extract discrete board state from state matrix.
        """
        pieces = {}
        for sq in range(64):
            max_conf = np.max(self.state[sq])
            if max_conf >= self.occupancy_threshold:
                piece_idx = np.argmax(self.state[sq])
                pieces[sq] = STATE_TO_PIECE[piece_idx]
        return pieces
    
    def _find_move_from_changes(self) -> Optional[chess.Move]:
        """
        STAGE 5: Analyze changes to find best legal move.
        """
        removed = set(self.previous_pieces.keys()) - set(self.current_pieces.keys())
        added = set(self.current_pieces.keys()) - set(self.previous_pieces.keys())
        
        if not removed or not added:
            return None
        
        # Score each legal move
        best_move = None
        best_score = 0
        
        whose_turn = 'white' if self.board.turn else 'black'
        
        for move in self.board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            score = 0
            
            # From square should be removed
            if from_sq in removed:
                # Check color matches turn
                if from_sq in self.previous_pieces:
                    _, color = self.previous_pieces[from_sq]
                    if color == whose_turn:
                        score += 3
                    else:
                        continue  # Wrong color
                else:
                    score += 2
            else:
                continue  # Source not removed
            
            # To square should be added
            if to_sq in added:
                if to_sq in self.current_pieces:
                    _, color = self.current_pieces[to_sq]
                    if color == whose_turn:
                        score += 3
                    else:
                        continue  # Wrong color at destination
                else:
                    score += 2
            else:
                # Maybe capture - piece replaced
                if to_sq in self.current_pieces and to_sq in self.previous_pieces:
                    curr_color = self.current_pieces[to_sq][1]
                    prev_color = self.previous_pieces[to_sq][1]
                    if curr_color != prev_color:
                        score += 2  # Color changed = capture
            
            # Check confidence changes in state matrix
            from_max_now = np.max(self.state[from_sq])
            to_max_now = np.max(self.state[to_sq])
            
            if from_max_now < 0.3:  # Source is clearly empty
                score += 1
            if to_max_now > 0.5:  # Destination clearly has piece
                score += 1
            
            # Castling bonus
            if self.board.is_castling(move):
                rook_info = {
                    chess.G1: (chess.H1, chess.F1),
                    chess.C1: (chess.A1, chess.D1),
                    chess.G8: (chess.H8, chess.F8),
                    chess.C8: (chess.A8, chess.D8),
                }
                if move.to_square in rook_info:
                    rf, rt = rook_info[move.to_square]
                    if rf in removed and rt in added:
                        score += 3
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_score >= 4:
            return best_move
        return None
    
    def process(self, frame: np.ndarray) -> Optional[str]:
        """Main processing pipeline."""
        if not self.is_tracking:
            return None
        
        self.frame_count += 1
        
        # STAGE 2-3: Detect and update state
        self._detect_and_update_state(frame)
        
        # STAGE 4: Extract board state
        self.previous_pieces = self.current_pieces.copy()
        self.current_pieces = self._extract_board_state()
        
        # Wait for enough pieces
        if len(self.current_pieces) < 30:
            return None
        
        # First time we have enough pieces
        if not self.previous_pieces:
            print(f"\n[{self.frame_count}] Detected {len(self.current_pieces)} pieces - TRACKING STARTED")
            return None
        
        # Cooldown
        if self.frame_count - self.last_move_frame < self.cooldown:
            return None
        
        # STAGE 5: Find move
        candidate = self._find_move_from_changes()
        
        if candidate is None:
            self.pending_move = None
            self.pending_count = 0
            return None
        
        # Debug
        if self.debug:
            removed = set(self.previous_pieces.keys()) - set(self.current_pieces.keys())
            added = set(self.current_pieces.keys()) - set(self.previous_pieces.keys())
            print(f"  [{self.frame_count}] Removed: {[SQUARE_NAMES[s] for s in removed]}")
            print(f"  [{self.frame_count}] Added: {[SQUARE_NAMES[s] for s in added]}")
            print(f"  Candidate: {self.board.san(candidate)}")
        
        # STAGE 6: Confirmation
        if candidate == self.pending_move:
            self.pending_count += 1
            if self.pending_count >= self.confirm_threshold:
                return self._execute_move(candidate)
        else:
            self.pending_move = candidate
            self.pending_count = 1
        
        return None
    
    def _execute_move(self, move: chess.Move) -> str:
        """Execute confirmed move."""
        san = self.board.san(move)
        self.board.push(move)
        self.moves.append(san)
        self.last_move_frame = self.frame_count
        self.pending_move = None
        self.pending_count = 0
        
        print(f"[{self.frame_count}] Move #{len(self.moves)}: {san}")
        return san
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Create visualization."""
        vis = frame.copy()
        
        if self.corners is not None:
            cv2.polylines(vis, [self.corners.astype(np.int32)], True, (0, 255, 0), 2)
        
        # Draw detected pieces
        if self.inv_transform is not None:
            for sq, (symbol, color) in self.current_pieces.items():
                f, r = sq % 8, sq // 8
                wx = (f + 0.5) * SQUARE_SIZE
                wy = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
                pt = np.array([[wx, wy]], dtype=np.float32)
                center = cv2.perspectiveTransform(pt[np.newaxis], self.inv_transform)[0, 0]
                cx, cy = int(center[0]), int(center[1])
                
                conf = np.max(self.state[sq])
                box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                
                # Fade based on confidence
                alpha = min(1.0, conf)
                box_color = tuple(int(c * alpha) for c in box_color)
                
                cv2.rectangle(vis, (cx-14, cy-14), (cx+14, cy+14), box_color, 2)
                cv2.putText(vis, symbol.upper(), (cx-6, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Status
        turn = "White" if self.board.turn else "Black"
        cv2.putText(vis, f"VERSION 2: Model-Based | Pieces: {len(self.current_pieces)} | {turn}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.moves:
            recent = self.moves[-8:]
            cv2.putText(vis, " ".join(recent), (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis
    
    def get_pgn(self, event_name: str = "Model Detector") -> str:
        """Generate PGN."""
        game = chess.pgn.Game()
        game.headers["Event"] = event_name
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        
        node = game
        board = chess.Board()
        for san in self.moves:
            try:
                move = board.parse_san(san)
                node = node.add_variation(move)
                board.push(move)
            except:
                continue
        
        if board.is_checkmate():
            game.headers["Result"] = "1-0" if not board.turn else "0-1"
        else:
            game.headers["Result"] = "*"
        
        return str(game)


def main():
    parser = argparse.ArgumentParser(
        description="VERSION 2: Model-Based Detector (Pure YOLO Classification)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PIPELINE:
  1. YOLO detects and classifies all pieces each frame
  2. Build 64x12 state matrix with temporal smoothing
  3. Extract discrete board state from matrix
  4. Compare states to detect piece movements
  5. Match to legal moves and confirm
        """
    )
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--output', '-o', help='Output PGN file')
    parser.add_argument('--model', '-m', default='models/pieces.pt')
    parser.add_argument('--crop', type=float, default=0.45)
    parser.add_argument('--debug', '-d', action='store_true')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    model_path = args.model
    if not Path(model_path).exists():
        for alt in ['models/pieces.pt', 'models/pieces_trained.pt']:
            if Path(alt).exists():
                model_path = alt
                break
    
    output_path = args.output or f"output/pgn/{video_path.stem}_v2_model.pgn"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_height = int(height * args.crop)
    
    print("\n" + "="*60)
    print("VERSION 2: MODEL-BASED DETECTOR")
    print("="*60)
    print(f"Video: {video_path.name}")
    print(f"Model: {model_path}")
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        sys.exit(1)
    frame = frame[:crop_height, :]
    
    detector = ModelBasedDetector(model_path=model_path)
    detector.debug = args.debug
    
    corners = interactive_corners(frame)
    if corners is None:
        cap.release()
        sys.exit(0)
    detector.calibrate(corners)
    detector.start()
    
    cv2.namedWindow("Model Detector", cv2.WINDOW_NORMAL)
    frame_delay = int(1000 / fps)
    paused = False
    frame_num = 0
    
    print("\nControls: SPACE=pause, D=debug, Q=quit\n")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            frame = frame[:crop_height, :]
            detector.process(frame)
        
        vis = detector.visualize(frame)
        
        pct = frame_num / total_frames if total_frames else 0
        bar_y = vis.shape[0] - 15
        bar_w = vis.shape[1] - 20
        cv2.rectangle(vis, (10, bar_y), (10 + bar_w, bar_y + 8), (50, 50, 50), -1)
        cv2.rectangle(vis, (10, bar_y), (10 + int(bar_w * pct), bar_y + 8), (0, 255, 0), -1)
        
        cv2.imshow("Model Detector", vis)
        
        key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d'):
            detector.debug = not detector.debug
    
    cap.release()
    cv2.destroyAllWindows()
    
    pgn_content = detector.get_pgn(f"Game: {video_path.stem}")
    with open(output_path, 'w') as f:
        f.write(pgn_content)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total moves: {len(detector.moves)}")
    print(f"PGN: {output_path}")
    print(f"\nMoves: {' '.join(detector.moves)}")


if __name__ == "__main__":
    main()
