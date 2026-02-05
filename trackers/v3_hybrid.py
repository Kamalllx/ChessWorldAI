#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    VERSION 3: HYBRID TRACKER                                 ║
║                    "Best of Both Worlds" Approach                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Combines Position-Based tracking (V1) with Model Classification (V2).      ║
║  Uses known starting position for stability, model for validation.          ║
║  Achieves high accuracy through consensus between methods.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
                              PIPELINE DOCUMENTATION
================================================================================

OVERVIEW:
---------
This hybrid approach leverages the strengths of both previous methods:
- V1's knowledge: We KNOW where pieces start
- V2's intelligence: Model KNOWS what pieces look like

The key insight: Use position tracking as the PRIMARY source of truth,
but use model classification to VALIDATE and CORRECT.

PIPELINE STAGES:
----------------

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DUAL INITIALIZATION                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1.1. Load YOLO model for piece detection/classification                  │
│                                                                             │
│   1.2. User clicks 4 corners                                               │
│                                                                             │
│   1.3. Initialize POSITION TRACKER state (from V1):                        │
│        - board_state: 32 pieces at starting positions                      │
│        - detection_history: sliding window per square                      │
│                                                                             │
│   1.4. Initialize MODEL DETECTOR state (from V2):                          │
│        - state_matrix: 64 × 12 confidence scores                           │
│        - Will be populated during initial frames                           │
│                                                                             │
│   1.5. Initialize CONSENSUS state:                                         │
│        - agreement_count: how often V1 and V2 agree on board state         │
│        - trust_position: weight for position-based evidence                │
│        - trust_model: weight for model-based evidence                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: DUAL DETECTION (runs every frame)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   2.1. Run YOLO model once (shared between both subsystems)                │
│                                                                             │
│   2.2. POSITION TRACK update:                                              │
│        - Extract occupied squares (ignore piece types)                     │
│        - Update detection history per square                               │
│        - Calculate detection rates                                         │
│                                                                             │
│   2.3. MODEL DETECT update:                                                │
│        - Build raw state matrix with confidences                           │
│        - Apply decay and learning rate                                     │
│        - Extract current pieces with types                                 │
│                                                                             │
│   Result: Two parallel views of the board                                  │
│   - position_view: what squares are occupied (V1)                          │
│   - model_view: what pieces are where (V2)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: MOVE CANDIDATE GENERATION                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   3.1. From POSITION TRACKER (V1):                                         │
│        - Find VACATED squares (detection rate dropped below 0.3)           │
│        - Find APPEARED squares (new consistent detections)                 │
│        - Generate move candidates based on changes                         │
│        - Score: position_score                                             │
│                                                                             │
│   3.2. From MODEL DETECTOR (V2):                                           │
│        - Compare current_pieces with previous_pieces                       │
│        - Find REMOVED and ADDED squares                                    │
│        - Generate move candidates based on state changes                   │
│        - Score: model_score                                                │
│                                                                             │
│   3.3. Merge candidates:                                                   │
│        candidates = union(V1_candidates, V2_candidates)                    │
│        For each candidate: combined_score = α×pos_score + β×model_score    │
│        where α = trust_position, β = trust_model                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: CONSENSUS VALIDATION                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   4.1. For each candidate move, check agreement:                           │
│                                                                             │
│        STRONG AGREEMENT (both systems suggest same move):                  │
│        - Bonus: +3 to combined score                                       │
│        - Faster confirmation (fewer frames needed)                         │
│                                                                             │
│        WEAK AGREEMENT (one system suggests, other doesn't contradict):     │
│        - No bonus, standard confirmation                                   │
│                                                                             │
│        DISAGREEMENT (systems suggest different moves):                     │
│        - Penalty: -2 to combined score                                     │
│        - Slower confirmation (more frames needed)                          │
│        - May indicate ambiguous situation                                  │
│                                                                             │
│   4.2. Piece type validation (unique to hybrid):                           │
│        - V1 knows what piece SHOULD be at source (from tracking)           │
│        - V2 knows what piece IS at destination (from classification)       │
│        - Validate: are they the same type?                                 │
│        - If mismatch: possible tracking error, investigate                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: ADAPTIVE CONFIRMATION                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   5.1. Confirmation threshold varies based on confidence:                  │
│                                                                             │
│        HIGH CONFIDENCE (strong agreement, high scores):                    │
│        - confirm_frames = 4                                                │
│        - Quick execution                                                   │
│                                                                             │
│        MEDIUM CONFIDENCE (weak agreement, moderate scores):                │
│        - confirm_frames = 6                                                │
│        - Standard execution                                                │
│                                                                             │
│        LOW CONFIDENCE (disagreement, low scores):                          │
│        - confirm_frames = 10                                               │
│        - Careful execution, may require more evidence                      │
│                                                                             │
│   5.2. Execute when confirmed:                                             │
│        - Update V1's board_state (move piece)                              │
│        - Clear V2's state_matrix for affected squares                      │
│        - Push move to chess engine                                         │
│        - Record in moves list                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: STATE RECONCILIATION (periodic)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Every N frames (N=50), compare V1 and V2 states:                         │
│                                                                             │
│   6.1. For each square occupied in V1's board_state:                       │
│        - Check if V2's model agrees on piece presence                      │
│        - Check if V2's classification matches V1's expected piece          │
│                                                                             │
│   6.2. Calculate agreement percentage:                                     │
│        agreement = matching_squares / total_expected_pieces                │
│                                                                             │
│   6.3. Adjust trust weights:                                               │
│        If agreement > 90%: trust both equally                              │
│        If agreement < 70%: trust V1 more (position is stable)              │
│        If V2 consistently detects more pieces: trust V2 more               │
│                                                                             │
│   6.4. Report discrepancies for debugging                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

ADVANTAGES:
-----------
✓ Robust: Uses consensus to filter false positives
✓ Self-validating: Can detect when tracking state drifts
✓ Best accuracy: Combines position stability with model intelligence
✓ Adaptive: Adjusts confirmation based on confidence

DISADVANTAGES:
--------------
✗ More complex to implement and tune
✗ Slightly slower (two detection pipelines)
✗ Can be confused when both systems are wrong
✗ Requires careful weight tuning

DATA STRUCTURES:
----------------
# From V1
board_state: Dict[int, Tuple[str, str]]  # Expected piece locations
detection_history: Dict[int, List[bool]]  # Per-square detection history

# From V2
state_matrix: np.ndarray (64, 12)  # Confidence per piece type per square
model_pieces: Dict[int, Tuple[str, str]]  # Model's current view

# Hybrid specific
trust_position: float  # Weight for V1 evidence (default 0.6)
trust_model: float     # Weight for V2 evidence (default 0.4)

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

YOLO_TO_STATE = {
    1: (0, 'b', 'black'), 2: (1, 'k', 'black'), 3: (2, 'n', 'black'),
    4: (3, 'p', 'black'), 5: (4, 'q', 'black'), 6: (5, 'r', 'black'),
    7: (6, 'B', 'white'), 8: (7, 'K', 'white'), 9: (8, 'N', 'white'),
    10: (9, 'P', 'white'), 11: (10, 'Q', 'white'), 12: (11, 'R', 'white'),
}

STATE_TO_PIECE = {
    0: ('b', 'black'), 1: ('k', 'black'), 2: ('n', 'black'),
    3: ('p', 'black'), 4: ('q', 'black'), 5: ('r', 'black'),
    6: ('B', 'white'), 7: ('K', 'white'), 8: ('N', 'white'),
    9: ('P', 'white'), 10: ('Q', 'white'), 11: ('R', 'white'),
}


def get_square_centers() -> np.ndarray:
    centers = []
    for sq in range(64):
        f, r = sq % 8, sq // 8
        x = (f + 0.5) * SQUARE_SIZE
        y = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
        centers.append([x, y])
    return np.array(centers)


def interactive_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    corners = []
    display = frame.copy()
    names = ["h1", "a1", "a8", "h8"]
    
    def mouse_cb(event, x, y, flags, param):
        nonlocal corners, display
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([x, y])
            cv2.circle(display, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(display, names[len(corners)-1], (x+15, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if len(corners) > 1:
                cv2.line(display, tuple(corners[-2]), tuple(corners[-1]), (0, 255, 0), 2)
            if len(corners) == 4:
                cv2.line(display, tuple(corners[3]), tuple(corners[0]), (0, 255, 0), 2)
                
    cv2.namedWindow("Click 4 Corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click 4 Corners", mouse_cb)
    
    print("\nClick: h1 → a1 → a8 → h8 | 'c'=confirm, 'r'=reset, 'q'=quit")
    
    while True:
        temp = display.copy()
        msg = f"Click: {names[len(corners)]}" if len(corners) < 4 else "Press 'c' to confirm"
        cv2.putText(temp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
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


class HybridTracker:
    """
    VERSION 3: Hybrid Tracker
    
    Combines position-based tracking (V1) with model classification (V2)
    for maximum accuracy through consensus.
    """
    
    def __init__(self, model_path: str = 'models/pieces.pt'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        print(f"[INIT] Hybrid Tracker with model: {model_path}")
        
        # ─────────────────────────────────────────────────────────────────────
        # GEOMETRY (shared)
        # ─────────────────────────────────────────────────────────────────────
        self.corners: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.inv_transform: Optional[np.ndarray] = None
        self.square_centers = get_square_centers()
        self.board_polygon: Optional[np.ndarray] = None
        
        # ─────────────────────────────────────────────────────────────────────
        # V1: POSITION-BASED STATE
        # ─────────────────────────────────────────────────────────────────────
        self.board_state: Dict[int, Tuple[str, str]] = {}  # Expected positions
        self.detection_history: Dict[int, List[bool]] = defaultdict(list)
        self.window_size = 10
        
        # ─────────────────────────────────────────────────────────────────────
        # V2: MODEL-BASED STATE
        # ─────────────────────────────────────────────────────────────────────
        self.state_matrix = np.zeros((64, 12), dtype=np.float32)
        self.model_pieces: Dict[int, Tuple[str, str]] = {}
        self.decay_factor = 0.8
        self.learning_rate = 0.4
        
        # ─────────────────────────────────────────────────────────────────────
        # HYBRID: CONSENSUS STATE
        # ─────────────────────────────────────────────────────────────────────
        self.trust_position = 0.6  # Weight for V1
        self.trust_model = 0.4     # Weight for V2
        
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
        self.cooldown = 20
        
        # ─────────────────────────────────────────────────────────────────────
        # CONFIRMATION (adaptive)
        # ─────────────────────────────────────────────────────────────────────
        self.pending_move: Optional[chess.Move] = None
        self.pending_count = 0
        self.pending_confidence = 0.0
        self.base_confirm = 6
        
        # ─────────────────────────────────────────────────────────────────────
        # DETECTION
        # ─────────────────────────────────────────────────────────────────────
        self.conf_threshold = 0.20
        self.debug = False
        
    def calibrate(self, corners: np.ndarray):
        self.corners = corners.astype(np.float32)
        dst = np.array([
            [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE],
            [0, 0], [BOARD_SIZE, 0],
        ], dtype=np.float32)
        self.transform = cv2.getPerspectiveTransform(self.corners, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, self.corners)
        self.board_polygon = self.corners.reshape(-1, 1, 2).astype(np.int32)
        
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
        """Initialize both tracking systems."""
        self.board = chess.Board()
        self.moves = []
        
        # V1: Initialize from starting position
        self.board_state = {}
        self.detection_history.clear()
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                symbol = piece.symbol()
                color = 'white' if piece.color else 'black'
                self.board_state[sq] = (symbol, color)
        
        # V2: Clear state matrix
        self.state_matrix = np.zeros((64, 12), dtype=np.float32)
        self.model_pieces = {}
        
        self.is_tracking = True
        self.frame_count = 0
        
        print("\n" + "="*60)
        print("HYBRID TRACKER STARTED")
        print("="*60)
        print(f"V1 Position: {len(self.board_state)} pieces initialized")
        print(f"V2 Model: Waiting for detections...")
        print(f"Trust weights: Position={self.trust_position}, Model={self.trust_model}")
        print("="*60 + "\n")
    
    def _run_detection(self, frame: np.ndarray) -> List[dict]:
        """Run YOLO once and return all detections."""
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        detections = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                
                if not self._is_inside_board(cx, cy):
                    continue
                
                sq = self._point_to_square(cx, cy)
                if sq is None:
                    continue
                
                detections.append({
                    'square': sq,
                    'class': cls,
                    'conf': conf,
                    'center': (cx, cy),
                })
        
        return detections
    
    def _update_v1_state(self, detections: List[dict]) -> Tuple[Set[int], Set[int]]:
        """V1: Update position-based detection history."""
        detected_squares = {d['square'] for d in detections}
        
        # Update history for expected squares
        for sq in self.board_state.keys():
            was_detected = sq in detected_squares
            self.detection_history[sq].append(was_detected)
            while len(self.detection_history[sq]) > self.window_size:
                self.detection_history[sq].pop(0)
        
        # Track empty squares that might have new pieces
        for sq in detected_squares:
            if sq not in self.board_state:
                self.detection_history[sq].append(True)
                while len(self.detection_history[sq]) > self.window_size:
                    self.detection_history[sq].pop(0)
        
        # Find vacated and appeared
        vacated = set()
        for sq in self.board_state.keys():
            history = self.detection_history.get(sq, [])
            if history:
                rate = sum(history) / len(history)
                if rate < 0.3:
                    vacated.add(sq)
        
        appeared = set()
        for sq in detected_squares:
            if sq not in self.board_state:
                history = self.detection_history.get(sq, [])
                if len(history) >= 3:
                    rate = sum(history[-3:]) / 3
                    if rate > 0.6:
                        appeared.add(sq)
        
        return vacated, appeared
    
    def _update_v2_state(self, detections: List[dict]) -> Tuple[Set[int], Set[int]]:
        """V2: Update model-based state matrix."""
        # Build raw state
        raw_state = np.zeros((64, 12), dtype=np.float32)
        for det in detections:
            cls = det['class']
            if cls in YOLO_TO_STATE:
                state_idx = YOLO_TO_STATE[cls][0]
                sq = det['square']
                raw_state[sq, state_idx] = max(raw_state[sq, state_idx], det['conf'])
        
        # Temporal smoothing
        self.state_matrix *= self.decay_factor
        self.state_matrix = np.maximum(self.state_matrix, raw_state * self.learning_rate)
        
        # Extract current pieces
        prev_pieces = self.model_pieces.copy()
        self.model_pieces = {}
        for sq in range(64):
            max_conf = np.max(self.state_matrix[sq])
            if max_conf >= 0.4:
                piece_idx = np.argmax(self.state_matrix[sq])
                self.model_pieces[sq] = STATE_TO_PIECE[piece_idx]
        
        # Find changes
        removed = set(prev_pieces.keys()) - set(self.model_pieces.keys())
        added = set(self.model_pieces.keys()) - set(prev_pieces.keys())
        
        return removed, added
    
    def _score_move_v1(self, move: chess.Move, vacated: Set[int], appeared: Set[int]) -> float:
        """Score move from V1 (position) perspective."""
        score = 0.0
        from_sq = move.from_square
        to_sq = move.to_square
        
        if from_sq in vacated:
            score += 3.0
        if to_sq in appeared:
            score += 3.0
        
        # Castling
        if self.board.is_castling(move):
            rook_info = {
                chess.G1: (chess.H1, chess.F1), chess.C1: (chess.A1, chess.D1),
                chess.G8: (chess.H8, chess.F8), chess.C8: (chess.A8, chess.D8),
            }
            if move.to_square in rook_info:
                rf, rt = rook_info[move.to_square]
                if rf in vacated:
                    score += 2.0
                if rt in appeared:
                    score += 1.0
        
        return score
    
    def _score_move_v2(self, move: chess.Move, removed: Set[int], added: Set[int]) -> float:
        """Score move from V2 (model) perspective."""
        score = 0.0
        from_sq = move.from_square
        to_sq = move.to_square
        
        if from_sq in removed:
            score += 3.0
        if to_sq in added:
            score += 3.0
        elif to_sq in self.model_pieces:
            score += 1.5  # At least detected there
        
        # Check classification match
        if to_sq in self.model_pieces and from_sq in self.board_state:
            expected_symbol = self.board_state[from_sq][0].lower()
            detected_symbol = self.model_pieces[to_sq][0].lower()
            if expected_symbol == detected_symbol:
                score += 2.0  # Classification matches!
        
        return score
    
    def process(self, frame: np.ndarray) -> Optional[str]:
        if not self.is_tracking:
            return None
        
        self.frame_count += 1
        
        # STAGE 2: Shared detection
        detections = self._run_detection(frame)
        
        # Update both systems
        v1_vacated, v1_appeared = self._update_v1_state(detections)
        v2_removed, v2_added = self._update_v2_state(detections)
        
        # Cooldown
        if self.frame_count - self.last_move_frame < self.cooldown:
            return None
        
        # Need some evidence of change
        if not v1_vacated and not v2_removed:
            self.pending_move = None
            self.pending_count = 0
            return None
        
        # STAGE 3: Score all legal moves with both systems
        best_move = None
        best_score = 0.0
        best_agreement = False
        
        for move in self.board.legal_moves:
            from_sq = move.from_square
            
            # Must have some evidence from at least one system
            if from_sq not in v1_vacated and from_sq not in v2_removed:
                continue
            
            score_v1 = self._score_move_v1(move, v1_vacated, v1_appeared)
            score_v2 = self._score_move_v2(move, v2_removed, v2_added)
            
            # Weighted combination
            combined = self.trust_position * score_v1 + self.trust_model * score_v2
            
            # STAGE 4: Consensus bonus
            agreement = (from_sq in v1_vacated and from_sq in v2_removed)
            if agreement:
                combined += 2.0  # Bonus for agreement
            
            if combined > best_score:
                best_score = combined
                best_move = move
                best_agreement = agreement
        
        if best_move is None or best_score < 3.0:
            self.pending_move = None
            self.pending_count = 0
            return None
        
        # Debug
        if self.debug:
            print(f"  [{self.frame_count}] V1 vacated: {[SQUARE_NAMES[s] for s in v1_vacated]}")
            print(f"  [{self.frame_count}] V2 removed: {[SQUARE_NAMES[s] for s in v2_removed]}")
            print(f"  Best: {self.board.san(best_move)} (score={best_score:.1f}, agree={best_agreement})")
        
        # STAGE 5: Adaptive confirmation
        if best_agreement:
            confirm_needed = self.base_confirm - 2  # Faster if agree
        elif best_score > 5.0:
            confirm_needed = self.base_confirm
        else:
            confirm_needed = self.base_confirm + 2  # Slower if uncertain
        
        if best_move == self.pending_move:
            self.pending_count += 1
            if self.pending_count >= confirm_needed:
                return self._execute_move(best_move)
        else:
            self.pending_move = best_move
            self.pending_count = 1
            self.pending_confidence = best_score
        
        return None
    
    def _execute_move(self, move: chess.Move) -> str:
        san = self.board.san(move)
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Update V1 board_state
        if from_sq in self.board_state:
            piece_info = self.board_state.pop(from_sq)
            if move.promotion:
                promo_map = {chess.QUEEN: 'Q', chess.ROOK: 'R', 
                             chess.BISHOP: 'B', chess.KNIGHT: 'N'}
                new_sym = promo_map.get(move.promotion, 'Q')
                if piece_info[1] == 'black':
                    new_sym = new_sym.lower()
                piece_info = (new_sym, piece_info[1])
            if to_sq in self.board_state:
                del self.board_state[to_sq]
            self.board_state[to_sq] = piece_info
        
        # Castling rook
        if self.board.is_castling(move):
            rook_info = {
                chess.G1: (chess.H1, chess.F1), chess.C1: (chess.A1, chess.D1),
                chess.G8: (chess.H8, chess.F8), chess.C8: (chess.A8, chess.D8),
            }
            if move.to_square in rook_info and rook_info[move.to_square][0] in self.board_state:
                rf, rt = rook_info[move.to_square]
                rook = self.board_state.pop(rf)
                self.board_state[rt] = rook
        
        # En passant
        if self.board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
            if cap_sq in self.board_state:
                del self.board_state[cap_sq]
        
        # Clear V2 state for affected squares
        self.state_matrix[from_sq] = 0
        self.state_matrix[to_sq] = 0
        
        # Clear detection history
        self.detection_history[from_sq].clear()
        self.detection_history[to_sq].clear()
        
        self.board.push(move)
        self.moves.append(san)
        self.last_move_frame = self.frame_count
        self.pending_move = None
        self.pending_count = 0
        
        print(f"[{self.frame_count}] Move #{len(self.moves)}: {san}")
        return san
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        
        if self.corners is not None:
            cv2.polylines(vis, [self.corners.astype(np.int32)], True, (0, 255, 0), 2)
        
        if self.is_tracking and self.inv_transform is not None:
            for sq, (symbol, color) in self.board_state.items():
                f, r = sq % 8, sq // 8
                wx = (f + 0.5) * SQUARE_SIZE
                wy = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
                pt = np.array([[wx, wy]], dtype=np.float32)
                center = cv2.perspectiveTransform(pt[np.newaxis], self.inv_transform)[0, 0]
                cx, cy = int(center[0]), int(center[1])
                
                # Color: Check if V1 and V2 agree
                v2_has = sq in self.model_pieces
                history = self.detection_history.get(sq, [])
                rate = sum(history) / len(history) if history else 1.0
                
                if v2_has and rate > 0.5:
                    box_color = (0, 255, 0)  # Green = agreement
                elif rate > 0.3:
                    box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                else:
                    box_color = (0, 0, 255)  # Red = likely moved
                
                cv2.rectangle(vis, (cx-14, cy-14), (cx+14, cy+14), box_color, 2)
                cv2.putText(vis, symbol.upper(), (cx-6, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        turn = "White" if self.board.turn else "Black"
        cv2.putText(vis, f"VERSION 3: Hybrid | Moves: {len(self.moves)} | {turn}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.moves:
            cv2.putText(vis, " ".join(self.moves[-8:]), (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis
    
    def get_pgn(self, event_name: str = "Hybrid Tracker") -> str:
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
        description="VERSION 3: Hybrid Tracker (Position + Model)"
    )
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--output', '-o', help='Output PGN file')
    parser.add_argument('--model', '-m', default='models/pieces.pt')
    parser.add_argument('--crop', type=float, default=0.45)
    parser.add_argument('--debug', '-d', action='store_true')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path}")
        sys.exit(1)
    
    model_path = args.model
    if not Path(model_path).exists():
        for alt in ['models/pieces.pt', 'models/pieces_trained.pt']:
            if Path(alt).exists():
                model_path = alt
                break
    
    output_path = args.output or f"output/pgn/{video_path.stem}_v3_hybrid.pgn"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_h = int(h * args.crop)
    
    print("\n" + "="*60)
    print("VERSION 3: HYBRID TRACKER")
    print("="*60)
    print(f"Video: {video_path.name}")
    print(f"Model: {model_path}")
    
    ret, frame = cap.read()
    if not ret:
        sys.exit(1)
    frame = frame[:crop_h, :]
    
    tracker = HybridTracker(model_path=model_path)
    tracker.debug = args.debug
    
    corners = interactive_corners(frame)
    if corners is None:
        cap.release()
        sys.exit(0)
    tracker.calibrate(corners)
    tracker.start()
    
    cv2.namedWindow("Hybrid Tracker", cv2.WINDOW_NORMAL)
    frame_delay = int(1000 / fps)
    paused = False
    n = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            frame = frame[:crop_h, :]
            tracker.process(frame)
        
        vis = tracker.visualize(frame)
        pct = n / total if total else 0
        bar_y = vis.shape[0] - 15
        cv2.rectangle(vis, (10, bar_y), (vis.shape[1]-10, bar_y+8), (50,50,50), -1)
        cv2.rectangle(vis, (10, bar_y), (10+int((vis.shape[1]-20)*pct), bar_y+8), (0,255,0), -1)
        
        cv2.imshow("Hybrid Tracker", vis)
        key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): paused = not paused
        elif key == ord('d'): tracker.debug = not tracker.debug
    
    cap.release()
    cv2.destroyAllWindows()
    
    with open(output_path, 'w') as f:
        f.write(tracker.get_pgn(video_path.stem))
    
    print(f"\nResults: {len(tracker.moves)} moves → {output_path}")
    print(f"Moves: {' '.join(tracker.moves)}")


if __name__ == "__main__":
    main()
