#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    VERSION 1: POSITION-BASED TRACKER                         ║
║                    "Perfect 32 Boxes" Approach                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This tracker TRUSTS the starting position completely and uses               ║
║  YOLO only to detect PRESENCE of pieces (not classification).               ║
║  Moves are inferred by detecting when pieces LEAVE and ARRIVE.              ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
                              PIPELINE DOCUMENTATION
================================================================================

OVERVIEW:
---------
This approach is based on the fundamental principle that we KNOW exactly where
all 32 pieces are at the start of a chess game. The standard starting position
is deterministic - we don't need to "detect" it.

The key insight: Rather than asking "what piece is here?", we ask "is the 
piece that SHOULD be here still present?"

PIPELINE STAGES:
----------------

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INITIALIZATION                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1.1. Load video and display first frame                                   │
│   1.2. User clicks 4 corners: h1 → a1 → a8 → h8                            │
│   1.3. Compute perspective transform matrix                                 │
│   1.4. Initialize board_state with standard chess starting position:       │
│                                                                             │
│        Rank 8: r n b q k b n r  (Black pieces)                             │
│        Rank 7: p p p p p p p p  (Black pawns)                              │
│        Rank 6: . . . . . . . .                                             │
│        Rank 5: . . . . . . . .                                             │
│        Rank 4: . . . . . . . .                                             │
│        Rank 3: . . . . . . . .                                             │
│        Rank 2: P P P P P P P P  (White pawns)                              │
│        Rank 1: R N B Q K B N R  (White pieces)                             │
│                                                                             │
│   1.5. For each of the 32 occupied squares, mark as "piece present"        │
│   1.6. Initialize python-chess board for legal move validation             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: PER-FRAME DETECTION (runs every frame)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   2.1. Crop frame to board region (removes irrelevant background)          │
│   2.2. Run YOLO model to detect ALL pieces (ignoring class labels)         │
│   2.3. For each detection:                                                  │
│        - Calculate center point (cx, cy)                                   │
│        - Check if inside board polygon                                     │
│        - Transform to board coordinates using perspective matrix           │
│        - Map to nearest square (0-63)                                      │
│   2.4. Result: Set of squares where pieces are currently detected          │
│                                                                             │
│   Note: We DON'T care WHAT piece is detected, only THAT something is there │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: OCCUPANCY TRACKING (sliding window)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For each square that SHOULD have a piece (according to board_state):     │
│                                                                             │
│   3.1. Maintain detection history: last N frames (window_size = 10)        │
│        Example: [True, True, True, False, False, False, False, ...]        │
│                                                                             │
│   3.2. Calculate detection_rate = count(detected) / window_size            │
│        - rate > 0.6: Piece is PRESENT (normal)                             │
│        - rate 0.3-0.6: Piece is UNCERTAIN (might be occluded)              │
│        - rate < 0.3: Piece has likely VACATED this square                  │
│                                                                             │
│   3.3. For squares NOT expected to have pieces:                            │
│        Track if detection becomes consistent (piece has ARRIVED)           │
│                                                                             │
│   Why sliding window? Single-frame detection is noisy - hands block view,  │
│   lighting changes, model confidence varies. We need temporal smoothing.   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: MOVE DETECTION                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   4.1. Identify VACATED squares:                                           │
│        Squares that SHOULD have piece but detection_rate < 0.3             │
│                                                                             │
│   4.2. Identify APPEARED squares:                                          │
│        Squares that SHOULDN'T have piece but now consistently detected     │
│                                                                             │
│   4.3. For each legal move in current position:                            │
│        Score = 0                                                           │
│        IF from_square in VACATED: Score += 3                               │
│        IF to_square in APPEARED: Score += 3                                │
│        IF to_square in detected: Score += 2                                │
│        (Special scoring for castling: both king and rook must move)        │
│        (Special scoring for en passant: captured pawn square empty)        │
│                                                                             │
│   4.4. Best move = move with highest score (minimum threshold: 4)          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: MOVE CONFIRMATION                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   5.1. Track "pending" move across frames                                  │
│   5.2. Same move must be best candidate for N consecutive frames           │
│        (confirm_threshold = 5 frames)                                      │
│   5.3. If confirmed:                                                        │
│        - Execute move on python-chess board                                │
│        - Update board_state (move piece from source to destination)        │
│        - Handle captures (remove captured piece from state)                │
│        - Handle castling (move rook too)                                   │
│        - Handle en passant (remove captured pawn)                          │
│        - Handle promotion (change pawn to promoted piece)                  │
│        - Clear detection history for affected squares                      │
│        - Start cooldown timer                                              │
│                                                                             │
│   5.4. Cooldown: No new moves detected for N frames (cooldown = 20)        │
│        Prevents detecting same move multiple times during piece motion     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

ADVANTAGES:
-----------
✓ No reliance on piece classification (works even with poor model)
✓ Robust to model errors - we only need presence detection
✓ Handles any board/piece style since we don't need accurate classification
✓ Very fast - classification output is ignored

DISADVANTAGES:
--------------
✗ Assumes game starts from standard position (no mid-game join)
✗ Can't recover if a move is missed (state becomes out of sync)
✗ Sensitive to piece handoff during long moves
✗ Cannot detect if wrong piece type moves (e.g., if queen moved but we thought bishop)

DATA STRUCTURES:
----------------
board_state: Dict[int, Tuple[str, str]]
    Maps square index (0-63) to (piece_symbol, color)
    Example: {0: ('R', 'white'), 1: ('N', 'white'), ...}

detection_history: Dict[int, List[Tuple[bool, int]]]
    Maps square index to list of (was_detected, frame_number)
    Used to calculate detection rate over sliding window

pending_move: Optional[chess.Move]
    The move we're trying to confirm

pending_count: int
    How many consecutive frames the pending move has been best

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


def get_square_centers() -> np.ndarray:
    """Pre-compute center coordinates for all 64 squares in board space."""
    centers = []
    for sq in range(64):
        f, r = sq % 8, sq // 8  # file and rank
        x = (f + 0.5) * SQUARE_SIZE
        y = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
        centers.append([x, y])
    return np.array(centers)


def interactive_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """Let user click 4 corners: h1 -> a1 -> a8 -> h8."""
    corners = []
    display = frame.copy()
    names = ["h1 (bottom-right)", "a1 (bottom-left)", "a8 (top-left)", "h8 (top-right)"]
    
    def mouse_cb(event, x, y, flags, param):
        nonlocal corners, display
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([x, y])
            color = (0, 255, 0)
            cv2.circle(display, (x, y), 10, color, -1)
            cv2.putText(display, names[len(corners)-1].split()[0], (x+15, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if len(corners) > 1:
                cv2.line(display, tuple(corners[-2]), tuple(corners[-1]), color, 2)
            if len(corners) == 4:
                cv2.line(display, tuple(corners[3]), tuple(corners[0]), color, 2)
                
    cv2.namedWindow("Click 4 Corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click 4 Corners", mouse_cb)
    
    print("\n" + "="*60)
    print("CORNER SELECTION")
    print("="*60)
    print("Click the 4 corners of the chessboard in this order:")
    print("  1. h1 (bottom-right, white's king-side)")
    print("  2. a1 (bottom-left, white's queen-side)")
    print("  3. a8 (top-left, black's queen-side)")
    print("  4. h8 (top-right, black's king-side)")
    print("\nControls: 'c' = confirm, 'r' = reset, 'q' = quit")
    print("="*60)
    
    while True:
        temp = display.copy()
        if len(corners) < 4:
            cv2.putText(temp, f"Click: {names[len(corners)]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            cv2.putText(temp, "Press 'c' to confirm corners", (10, 30),
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


class PositionBasedTracker:
    """
    VERSION 1: Position-Based Tracker
    
    Philosophy: Trust the starting position, detect piece PRESENCE only.
    We know where all 32 pieces are - just track when they leave/arrive.
    """
    
    def __init__(self, model_path: str = 'models/pieces.pt'):
        """Initialize the tracker with YOLO model."""
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
        # BOARD STATE (the "32 boxes")
        # Maps square_index -> (piece_symbol, color)
        # ─────────────────────────────────────────────────────────────────────
        self.board_state: Dict[int, Tuple[str, str]] = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # DETECTION HISTORY (sliding window)
        # Maps square_index -> List of (was_detected, frame_num)
        # ─────────────────────────────────────────────────────────────────────
        self.detection_history: Dict[int, List[Tuple[bool, int]]] = defaultdict(list)
        self.window_size = 10  # Frames to consider
        
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
        self.cooldown = 20  # Frames between moves
        
        # ─────────────────────────────────────────────────────────────────────
        # MOVE CONFIRMATION
        # ─────────────────────────────────────────────────────────────────────
        self.pending_move: Optional[chess.Move] = None
        self.pending_count = 0
        self.confirm_threshold = 5  # Frames to confirm a move
        
        # ─────────────────────────────────────────────────────────────────────
        # DETECTION SETTINGS
        # ─────────────────────────────────────────────────────────────────────
        self.conf_threshold = 0.15  # Low threshold - we just need presence
        self.debug = False
        
    def calibrate(self, corners: np.ndarray):
        """
        STAGE 1.2-1.3: Compute perspective transform from clicked corners.
        
        The corners define the mapping from video pixels to board coordinates.
        Board coordinate system: (0,0) at a1, (BOARD_SIZE, BOARD_SIZE) at h8.
        """
        self.corners = corners.astype(np.float32)
        
        # Destination points in board space
        dst = np.array([
            [BOARD_SIZE, BOARD_SIZE],  # h1 -> bottom-right
            [0, BOARD_SIZE],           # a1 -> bottom-left
            [0, 0],                    # a8 -> top-left
            [BOARD_SIZE, 0],           # h8 -> top-right
        ], dtype=np.float32)
        
        self.transform = cv2.getPerspectiveTransform(self.corners, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, self.corners)
        self.board_polygon = self.corners.reshape(-1, 1, 2).astype(np.int32)
        
        print("[CALIBRATE] Perspective transform computed")
        
    def _is_inside_board(self, x: float, y: float) -> bool:
        """Check if a point is inside the board polygon."""
        if self.board_polygon is None:
            return True
        return cv2.pointPolygonTest(self.board_polygon, (x, y), False) >= 0
    
    def _point_to_square(self, x: float, y: float) -> Optional[int]:
        """
        Transform a pixel coordinate to a square index (0-63).
        Returns None if point is outside the board.
        """
        if not self._is_inside_board(x, y):
            return None
            
        pt = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.transform)[0, 0]
        
        if 0 <= warped[0] < BOARD_SIZE and 0 <= warped[1] < BOARD_SIZE:
            # Find nearest square center
            dists = np.linalg.norm(self.square_centers - warped, axis=1)
            return int(np.argmin(dists))
        return None
    
    def start(self):
        """
        STAGE 1.4-1.6: Initialize from standard starting position.
        
        We TRUST that the game starts from the standard position.
        All 32 pieces are placed on their starting squares.
        """
        self.board = chess.Board()
        self.moves = []
        self.board_state = {}
        self.detection_history.clear()
        
        # Copy state from chess.Board (which has standard starting position)
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                symbol = piece.symbol()
                color = 'white' if piece.color else 'black'
                self.board_state[sq] = (symbol, color)
        
        self.is_tracking = True
        self.frame_count = 0
        self.last_move_frame = 0
        self.pending_move = None
        self.pending_count = 0
        
        print("\n" + "="*60)
        print("TRACKING STARTED")
        print("="*60)
        print(f"Initialized with {len(self.board_state)} pieces in starting position")
        print("="*60 + "\n")
    
    def _detect_occupied_squares(self, frame: np.ndarray) -> Set[int]:
        """
        STAGE 2: Run YOLO and get set of squares with detected pieces.
        
        NOTE: We ignore the class labels! We only care about PRESENCE.
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        occupied: Set[int] = set()
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                
                # Calculate center point
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                
                # Must be inside board
                if not self._is_inside_board(cx, cy):
                    continue
                
                # Map to square
                sq = self._point_to_square(cx, cy)
                if sq is not None:
                    occupied.add(sq)
        
        return occupied
    
    def _update_detection_history(self, detected_squares: Set[int]):
        """
        STAGE 3: Update sliding window detection history.
        
        For each square that SHOULD have a piece, track whether it was
        detected in this frame.
        """
        for sq in self.board_state.keys():
            was_detected = sq in detected_squares
            self.detection_history[sq].append((was_detected, self.frame_count))
            
            # Keep only recent history (sliding window)
            while len(self.detection_history[sq]) > self.window_size:
                self.detection_history[sq].pop(0)
        
        # Also track empty squares that might now have pieces
        for sq in detected_squares:
            if sq not in self.board_state:
                self.detection_history[sq].append((True, self.frame_count))
                while len(self.detection_history[sq]) > self.window_size:
                    self.detection_history[sq].pop(0)
    
    def _get_detection_rate(self, sq: int) -> float:
        """Get the detection rate for a square over the sliding window."""
        history = self.detection_history.get(sq, [])
        if not history:
            return 1.0  # Assume present if no history yet
        detected_count = sum(1 for d, _ in history if d)
        return detected_count / len(history)
    
    def _find_changes(self, detected_squares: Set[int]) -> Tuple[Set[int], Set[int]]:
        """
        STAGE 4.1-4.2: Find vacated and appeared squares.
        
        VACATED: Square should have piece but detection rate is low
        APPEARED: Square shouldn't have piece but now consistently detected
        """
        expected_occupied = set(self.board_state.keys())
        
        # Vacated: expected but not detected recently
        vacated = set()
        for sq in expected_occupied:
            rate = self._get_detection_rate(sq)
            if rate < 0.3:  # Less than 30% detection = likely moved away
                vacated.add(sq)
        
        # Appeared: not expected but consistently detected
        appeared = set()
        for sq in detected_squares:
            if sq not in expected_occupied:
                history = self.detection_history.get(sq, [])
                if len(history) >= 3:
                    recent_rate = sum(1 for d, _ in history[-3:] if d) / 3
                    if recent_rate > 0.6:  # Detected in >60% of last 3 frames
                        appeared.add(sq)
        
        return vacated, appeared
    
    def _score_legal_moves(self, vacated: Set[int], appeared: Set[int], 
                           detected: Set[int]) -> List[Tuple[chess.Move, int]]:
        """
        STAGE 4.3: Score each legal move based on evidence.
        
        Returns list of (move, score) sorted by score descending.
        """
        scored = []
        
        for move in self.board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            score = 0
            
            # FROM square should be vacated (piece left)
            if from_sq in vacated:
                score += 3
            else:
                continue  # Skip if source isn't vacated
            
            # TO square should have piece appeared or be detected
            if to_sq in appeared:
                score += 3
            elif to_sq in detected:
                score += 2
            
            # CASTLING: Both king and rook must move
            if self.board.is_castling(move):
                rook_moves = {
                    chess.G1: (chess.H1, chess.F1),  # White kingside
                    chess.C1: (chess.A1, chess.D1),  # White queenside
                    chess.G8: (chess.H8, chess.F8),  # Black kingside
                    chess.C8: (chess.A8, chess.D8),  # Black queenside
                }
                rook_info = rook_moves.get(move.to_square)
                if rook_info:
                    rook_from, rook_to = rook_info
                    if rook_from in vacated:
                        score += 2
                    if rook_to in detected:
                        score += 1
            
            # EN PASSANT: Captured pawn square should be empty
            if self.board.is_en_passant(move):
                captured_sq = chess.square(chess.square_file(to_sq), 
                                          chess.square_rank(from_sq))
                if captured_sq in vacated:
                    score += 2
            
            if score > 0:
                scored.append((move, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def process(self, frame: np.ndarray) -> Optional[str]:
        """
        Main processing pipeline for a single frame.
        Returns SAN string if a move was detected, None otherwise.
        """
        if not self.is_tracking:
            return None
        
        self.frame_count += 1
        
        # STAGE 2: Detect occupied squares
        detected = self._detect_occupied_squares(frame)
        
        # STAGE 3: Update history
        self._update_detection_history(detected)
        
        # Check cooldown
        if self.frame_count - self.last_move_frame < self.cooldown:
            return None
        
        # STAGE 4.1-4.2: Find changes
        vacated, appeared = self._find_changes(detected)
        
        if not vacated:
            self.pending_move = None
            self.pending_count = 0
            return None
        
        # Debug output
        if self.debug and vacated:
            v_names = [SQUARE_NAMES[s] for s in vacated]
            a_names = [SQUARE_NAMES[s] for s in appeared]
            print(f"  [Frame {self.frame_count}] Vacated: {v_names}, Appeared: {a_names}")
        
        # STAGE 4.3: Score legal moves
        scored_moves = self._score_legal_moves(vacated, appeared, detected)
        
        if not scored_moves:
            self.pending_move = None
            self.pending_count = 0
            return None
        
        best_move, best_score = scored_moves[0]
        
        # STAGE 5: Confirmation
        if best_score >= 4:  # Minimum threshold
            if best_move == self.pending_move:
                self.pending_count += 1
                if self.pending_count >= self.confirm_threshold:
                    return self._execute_move(best_move)
            else:
                self.pending_move = best_move
                self.pending_count = 1
                if self.debug:
                    print(f"  Pending: {self.board.san(best_move)} (score={best_score})")
        else:
            self.pending_move = None
            self.pending_count = 0
        
        return None
    
    def _execute_move(self, move: chess.Move) -> str:
        """
        STAGE 5.3: Execute confirmed move and update state.
        """
        san = self.board.san(move)
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Update board_state
        if from_sq in self.board_state:
            piece_info = self.board_state.pop(from_sq)
            
            # Handle promotion
            if move.promotion:
                promo_map = {
                    chess.QUEEN: 'Q', chess.ROOK: 'R', 
                    chess.BISHOP: 'B', chess.KNIGHT: 'N'
                }
                new_symbol = promo_map.get(move.promotion, 'Q')
                if piece_info[1] == 'black':
                    new_symbol = new_symbol.lower()
                piece_info = (new_symbol, piece_info[1])
            
            # Handle capture
            if to_sq in self.board_state:
                del self.board_state[to_sq]
            
            self.board_state[to_sq] = piece_info
        
        # Handle castling rook
        if self.board.is_castling(move):
            rook_moves = {
                chess.G1: (chess.H1, chess.F1),
                chess.C1: (chess.A1, chess.D1),
                chess.G8: (chess.H8, chess.F8),
                chess.C8: (chess.A8, chess.D8),
            }
            rook_info = rook_moves.get(move.to_square)
            if rook_info and rook_info[0] in self.board_state:
                rook = self.board_state.pop(rook_info[0])
                self.board_state[rook_info[1]] = rook
        
        # Handle en passant
        if self.board.is_en_passant(move):
            captured_sq = chess.square(chess.square_file(to_sq), 
                                      chess.square_rank(from_sq))
            if captured_sq in self.board_state:
                del self.board_state[captured_sq]
        
        # Push move to chess engine
        self.board.push(move)
        self.moves.append(san)
        
        # Reset state
        self.last_move_frame = self.frame_count
        self.pending_move = None
        self.pending_count = 0
        
        # Clear history for affected squares
        self.detection_history[from_sq].clear()
        self.detection_history[to_sq].clear()
        
        print(f"[{self.frame_count}] Move #{len(self.moves)}: {san}")
        return san
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Create visualization overlay."""
        vis = frame.copy()
        
        # Draw board outline
        if self.corners is not None:
            cv2.polylines(vis, [self.corners.astype(np.int32)], True, (0, 255, 0), 2)
        
        # Draw piece boxes
        if self.is_tracking and self.inv_transform is not None:
            for sq, (symbol, color) in self.board_state.items():
                f, r = sq % 8, sq // 8
                wx = (f + 0.5) * SQUARE_SIZE
                wy = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
                pt = np.array([[wx, wy]], dtype=np.float32)
                center = cv2.perspectiveTransform(pt[np.newaxis], self.inv_transform)[0, 0]
                cx, cy = int(center[0]), int(center[1])
                
                # Color based on detection rate
                rate = self._get_detection_rate(sq)
                if rate > 0.6:
                    box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                elif rate > 0.3:
                    box_color = (0, 165, 255)  # Orange = uncertain
                else:
                    box_color = (0, 0, 255)  # Red = likely moved
                
                cv2.rectangle(vis, (cx-14, cy-14), (cx+14, cy+14), box_color, 2)
                cv2.putText(vis, symbol.upper(), (cx-6, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Status
        turn = "White" if self.board.turn else "Black"
        cv2.putText(vis, f"VERSION 1: Position-Based | Moves: {len(self.moves)} | {turn}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.moves:
            recent = self.moves[-8:]
            cv2.putText(vis, " ".join(recent), (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis
    
    def get_pgn(self, event_name: str = "Position Tracker") -> str:
        """Generate PGN string from recorded moves."""
        game = chess.pgn.Game()
        game.headers["Event"] = event_name
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "Player 1"
        game.headers["Black"] = "Player 2"
        
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
        elif board.is_stalemate() or board.is_insufficient_material():
            game.headers["Result"] = "1/2-1/2"
        else:
            game.headers["Result"] = "*"
        
        return str(game)


def main():
    parser = argparse.ArgumentParser(
        description="VERSION 1: Position-Based Tracker (32 Boxes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PIPELINE:
  1. Initialize with standard starting position (32 pieces)
  2. Use YOLO to detect piece PRESENCE (not classification)
  3. Track which squares become empty and which become occupied
  4. Match changes to legal moves and confirm over multiple frames
        """
    )
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--output', '-o', help='Output PGN file')
    parser.add_argument('--model', '-m', default='models/pieces.pt', help='YOLO model path')
    parser.add_argument('--crop', type=float, default=0.45, help='Crop ratio from top')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    model_path = args.model
    if not Path(model_path).exists():
        for alt in ['models/pieces.pt', 'models/pieces_trained.pt', '../models/pieces.pt']:
            if Path(alt).exists():
                model_path = alt
                break
    
    output_path = args.output or f"output/pgn/{video_path.stem}_v1_position.pgn"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_height = int(height * args.crop)
    
    print("\n" + "="*60)
    print("VERSION 1: POSITION-BASED TRACKER")
    print("="*60)
    print(f"Video: {video_path.name}")
    print(f"Frames: {total_frames} @ {fps:.1f} FPS")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print("="*60)
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        sys.exit(1)
    frame = frame[:crop_height, :]
    
    # Initialize tracker
    tracker = PositionBasedTracker(model_path=model_path)
    tracker.debug = args.debug
    
    # Calibrate
    corners = interactive_corners(frame)
    if corners is None:
        print("Cancelled")
        cap.release()
        sys.exit(0)
    tracker.calibrate(corners)
    
    # Start tracking
    tracker.start()
    
    # Main loop
    cv2.namedWindow("Position Tracker", cv2.WINDOW_NORMAL)
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
            tracker.process(frame)
        
        vis = tracker.visualize(frame)
        
        # Progress bar
        pct = frame_num / total_frames if total_frames else 0
        bar_y = vis.shape[0] - 15
        bar_w = vis.shape[1] - 20
        cv2.rectangle(vis, (10, bar_y), (10 + bar_w, bar_y + 8), (50, 50, 50), -1)
        cv2.rectangle(vis, (10, bar_y), (10 + int(bar_w * pct), bar_y + 8), (0, 255, 0), -1)
        
        cv2.imshow("Position Tracker", vis)
        
        key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord('d'):
            tracker.debug = not tracker.debug
            print(f"Debug: {'ON' if tracker.debug else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save PGN
    pgn_content = tracker.get_pgn(f"Game: {video_path.stem}")
    with open(output_path, 'w') as f:
        f.write(pgn_content)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total moves detected: {len(tracker.moves)}")
    print(f"PGN saved to: {output_path}")
    print(f"\nMoves: {' '.join(tracker.moves)}")
    print("="*60)


if __name__ == "__main__":
    main()
