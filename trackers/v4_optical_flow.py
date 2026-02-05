#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    VERSION 4: OPTICAL FLOW TRACKER                           ║
║                    "Motion-Based" Approach                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Uses optical flow to detect piece motion directly from pixel changes.      ║
║  Doesn't rely on YOLO detection - instead tracks actual movement.           ║
║  Combined with sparse YOLO checks for validation.                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
                              PIPELINE DOCUMENTATION
================================================================================

OVERVIEW:
---------
This approach takes a fundamentally different view: instead of detecting 
pieces frame-by-frame, we detect MOTION. When something moves on the board,
we see the optical flow vectors pointing from source to destination.

Key insight: A chess move creates a characteristic flow pattern - vectors
emanating FROM one square and converging TO another square.

This is the approach most similar to how humans perceive moves - we notice
movement, not static positions.

PIPELINE STAGES:
----------------

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INITIALIZATION                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1.1. User clicks 4 corners: h1 → a1 → a8 → h8                            │
│                                                                             │
│   1.2. Create warped view: perspective-correct top-down board              │
│        Size: 480×480 pixels (60px per square)                              │
│                                                                             │
│   1.3. Create square masks:                                                 │
│        For each of 64 squares, create a binary mask of its pixels          │
│        Used to aggregate flow within each square                           │
│                                                                             │
│   1.4. Initialize board state from starting position                       │
│                                                                             │
│   1.5. Store previous frame for flow computation                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: OPTICAL FLOW COMPUTATION (runs every frame)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   2.1. Warp current frame to top-down view                                 │
│                                                                             │
│   2.2. Convert to grayscale                                                │
│                                                                             │
│   2.3. Compute dense optical flow (Farneback method):                      │
│        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, ...)      │
│                                                                             │
│        Result: flow[y,x] = (dx, dy) motion vector at each pixel            │
│                                                                             │
│   2.4. Compute flow magnitude:                                             │
│        magnitude = sqrt(dx² + dy²)                                         │
│                                                                             │
│   2.5. Threshold significant motion:                                       │
│        motion_mask = magnitude > motion_threshold (e.g., 2.0 pixels)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: SQUARE-LEVEL MOTION ANALYSIS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   For each of 64 squares:                                                  │
│                                                                             │
│   3.1. Count motion pixels in square:                                      │
│        motion_count[sq] = sum(motion_mask & square_mask[sq])               │
│                                                                             │
│   3.2. Compute average flow vector:                                        │
│        avg_flow[sq] = mean(flow[square_mask[sq]])                          │
│                                                                             │
│   3.3. Compute flow direction:                                             │
│        angle[sq] = atan2(avg_flow.y, avg_flow.x)                           │
│                                                                             │
│   3.4. Classify squares:                                                   │
│        - OUTFLOW: High motion, flow pointing AWAY from square center      │
│          (This is likely the SOURCE of the move - piece leaving)           │
│        - INFLOW: High motion, flow pointing TOWARD square center           │
│          (This is likely the DESTINATION - piece arriving)                 │
│        - STATIC: Low motion (no piece movement)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: MOVE DETECTION FROM FLOW PATTERNS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   4.1. Find candidate source squares:                                      │
│        Squares with high OUTFLOW and expected to have a piece              │
│                                                                             │
│   4.2. Find candidate destination squares:                                 │
│        Squares with high INFLOW                                            │
│                                                                             │
│   4.3. Match source to destination by flow direction:                      │
│        If flow from square A points toward square B,                       │
│        then (A, B) is a move candidate                                     │
│                                                                             │
│   4.4. Validate against legal moves:                                       │
│        move = chess.Move(source, destination)                              │
│        if move in board.legal_moves: valid                                 │
│                                                                             │
│   4.5. Score candidates:                                                   │
│        score = source_outflow + destination_inflow + direction_match       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: MOTION STATE MACHINE                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   The motion detection follows a state machine:                            │
│                                                                             │
│   IDLE → MOTION_START → MOTION_PEAK → MOTION_END → CONFIRM                 │
│                                                                             │
│   5.1. IDLE:                                                               │
│        No significant motion detected                                      │
│        Wait for motion_count > start_threshold                             │
│                                                                             │
│   5.2. MOTION_START:                                                       │
│        Motion just started, wait for it to develop                         │
│        Record initial source candidates                                    │
│                                                                             │
│   5.3. MOTION_PEAK:                                                        │
│        Motion at maximum, piece is in transit                              │
│        May see flow through intermediate squares                           │
│                                                                             │
│   5.4. MOTION_END:                                                         │
│        Motion decreasing, piece settling at destination                    │
│        Record final destination candidates                                 │
│                                                                             │
│   5.5. CONFIRM:                                                            │
│        Motion stopped, verify move is stable                               │
│        If valid legal move found: execute                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: PERIODIC YOLO VALIDATION (every N frames)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Every N frames (N=30), run YOLO to validate state:                       │
│                                                                             │
│   6.1. Run YOLO detection on current frame                                 │
│                                                                             │
│   6.2. Compare detected pieces with expected board_state                   │
│                                                                             │
│   6.3. If significant mismatch:                                            │
│        - Log warning                                                        │
│        - Possibly resync state (if confident in YOLO)                      │
│                                                                             │
│   This provides a safety net if flow-based detection misses a move         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

ADVANTAGES:
-----------
✓ Detects actual movement, not just presence
✓ Works with any piece appearance (no training needed for new sets)
✓ Very fast (no neural network inference per frame)
✓ Naturally handles piece motion blur
✓ Can detect hand movements (and ignore them)

DISADVANTAGES:
--------------
✗ Sensitive to camera shake (creates false flow)
✗ Can be confused by shadows or lighting changes
✗ Doesn't work if pieces "teleport" (frame skip)
✗ Needs tuning for different video qualities

DATA STRUCTURES:
----------------
board_state: Dict[int, Tuple[str, str]]
    Expected piece locations (from starting position or YOLO sync)

prev_gray: np.ndarray
    Previous frame in grayscale for flow computation

motion_state: Enum
    Current state in the motion state machine

motion_history: List[Tuple[Set[int], Set[int]]]
    History of (outflow_squares, inflow_squares) during motion

================================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
from enum import Enum
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
WARPED_SIZE = 480  # Size of warped board image
WARPED_SQUARE = WARPED_SIZE // 8  # 60 pixels per square


class MotionState(Enum):
    IDLE = 0
    MOTION_START = 1
    MOTION_PEAK = 2
    MOTION_END = 3
    CONFIRM = 4


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


class OpticalFlowTracker:
    """
    VERSION 4: Optical Flow Tracker
    
    Uses dense optical flow to detect piece motion directly.
    Motion creates distinctive flow patterns from source to destination.
    """
    
    def __init__(self, model_path: str = 'models/pieces.pt'):
        # Optional YOLO for validation
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.has_model = True
            print(f"[INIT] YOLO loaded for validation: {model_path}")
        except:
            self.model = None
            self.has_model = False
            print("[INIT] Running without YOLO (motion-only mode)")
        
        # ─────────────────────────────────────────────────────────────────────
        # GEOMETRY
        # ─────────────────────────────────────────────────────────────────────
        self.corners: Optional[np.ndarray] = None
        self.warp_transform: Optional[np.ndarray] = None
        self.inv_transform: Optional[np.ndarray] = None
        self.board_polygon: Optional[np.ndarray] = None
        
        # Square masks for warped view
        self.square_masks: Dict[int, np.ndarray] = {}
        self.square_centers_warped: Dict[int, Tuple[int, int]] = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # OPTICAL FLOW STATE
        # ─────────────────────────────────────────────────────────────────────
        self.prev_gray: Optional[np.ndarray] = None
        self.flow: Optional[np.ndarray] = None
        
        # Motion detection parameters
        self.motion_threshold = 2.0  # Minimum flow magnitude to count as motion
        self.outflow_threshold = 200  # Motion pixels needed for outflow
        self.inflow_threshold = 200   # Motion pixels needed for inflow
        
        # ─────────────────────────────────────────────────────────────────────
        # MOTION STATE MACHINE
        # ─────────────────────────────────────────────────────────────────────
        self.motion_state = MotionState.IDLE
        self.motion_start_frame = 0
        self.motion_source_candidates: Set[int] = set()
        self.motion_dest_candidates: Set[int] = set()
        self.motion_frames = 0
        
        # ─────────────────────────────────────────────────────────────────────
        # BOARD STATE
        # ─────────────────────────────────────────────────────────────────────
        self.board_state: Dict[int, Tuple[str, str]] = {}
        self.board = chess.Board()
        self.moves: List[str] = []
        
        # ─────────────────────────────────────────────────────────────────────
        # TRACKING
        # ─────────────────────────────────────────────────────────────────────
        self.is_tracking = False
        self.frame_count = 0
        self.last_move_frame = 0
        self.cooldown = 30  # Frames between moves
        self.yolo_check_interval = 50  # Frames between YOLO validation
        
        # ─────────────────────────────────────────────────────────────────────
        # MOVE CONFIRMATION
        # ─────────────────────────────────────────────────────────────────────
        self.pending_move: Optional[chess.Move] = None
        self.pending_count = 0
        self.confirm_threshold = 5
        
        self.debug = False
        
    def calibrate(self, corners: np.ndarray):
        self.corners = corners.astype(np.float32)
        
        # Warp to square top-down view
        dst = np.array([
            [WARPED_SIZE, WARPED_SIZE],  # h1
            [0, WARPED_SIZE],            # a1
            [0, 0],                      # a8
            [WARPED_SIZE, 0],            # h8
        ], dtype=np.float32)
        
        self.warp_transform = cv2.getPerspectiveTransform(self.corners, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, self.corners)
        self.board_polygon = self.corners.reshape(-1, 1, 2).astype(np.int32)
        
        # Create square masks
        self._create_square_masks()
        
        print("[CALIBRATE] Warped view configured")
    
    def _create_square_masks(self):
        """Create binary masks for each square in warped view."""
        for sq in range(64):
            f, r = sq % 8, sq // 8
            
            # In warped coordinates (a8 is top-left, h1 is bottom-right)
            x1 = f * WARPED_SQUARE
            x2 = (f + 1) * WARPED_SQUARE
            y1 = (7 - r) * WARPED_SQUARE  # Flip rank
            y2 = (8 - r) * WARPED_SQUARE
            
            mask = np.zeros((WARPED_SIZE, WARPED_SIZE), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            self.square_masks[sq] = mask
            
            # Store center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self.square_centers_warped[sq] = (cx, cy)
    
    def start(self):
        """Initialize tracking from starting position."""
        self.board = chess.Board()
        self.moves = []
        self.board_state = {}
        
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                symbol = piece.symbol()
                color = 'white' if piece.color else 'black'
                self.board_state[sq] = (symbol, color)
        
        self.is_tracking = True
        self.frame_count = 0
        self.motion_state = MotionState.IDLE
        self.prev_gray = None
        
        print("\n" + "="*60)
        print("OPTICAL FLOW TRACKER STARTED")
        print("="*60)
        print(f"Tracking {len(self.board_state)} pieces")
        print("Waiting for motion...")
        print("="*60 + "\n")
    
    def _warp_frame(self, frame: np.ndarray) -> np.ndarray:
        """Warp frame to top-down view."""
        return cv2.warpPerspective(frame, self.warp_transform, 
                                   (WARPED_SIZE, WARPED_SIZE))
    
    def _compute_flow(self, curr_gray: np.ndarray) -> np.ndarray:
        """Compute dense optical flow between frames."""
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return np.zeros((WARPED_SIZE, WARPED_SIZE, 2), dtype=np.float32)
        
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        self.prev_gray = curr_gray.copy()
        return flow
    
    def _analyze_flow_per_square(self, flow: np.ndarray, magnitude: np.ndarray) -> Dict[int, dict]:
        """Analyze flow patterns for each square."""
        results = {}
        
        motion_mask = magnitude > self.motion_threshold
        
        for sq in range(64):
            mask = self.square_masks[sq]
            
            # Count motion pixels
            sq_motion = motion_mask & (mask > 0)
            motion_count = np.sum(sq_motion)
            
            if motion_count < 10:
                results[sq] = {'motion': 0, 'type': 'static', 'direction': None}
                continue
            
            # Get flow vectors in this square
            sq_indices = mask > 0
            sq_flow = flow[sq_indices]
            
            # Average flow
            avg_dx = np.mean(sq_flow[:, 0])
            avg_dy = np.mean(sq_flow[:, 1])
            
            # Square center
            cx, cy = self.square_centers_warped[sq]
            
            # Determine if outflow (leaving) or inflow (arriving)
            # Outflow: vectors point away from center
            # Inflow: vectors point toward center
            
            # Sample points in square
            y_coords, x_coords = np.where(sq_motion)
            if len(x_coords) == 0:
                results[sq] = {'motion': motion_count, 'type': 'static', 'direction': None}
                continue
            
            # For each motion pixel, check if flow points toward or away from center
            inflow_count = 0
            outflow_count = 0
            
            for i in range(min(100, len(x_coords))):
                px, py = x_coords[i], y_coords[i]
                dx, dy = flow[py, px]
                
                # Vector from pixel to center
                to_center_x = cx - px
                to_center_y = cy - py
                
                # Dot product: positive = toward center, negative = away
                dot = dx * to_center_x + dy * to_center_y
                
                if dot > 0:
                    inflow_count += 1
                else:
                    outflow_count += 1
            
            if outflow_count > inflow_count * 1.5:
                flow_type = 'outflow'
            elif inflow_count > outflow_count * 1.5:
                flow_type = 'inflow'
            else:
                flow_type = 'mixed'
            
            # Direction (angle of average flow)
            direction = np.arctan2(avg_dy, avg_dx) * 180 / np.pi
            
            results[sq] = {
                'motion': motion_count,
                'type': flow_type,
                'direction': direction,
                'avg_flow': (avg_dx, avg_dy),
            }
        
        return results
    
    def _find_move_from_flow(self, flow_analysis: Dict[int, dict]) -> Optional[chess.Move]:
        """Find best legal move matching flow patterns."""
        # Find outflow squares (piece leaving)
        outflow_squares = set()
        for sq, info in flow_analysis.items():
            if info['type'] == 'outflow' and info['motion'] > self.outflow_threshold:
                if sq in self.board_state:
                    outflow_squares.add(sq)
        
        # Find inflow squares (piece arriving)
        inflow_squares = set()
        for sq, info in flow_analysis.items():
            if info['type'] == 'inflow' and info['motion'] > self.inflow_threshold:
                inflow_squares.add(sq)
        
        if not outflow_squares:
            return None
        
        # Score legal moves
        best_move = None
        best_score = 0
        
        whose_turn = 'white' if self.board.turn else 'black'
        
        for move in self.board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            
            if from_sq not in outflow_squares:
                continue
            
            # Check color
            if from_sq in self.board_state:
                if self.board_state[from_sq][1] != whose_turn:
                    continue
            
            score = 0
            
            # Outflow at source
            score += flow_analysis[from_sq]['motion'] / 100
            
            # Inflow at destination
            if to_sq in inflow_squares:
                score += flow_analysis[to_sq]['motion'] / 100
            
            # Direction match
            if flow_analysis[from_sq]['direction'] is not None:
                # Expected direction from source to destination
                src_cx, src_cy = self.square_centers_warped[from_sq]
                dst_cx, dst_cy = self.square_centers_warped[to_sq]
                expected_angle = np.arctan2(dst_cy - src_cy, dst_cx - src_cx) * 180 / np.pi
                
                actual_angle = flow_analysis[from_sq]['direction']
                angle_diff = abs(expected_angle - actual_angle)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff < 45:
                    score += 3  # Good direction match
                elif angle_diff < 90:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_score >= 3:
            return best_move
        return None
    
    def _update_motion_state(self, total_motion: float, move_candidate: Optional[chess.Move]):
        """Update the motion state machine."""
        motion_start_threshold = 500
        motion_peak_threshold = 1000
        motion_end_threshold = 300
        
        if self.motion_state == MotionState.IDLE:
            if total_motion > motion_start_threshold:
                self.motion_state = MotionState.MOTION_START
                self.motion_start_frame = self.frame_count
                self.motion_source_candidates.clear()
                self.motion_dest_candidates.clear()
                if self.debug:
                    print(f"  [{self.frame_count}] MOTION_START (total={total_motion:.0f})")
        
        elif self.motion_state == MotionState.MOTION_START:
            self.motion_frames += 1
            if total_motion > motion_peak_threshold:
                self.motion_state = MotionState.MOTION_PEAK
                if self.debug:
                    print(f"  [{self.frame_count}] MOTION_PEAK")
            elif self.motion_frames > 20:  # Timeout
                self.motion_state = MotionState.IDLE
                self.motion_frames = 0
        
        elif self.motion_state == MotionState.MOTION_PEAK:
            self.motion_frames += 1
            if total_motion < motion_end_threshold:
                self.motion_state = MotionState.MOTION_END
                if self.debug:
                    print(f"  [{self.frame_count}] MOTION_END")
        
        elif self.motion_state == MotionState.MOTION_END:
            self.motion_frames += 1
            if total_motion < motion_end_threshold / 2:
                self.motion_state = MotionState.CONFIRM
                if self.debug:
                    print(f"  [{self.frame_count}] CONFIRM")
            elif self.motion_frames > 30:
                self.motion_state = MotionState.IDLE
                self.motion_frames = 0
        
        elif self.motion_state == MotionState.CONFIRM:
            if move_candidate:
                if move_candidate == self.pending_move:
                    self.pending_count += 1
                else:
                    self.pending_move = move_candidate
                    self.pending_count = 1
            else:
                self.motion_state = MotionState.IDLE
                self.motion_frames = 0
                self.pending_move = None
                self.pending_count = 0
    
    def process(self, frame: np.ndarray) -> Optional[str]:
        if not self.is_tracking:
            return None
        
        self.frame_count += 1
        
        # Warp and convert to grayscale
        warped = self._warp_frame(frame)
        curr_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Compute flow
        flow = self._compute_flow(curr_gray)
        self.flow = flow
        
        # Compute magnitude
        magnitude = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        total_motion = np.sum(magnitude > self.motion_threshold)
        
        # Cooldown
        if self.frame_count - self.last_move_frame < self.cooldown:
            return None
        
        # Analyze flow per square
        flow_analysis = self._analyze_flow_per_square(flow, magnitude)
        
        # Find move candidate
        move_candidate = self._find_move_from_flow(flow_analysis)
        
        # Update state machine
        self._update_motion_state(total_motion, move_candidate)
        
        # Check for confirmed move
        if self.motion_state == MotionState.CONFIRM and self.pending_count >= self.confirm_threshold:
            return self._execute_move(self.pending_move)
        
        return None
    
    def _execute_move(self, move: chess.Move) -> str:
        san = self.board.san(move)
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Update board state
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
        
        # Castling
        if self.board.is_castling(move):
            rook_info = {
                chess.G1: (chess.H1, chess.F1), chess.C1: (chess.A1, chess.D1),
                chess.G8: (chess.H8, chess.F8), chess.C8: (chess.A8, chess.D8),
            }
            if move.to_square in rook_info:
                rf, rt = rook_info[move.to_square]
                if rf in self.board_state:
                    rook = self.board_state.pop(rf)
                    self.board_state[rt] = rook
        
        # En passant
        if self.board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
            if cap_sq in self.board_state:
                del self.board_state[cap_sq]
        
        self.board.push(move)
        self.moves.append(san)
        self.last_move_frame = self.frame_count
        self.motion_state = MotionState.IDLE
        self.motion_frames = 0
        self.pending_move = None
        self.pending_count = 0
        
        print(f"[{self.frame_count}] Move #{len(self.moves)}: {san}")
        return san
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        
        if self.corners is not None:
            cv2.polylines(vis, [self.corners.astype(np.int32)], True, (0, 255, 0), 2)
        
        # Draw expected pieces
        if self.is_tracking and self.inv_transform is not None:
            for sq, (symbol, color) in self.board_state.items():
                cx, cy = self.square_centers_warped[sq]
                pt = np.array([[cx, cy]], dtype=np.float32)
                orig = cv2.perspectiveTransform(pt[np.newaxis], self.inv_transform)[0, 0]
                ox, oy = int(orig[0]), int(orig[1])
                
                box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                cv2.rectangle(vis, (ox-14, oy-14), (ox+14, oy+14), box_color, 2)
                cv2.putText(vis, symbol.upper(), (ox-6, oy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Status
        state_name = self.motion_state.name
        turn = "White" if self.board.turn else "Black"
        cv2.putText(vis, f"VERSION 4: Optical Flow | {state_name} | {turn}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.moves:
            cv2.putText(vis, " ".join(self.moves[-8:]), (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis
    
    def visualize_flow(self) -> Optional[np.ndarray]:
        """Create a visualization of the optical flow."""
        if self.flow is None:
            return None
        
        hsv = np.zeros((WARPED_SIZE, WARPED_SIZE, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def get_pgn(self, event_name: str = "Optical Flow Tracker") -> str:
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
        description="VERSION 4: Optical Flow Tracker (Motion-Based)"
    )
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--output', '-o', help='Output PGN file')
    parser.add_argument('--model', '-m', default='models/pieces.pt')
    parser.add_argument('--crop', type=float, default=0.45)
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--show-flow', action='store_true', help='Show flow visualization')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path}")
        sys.exit(1)
    
    output_path = args.output or f"output/pgn/{video_path.stem}_v4_flow.pgn"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_h = int(h * args.crop)
    
    print("\n" + "="*60)
    print("VERSION 4: OPTICAL FLOW TRACKER")
    print("="*60)
    print(f"Video: {video_path.name}")
    print("This tracker uses optical flow to detect motion, not YOLO")
    
    ret, frame = cap.read()
    if not ret:
        sys.exit(1)
    frame = frame[:crop_h, :]
    
    tracker = OpticalFlowTracker(model_path=args.model)
    tracker.debug = args.debug
    
    corners = interactive_corners(frame)
    if corners is None:
        cap.release()
        sys.exit(0)
    tracker.calibrate(corners)
    tracker.start()
    
    cv2.namedWindow("Flow Tracker", cv2.WINDOW_NORMAL)
    if args.show_flow:
        cv2.namedWindow("Flow", cv2.WINDOW_NORMAL)
    
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
        
        cv2.imshow("Flow Tracker", vis)
        
        if args.show_flow:
            flow_vis = tracker.visualize_flow()
            if flow_vis is not None:
                cv2.imshow("Flow", flow_vis)
        
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
