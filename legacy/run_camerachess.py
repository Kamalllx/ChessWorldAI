#!/usr/bin/env python3
"""
ChessWorldAI - CameraChess Hybrid Tracker
=========================================
Combines the best of CameraChessWeb (state-based detection) with our YOLO pipeline.

Key features from CameraChessWeb:
1. 64x13 state matrix (each square has class confidence scores)
2. State decay - old detections fade over time  
3. Move scoring based on "from" squares being empty and "to" squares having expected piece
4. Pre-computed legal move pairs for efficient matching
5. Score thresholds for move confirmation

Key features from our approach:
1. Interactive corner calibration
2. YOLO detection with clustering
3. Perspective transform to map detections to squares
4. PGN export

Usage:
    python run_camerachess.py videos/game_4.mp4 --speed 2.0
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
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BOARD_SIZE, SQUARE_SIZE

# =============================================================================
# CONSTANTS
# =============================================================================

SQUARE_NAMES = [f"{chr(ord('a')+f)}{r+1}" for r in range(8) for f in range(8)]

# Class labels matching CameraChessWeb format (12 classes for pieces, 0=empty)
# Our model: 0=bishop(generic), 1-6=black, 7-12=white
# CameraChessWeb: b, k, n, p, q, r, B, K, N, P, Q, R (0-11)

# Map from our YOLO class to piece info
YOLO_CLASS_TO_PIECE = {
    0: None,     # generic bishop - skip
    1: ('b', 'black', 0),   # black-bishop -> label index 0
    2: ('k', 'black', 1),   # black-king -> label index 1
    3: ('n', 'black', 2),   # black-knight -> label index 2
    4: ('p', 'black', 3),   # black-pawn -> label index 3
    5: ('q', 'black', 4),   # black-queen -> label index 4
    6: ('r', 'black', 5),   # black-rook -> label index 5
    7: ('B', 'white', 6),   # white-bishop -> label index 6
    8: ('K', 'white', 7),   # white-king -> label index 7
    9: ('N', 'white', 8),   # white-knight -> label index 8
    10: ('P', 'white', 9),  # white-pawn -> label index 9
    11: ('Q', 'white', 10), # white-queen -> label index 10
    12: ('R', 'white', 11), # white-rook -> label index 11
}

# Reverse: label index to chess symbol
LABEL_TO_SYMBOL = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']
SYMBOL_TO_LABEL = {s: i for i, s in enumerate(LABEL_TO_SYMBOL)}

# Chess.js piece type to label index
PIECE_TO_LABEL = {
    ('b', 'black'): 0, ('k', 'black'): 1, ('n', 'black'): 2,
    ('p', 'black'): 3, ('q', 'black'): 4, ('r', 'black'): 5,
    ('b', 'white'): 6, ('k', 'white'): 7, ('n', 'white'): 8,
    ('p', 'white'): 9, ('q', 'white'): 10, ('r', 'white'): 11,
}

# Castling rook positions: to_square -> (rook_from, rook_to, rook_label_idx)
CASTLING_MAP = {
    chess.G1: (chess.H1, chess.F1, 11),  # white kingside
    chess.C1: (chess.A1, chess.D1, 11),  # white queenside
    chess.G8: (chess.H8, chess.F8, 5),   # black kingside
    chess.C8: (chess.A8, chess.D8, 5),   # black queenside
}


# =============================================================================
# DATA STRUCTURES (from CameraChessWeb)
# =============================================================================

@dataclass
class MoveData:
    """Data about a move for scoring."""
    san: str
    from_squares: List[int]  # Squares that should be empty after move
    to_squares: List[int]    # Squares that should have pieces after move
    targets: List[int]       # Expected piece label index at each to_square


@dataclass  
class MovePair:
    """A legal move and all possible opponent responses."""
    move1: MoveData
    move2: Optional[MoveData]  # None if it's checkmate/stalemate
    combined: Optional[MoveData]  # Combined data for both moves


def get_square_centers() -> np.ndarray:
    """Get center coordinates for each square in board space."""
    centers = []
    for sq in range(64):
        f, r = sq % 8, sq // 8
        x = (f + 0.5) * SQUARE_SIZE
        y = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
        centers.append([x, y])
    return np.array(centers)


def zeros(rows: int, cols: int) -> List[List[float]]:
    """Create 2D zero array."""
    return [[0.0] * cols for _ in range(rows)]


# =============================================================================
# MOVE DATA EXTRACTION
# =============================================================================

def get_move_data(board: chess.Board, move: chess.Move) -> MoveData:
    """Extract move data in CameraChessWeb format."""
    from_squares = [move.from_square]
    to_squares = [move.to_square]
    
    # Get piece info
    piece = board.piece_at(move.from_square)
    piece_type = piece.symbol().lower()
    color = 'white' if piece.color else 'black'
    
    # Handle promotion
    if move.promotion:
        promo_map = {chess.QUEEN: 'q', chess.ROOK: 'r', 
                     chess.BISHOP: 'b', chess.KNIGHT: 'n'}
        piece_type = promo_map.get(move.promotion, 'q')
    
    targets = [PIECE_TO_LABEL[(piece_type, color)]]
    
    # Castling - add rook movement
    if board.is_castling(move):
        info = CASTLING_MAP.get(move.to_square)
        if info:
            rook_from, rook_to, rook_label = info
            from_squares.append(rook_from)
            to_squares.append(rook_to)
            targets.append(rook_label)
    
    # En passant - captured pawn square should be empty
    elif board.is_en_passant(move):
        cap_sq = chess.square(chess.square_file(move.to_square), 
                              chess.square_rank(move.from_square))
        from_squares.append(cap_sq)
    
    san = board.san(move)
    return MoveData(san=san, from_squares=from_squares, 
                    to_squares=to_squares, targets=targets)


def combine_move_data(data1: MoveData, data2: MoveData) -> MoveData:
    """Combine two consecutive moves, filtering overlapping squares."""
    bad_squares = data2.from_squares + data2.to_squares
    
    # Filter move1's data
    from1 = [sq for sq in data1.from_squares if sq not in bad_squares]
    to1, targets1 = [], []
    for i, sq in enumerate(data1.to_squares):
        if sq not in bad_squares:
            to1.append(sq)
            targets1.append(data1.targets[i])
    
    # Combine
    from_all = from1 + data2.from_squares
    to_all = to1 + data2.to_squares
    targets_all = targets1 + data2.targets
    
    return MoveData(san=f"{data1.san} {data2.san}",
                    from_squares=from_all, to_squares=to_all, targets=targets_all)


def get_move_pairs(board: chess.Board) -> List[MovePair]:
    """Get all legal moves with possible responses (CameraChessWeb style)."""
    pairs = []
    
    for move1 in board.legal_moves:
        data1 = get_move_data(board, move1)
        board.push(move1)
        
        if board.legal_moves.count() == 0:
            # Checkmate or stalemate
            pairs.append(MovePair(move1=data1, move2=None, combined=None))
        else:
            for move2 in board.legal_moves:
                data2 = get_move_data(board, move2)
                combined = combine_move_data(data1, data2)
                pairs.append(MovePair(move1=data1, move2=data2, combined=combined))
        
        board.pop()
    
    return pairs


# =============================================================================
# CORNER CALIBRATION
# =============================================================================

def interactive_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """Click 4 corners: h1, a1, a8, h8."""
    corners = []
    display = frame.copy()
    names = ["h1 (white's right)", "a1 (white's left)", 
             "a8 (black's left)", "h8 (black's right)"]
    short_names = ["h1", "a1", "a8", "h8"]
    
    def mouse_cb(event, x, y, flags, param):
        nonlocal corners, display
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([x, y])
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display, short_names[len(corners)-1], (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(corners) > 1:
                cv2.line(display, tuple(corners[-2]), tuple(corners[-1]), (0, 255, 0), 2)
            if len(corners) == 4:
                cv2.line(display, tuple(corners[3]), tuple(corners[0]), (0, 255, 0), 2)
                
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", mouse_cb)
    
    print("\n" + "="*50)
    print("CORNER CALIBRATION")
    print("="*50)
    print("Click corners in order:")
    for i, name in enumerate(names):
        print(f"  {i+1}. {name}")
    print("\nPress 'c' to confirm, 'r' to reset, 'q' to quit")
    
    while True:
        temp = display.copy()
        if len(corners) < 4:
            cv2.putText(temp, f"Click: {names[len(corners)]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(temp, "Press 'c' to confirm corners", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", temp)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('c') and len(corners) == 4:
            cv2.destroyWindow("Calibration")
            return np.array(corners, dtype=np.float32)
        elif key == ord('r'):
            corners = []
            display = frame.copy()
        elif key == ord('q'):
            cv2.destroyWindow("Calibration")
            return None
    return None


# =============================================================================
# MAIN TRACKER CLASS (CameraChess Hybrid)
# =============================================================================

class CameraChessTracker:
    """
    Hybrid tracker combining CameraChessWeb's state-based approach with YOLO.
    
    Key insight from CameraChessWeb:
    - Maintain a 64x12 state matrix where state[sq][cls] = confidence that 
      piece class 'cls' is on square 'sq'
    - Decay old state, update with new detections
    - Score moves by checking if "from" squares are empty (low max confidence)
      and "to" squares have expected pieces (high specific confidence)
    """
    
    def __init__(self, model_path: str = 'models/pieces.pt'):
        from ultralytics import YOLO
        
        self.model = YOLO(model_path)
        print(f"✓ Loaded model: {model_path}")
        
        # Geometry
        self.corners: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.inv_transform: Optional[np.ndarray] = None
        self.square_centers = get_square_centers()
        self.board_polygon: Optional[np.ndarray] = None
        
        # State matrix: 64 squares x 12 piece classes
        # state[sq][cls] = confidence that piece 'cls' is on 'sq'
        self.state: List[List[float]] = zeros(64, 12)
        
        # Decay factor - how fast old detections fade
        self.decay = 0.5  # CameraChessWeb default
        
        # Score thresholds for move detection
        self.from_threshold = 0.5  # "From" square should have max_score < this
        self.to_threshold = 0.5    # "To" square should have target_score > this
        
        # Chess engine
        self.board = chess.Board()
        self.moves: List[str] = []
        self.move_pairs: List[MovePair] = []
        
        # Possible moves tracking (CameraChessWeb style)
        self.possible_moves: Set[str] = set()
        
        # Greedy move confirmation (move after 1 second of stable detection)
        self.greedy_move_times: Dict[str, float] = {}
        self.greedy_timeout = 1.0  # seconds
        
        # Tracking
        self.is_tracking = False
        self.frame_count = 0
        self.last_move_time = 0
        
        # Detection parameters
        self.conf_threshold = 0.15
        self.cluster_distance = 30
        
        # Debug
        self.debug_mode = False
        self.last_update: List[List[float]] = zeros(64, 12)
        self.last_raw_detections: List[dict] = []
        
    def calibrate(self, corners: np.ndarray):
        """Set up perspective transform."""
        self.corners = corners.astype(np.float32)
        dst = np.array([
            [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE],
            [0, 0], [BOARD_SIZE, 0],
        ], dtype=np.float32)
        self.transform = cv2.getPerspectiveTransform(self.corners, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, self.corners)
        self.board_polygon = self.corners.reshape(-1, 1, 2).astype(np.int32)
        
    def start(self):
        """Start tracking from standard starting position."""
        self.is_tracking = True
        self.state = zeros(64, 12)
        self.board = chess.Board()
        self.moves = []
        self.move_pairs = get_move_pairs(self.board)
        self.possible_moves = set()
        self.greedy_move_times = {}
        
        # Initialize state with starting position
        self._init_starting_state()
        
        print("\n" + "="*50)
        print("TRACKING STARTED - CameraChess Hybrid")
        print("="*50)
        print("Controls: Q=quit, SPACE=pause, D=debug, +/-=speed")
        
    def _init_starting_state(self):
        """Initialize state matrix with starting position."""
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                piece_type = piece.symbol().lower()
                color = 'white' if piece.color else 'black'
                label_idx = PIECE_TO_LABEL.get((piece_type, color))
                if label_idx is not None:
                    self.state[sq][label_idx] = 1.0
    
    def _is_inside_board(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside board boundary."""
        if self.board_polygon is None:
            return True
        result = cv2.pointPolygonTest(self.board_polygon, 
                                       (float(point[0]), float(point[1])), False)
        return result >= 0
    
    def _cluster_detections(self, detections: List[dict]) -> List[dict]:
        """Cluster close detections, keep highest confidence."""
        if len(detections) <= 1:
            return detections
            
        sorted_dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        kept = []
        used = [False] * len(sorted_dets)
        
        for i, det in enumerate(sorted_dets):
            if used[i]:
                continue
            used[i] = True
            kept.append(det)
            
            cx1, cy1 = det['center']
            for j in range(i + 1, len(sorted_dets)):
                if used[j]:
                    continue
                cx2, cy2 = sorted_dets[j]['center']
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if dist < self.cluster_distance:
                    used[j] = True
        return kept
    
    def _detect(self, frame: np.ndarray) -> List[dict]:
        """Run YOLO and get filtered detections."""
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    
                    if not self._is_inside_board((cx, cy)):
                        continue
                    if cls == 0:  # Skip generic
                        continue
                    
                    piece_info = YOLO_CLASS_TO_PIECE.get(cls)
                    if piece_info is None:
                        continue
                    
                    symbol, color, label_idx = piece_info
                    detections.append({
                        'center': (cx, cy),
                        'class': cls,
                        'label_idx': label_idx,
                        'confidence': conf,
                        'symbol': symbol,
                        'bbox': box
                    })
        
        clustered = self._cluster_detections(detections)
        self.last_raw_detections = clustered
        return clustered
    
    def _point_to_square(self, point: Tuple[float, float]) -> Optional[int]:
        """Map image point to square index."""
        pt = np.array([[point]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.transform)[0, 0]
        
        if 0 <= warped[0] < BOARD_SIZE and 0 <= warped[1] < BOARD_SIZE:
            dists = np.linalg.norm(self.square_centers - warped, axis=1)
            return int(np.argmin(dists))
        return None
    
    def _get_update(self, detections: List[dict]) -> List[List[float]]:
        """
        Build update matrix from detections (CameraChessWeb style).
        update[sq][cls] = max confidence of class 'cls' detected on 'sq'
        """
        update = zeros(64, 12)
        
        for det in detections:
            sq = self._point_to_square(det['center'])
            if sq is not None:
                label_idx = det['label_idx']
                update[sq][label_idx] = max(update[sq][label_idx], det['confidence'])
        
        return update
    
    def _update_state(self, update: List[List[float]]):
        """
        Update state with decay (CameraChessWeb style).
        new_state = decay * old_state + (1 - decay) * update
        """
        for sq in range(64):
            for cls in range(12):
                self.state[sq][cls] = (self.decay * self.state[sq][cls] + 
                                       (1 - self.decay) * update[sq][cls])
        self.last_update = update
    
    def _calculate_score(self, move_data: MoveData) -> float:
        """
        Calculate move score (CameraChessWeb style).
        - "From" squares should be empty (1 - max_score - threshold)
        - "To" squares should have expected piece (target_score - threshold)
        """
        score = 0.0
        
        # From squares should be empty
        for sq in move_data.from_squares:
            max_score = max(self.state[sq])
            score += (1.0 - max_score - self.from_threshold)
        
        # To squares should have expected pieces
        for i, sq in enumerate(move_data.to_squares):
            target_cls = move_data.targets[i]
            target_score = self.state[sq][target_cls]
            score += (target_score - self.to_threshold)
        
        return score
    
    def _process_state(self) -> Tuple[Optional[MoveData], Optional[MoveData], float, float]:
        """
        Find best matching move pair (CameraChessWeb style).
        Returns: (best_move1, best_combined, best_score1, best_joint_score)
        """
        best_score1 = float('-inf')
        best_score2 = float('-inf')
        best_joint_score = float('-inf')
        best_move1: Optional[MoveData] = None
        best_combined: Optional[MoveData] = None
        seen: Set[str] = set()
        
        for pair in self.move_pairs:
            # Score move1 (only once per SAN)
            if pair.move1.san not in seen:
                seen.add(pair.move1.san)
                score1 = self._calculate_score(pair.move1)
                
                if score1 > 0:
                    self.possible_moves.add(pair.move1.san)
                
                if score1 > best_score1:
                    best_score1 = score1
                    best_move1 = pair.move1
            
            # Score combined (both moves)
            if pair.move2 is None or pair.combined is None:
                continue
            if pair.move1.san not in self.possible_moves:
                continue
            
            score2 = self._calculate_score(pair.move2)
            if score2 < 0:
                continue
            if score2 > best_score2:
                best_score2 = score2
            
            joint_score = self._calculate_score(pair.combined)
            if joint_score > best_joint_score:
                best_joint_score = joint_score
                best_combined = pair.combined
        
        return best_move1, best_combined, best_score1, best_joint_score
    
    def process(self, frame: np.ndarray, current_time: float) -> Optional[str]:
        """Process frame and detect moves."""
        if not self.is_tracking or self.transform is None:
            return None
        
        self.frame_count += 1
        
        # Detect pieces
        detections = self._detect(frame)
        
        # Build update and update state
        update = self._get_update(detections)
        self._update_state(update)
        
        # Find best moves
        best_move1, best_combined, score1, joint_score = self._process_state()
        
        # Check for confirmed move (CameraChessWeb style)
        executed = None
        
        # Two-move confirmation (opponent already moved)
        if best_combined is not None:
            move_san = best_combined.san.split()[0]  # First move only
            has_move = (score1 > 0 and joint_score > 0 and 
                       move_san in self.possible_moves)
            if has_move:
                executed = self._execute_move(move_san)
        
        # Greedy single-move confirmation (after timeout)
        if executed is None and best_move1 is not None and score1 > 0:
            move_san = best_move1.san
            
            if move_san not in self.greedy_move_times:
                self.greedy_move_times[move_san] = current_time
            
            elapsed = current_time - self.greedy_move_times[move_san]
            if elapsed > self.greedy_timeout:
                executed = self._execute_move(move_san)
                self.greedy_move_times = {}
        
        return executed
    
    def _execute_move(self, san: str) -> Optional[str]:
        """Execute a move on the board."""
        try:
            move = self.board.parse_san(san)
            self.board.push(move)
            self.moves.append(san)
            
            # Regenerate move pairs for new position
            self.move_pairs = get_move_pairs(self.board)
            self.possible_moves.clear()
            
            print(f"[Frame {self.frame_count}] Move #{len(self.moves)}: {san}")
            return san
            
        except Exception as e:
            if self.debug_mode:
                print(f"  Move error: {san} - {e}")
            return None
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Draw visualization overlay."""
        vis = frame.copy()
        
        if self.corners is not None:
            cv2.polylines(vis, [self.corners.astype(np.int32)], True, (0, 255, 0), 2)
        
        if self.is_tracking and self.inv_transform is not None:
            # Draw each square with color based on detection
            for sq in range(64):
                f, r = sq % 8, sq // 8
                wx = (f + 0.5) * SQUARE_SIZE
                wy = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
                pt = np.array([[wx, wy]], dtype=np.float32)
                center = cv2.perspectiveTransform(pt[np.newaxis], self.inv_transform)[0, 0]
                cx, cy = int(center[0]), int(center[1])
                
                # Get piece on this square according to chess board
                piece = self.board.piece_at(sq)
                if piece:
                    piece_type = piece.symbol().lower()
                    color = 'white' if piece.color else 'black'
                    label_idx = PIECE_TO_LABEL.get((piece_type, color))
                    
                    # Get detection confidence for this piece
                    conf = self.state[sq][label_idx] if label_idx is not None else 0
                    
                    # Color based on confidence
                    if conf > 0.5:
                        box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                    elif conf > 0.2:
                        box_color = (0, 165, 255) if color == 'white' else (180, 100, 180)
                    else:
                        box_color = (0, 0, 255)  # Low confidence = red
                    
                    symbol = piece.symbol()
                    cv2.rectangle(vis, (cx-16, cy-16), (cx+16, cy+16), box_color, 2)
                    cv2.putText(vis, symbol.upper(), (cx-6, cy+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # Debug: show raw detections
            if self.debug_mode:
                for det in self.last_raw_detections:
                    dcx, dcy = int(det['center'][0]), int(det['center'][1])
                    label = f"{det['symbol']}:{det['confidence']:.2f}"
                    cv2.circle(vis, (dcx, dcy), 4, (0, 255, 255), -1)
                    cv2.putText(vis, label, (dcx+5, dcy-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        # Status
        turn = "White" if self.board.turn else "Black"
        status = f"Moves: {len(self.moves)} | {turn} to move"
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.moves:
            recent = self.moves[-10:]
            move_str = " ".join(recent)
            cv2.putText(vis, move_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        return vis
    
    def pgn(self, name: str) -> str:
        """Generate PGN string."""
        game = chess.pgn.Game()
        game.headers["Event"] = "ChessWorldAI CameraChess"
        game.headers["Site"] = name
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        
        node = game
        board = chess.Board()
        for san in self.moves:
            try:
                move = board.parse_san(san)
                node = node.add_variation(move)
                board.push(move)
            except Exception as e:
                print(f"Warning: Could not parse '{san}': {e}")
                continue
        
        if board.is_checkmate():
            game.headers["Result"] = "1-0" if not board.turn else "0-1"
        elif board.is_stalemate() or board.is_insufficient_material():
            game.headers["Result"] = "1/2-1/2"
        else:
            game.headers["Result"] = "*"
        
        return str(game)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ChessWorldAI CameraChess Hybrid")
    parser.add_argument('video', help='Video file')
    parser.add_argument('--output', '-o', help='Output PGN')
    parser.add_argument('--model', '-m', default='models/pieces.pt')
    parser.add_argument('--speed', '-s', type=float, default=1.0)
    parser.add_argument('--crop', type=float, default=0.45)
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--decay', type=float, default=0.5, help='State decay factor')
    parser.add_argument('--from-thr', type=float, default=0.5, help='From threshold')
    parser.add_argument('--to-thr', type=float, default=0.5, help='To threshold')
    
    args = parser.parse_args()
    
    video = Path(args.video)
    if not video.exists():
        print(f"Error: {video} not found")
        sys.exit(1)
    
    # Find model
    model_path = args.model
    if not Path(model_path).exists():
        for alt in ['models/pieces.pt', 'models/pieces_trained.pt']:
            if Path(alt).exists():
                model_path = alt
                break
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    output = args.output or f"output/pgn/{video.stem}_camerachess.pgn"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*50}")
    print("ChessWorldAI - CameraChess Hybrid")
    print(f"{'='*50}")
    print(f"Video: {video.name}")
    print(f"Model: {model_path}")
    print(f"Decay: {args.decay}, From threshold: {args.from_thr}, To threshold: {args.to_thr}")
    
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_h = int(h * args.crop)
    
    print(f"Resolution: {w}x{h} -> {w}x{crop_h} (cropped)")
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        sys.exit(1)
    frame = frame[:crop_h, :]
    
    tracker = CameraChessTracker(model_path=model_path)
    tracker.debug_mode = args.debug
    tracker.decay = args.decay
    tracker.from_threshold = args.from_thr
    tracker.to_threshold = args.to_thr
    
    corners = interactive_corners(frame)
    if corners is None:
        print("Calibration cancelled")
        sys.exit(0)
    
    tracker.calibrate(corners)
    tracker.start()
    
    cv2.namedWindow("CameraChess", cv2.WINDOW_NORMAL)
    
    paused = False
    speed = args.speed
    frame_delay = max(1, int((1000 / fps) / speed))
    
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame = frame[:crop_h, :]
            
            current_time = (cv2.getTickCount() / cv2.getTickFrequency()) - start_time
            tracker.process(frame, current_time)
        
        vis = tracker.visualize(frame)
        
        # Progress bar
        progress = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / total
        bar_w = int(vis.shape[1] * 0.8)
        bar_x = int(vis.shape[1] * 0.1)
        bar_y = vis.shape[0] - 20
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 10), (0, 255, 0), -1)
        
        cv2.imshow("CameraChess", vis)
        
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord('d'):
            tracker.debug_mode = not tracker.debug_mode
            print(f"Debug: {'ON' if tracker.debug_mode else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            speed = min(8.0, speed * 1.5)
            frame_delay = max(1, int((1000 / fps) / speed))
            print(f"Speed: {speed:.1f}x")
        elif key == ord('-'):
            speed = max(0.25, speed / 1.5)
            frame_delay = max(1, int((1000 / fps) / speed))
            print(f"Speed: {speed:.1f}x")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save PGN
    pgn_content = tracker.pgn(video.stem)
    with open(output, 'w') as f:
        f.write(pgn_content)
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Moves detected: {len(tracker.moves)}")
    print(f"PGN saved to: {output}")
    print(f"\nMoves: {' '.join(tracker.moves)}")
    print(f"\n{pgn_content}")


if __name__ == "__main__":
    main()
