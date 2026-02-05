#!/usr/bin/env python3
"""
ChessWorldAI - Hybrid Tracker
==============================
1. Detect initial 32 pieces using YOLO
2. Track them using state matrix + model scoring
3. Normal video playback speed
4. All detection within board borders

Usage:
    python run_hybrid.py videos/game_4.mp4
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import sys
import argparse
import chess
import chess.pgn
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.config import BOARD_SIZE, SQUARE_SIZE

# =============================================================================
# CONSTANTS
# =============================================================================

SQUARE_NAMES = [f"{chr(ord('a')+f)}{r+1}" for r in range(8) for f in range(8)]

# YOLO class -> (symbol, color, label_index)
YOLO_CLASS_MAP = {
    0: None,  # generic - skip
    1: ('b', 'black', 0),   # black-bishop
    2: ('k', 'black', 1),   # black-king
    3: ('n', 'black', 2),   # black-knight
    4: ('p', 'black', 3),   # black-pawn
    5: ('q', 'black', 4),   # black-queen
    6: ('r', 'black', 5),   # black-rook
    7: ('B', 'white', 6),   # white-bishop
    8: ('K', 'white', 7),   # white-king
    9: ('N', 'white', 8),   # white-knight
    10: ('P', 'white', 9),  # white-pawn
    11: ('Q', 'white', 10), # white-queen
    12: ('R', 'white', 11), # white-rook
}

PIECE_TO_LABEL = {
    ('b', 'black'): 0, ('k', 'black'): 1, ('n', 'black'): 2,
    ('p', 'black'): 3, ('q', 'black'): 4, ('r', 'black'): 5,
    ('b', 'white'): 6, ('k', 'white'): 7, ('n', 'white'): 8,
    ('p', 'white'): 9, ('q', 'white'): 10, ('r', 'white'): 11,
}

LABEL_TO_SYMBOL = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']


def get_square_centers() -> np.ndarray:
    centers = []
    for sq in range(64):
        f, r = sq % 8, sq // 8
        x = (f + 0.5) * SQUARE_SIZE
        y = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
        centers.append([x, y])
    return np.array(centers)


# =============================================================================
# CORNER CALIBRATION
# =============================================================================

def interactive_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    corners = []
    display = frame.copy()
    names = ["h1", "a1", "a8", "h8"]
    
    def mouse_cb(event, x, y, flags, param):
        nonlocal corners, display
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([x, y])
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display, names[len(corners)-1], (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(corners) > 1:
                cv2.line(display, tuple(corners[-2]), tuple(corners[-1]), (0, 255, 0), 2)
            if len(corners) == 4:
                cv2.line(display, tuple(corners[3]), tuple(corners[0]), (0, 255, 0), 2)
                
    cv2.namedWindow("Click Corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click Corners", mouse_cb)
    
    print("\nClick corners: h1 -> a1 -> a8 -> h8")
    print("'c' = confirm, 'r' = reset, 'q' = quit")
    
    while True:
        temp = display.copy()
        if len(corners) < 4:
            cv2.putText(temp, f"Click: {names[len(corners)]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(temp, "Press 'c' to confirm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Click Corners", temp)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('c') and len(corners) == 4:
            cv2.destroyWindow("Click Corners")
            return np.array(corners, dtype=np.float32)
        elif key == ord('r'):
            corners = []
            display = frame.copy()
        elif key == ord('q'):
            cv2.destroyWindow("Click Corners")
            return None


# =============================================================================
# HYBRID TRACKER
# =============================================================================

class HybridTracker:
    """
    Hybrid approach:
    1. Detect initial 32 pieces with YOLO
    2. Maintain 64x12 state matrix
    3. Score moves using state + model detections
    4. All within board borders
    """
    
    def __init__(self, model_path: str = 'models/pieces.pt'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        print(f"✓ Model: {model_path}")
        
        # Geometry
        self.corners: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.inv_transform: Optional[np.ndarray] = None
        self.square_centers = get_square_centers()
        self.board_polygon: Optional[np.ndarray] = None
        
        # State: 64 squares x 12 piece types
        self.state = np.zeros((64, 12), dtype=np.float32)
        self.decay = 0.7  # How fast old detections fade
        
        # Chess
        self.board = chess.Board()
        self.moves: List[str] = []
        
        # Tracking state
        self.is_tracking = False
        self.frame_count = 0
        self.last_move_frame = 0
        self.cooldown_frames = 15  # Wait longer after moves
        
        # Move confirmation
        self.pending_san: Optional[str] = None
        self.pending_frames = 0
        self.confirm_frames = 8  # Need more consistent detection
        
        # Detection params
        self.conf_threshold = 0.15
        
        # Debug
        self.last_detections: List[dict] = []
        self.debug = False
        
    def calibrate(self, corners: np.ndarray):
        """Set perspective transform from clicked corners."""
        self.corners = corners.astype(np.float32)
        dst = np.array([
            [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE],
            [0, 0], [BOARD_SIZE, 0],
        ], dtype=np.float32)
        self.transform = cv2.getPerspectiveTransform(self.corners, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, self.corners)
        self.board_polygon = self.corners.reshape(-1, 1, 2).astype(np.int32)
        
    def _is_inside_board(self, x: float, y: float) -> bool:
        """Check if point is inside the board polygon."""
        if self.board_polygon is None:
            return True
        return cv2.pointPolygonTest(self.board_polygon, (x, y), False) >= 0
    
    def _point_to_square(self, x: float, y: float) -> Optional[int]:
        """Map image point to square index (0-63)."""
        if not self._is_inside_board(x, y):
            return None
        pt = np.array([[[x, y]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.transform)[0, 0]
        if 0 <= warped[0] < BOARD_SIZE and 0 <= warped[1] < BOARD_SIZE:
            dists = np.linalg.norm(self.square_centers - warped, axis=1)
            return int(np.argmin(dists))
        return None
    
    def _detect(self, frame: np.ndarray) -> List[dict]:
        """Run YOLO detection, filter to board only."""
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
                
                # ONLY inside board
                if not self._is_inside_board(cx, cy):
                    continue
                
                info = YOLO_CLASS_MAP.get(cls)
                if info is None:
                    continue
                
                symbol, color, label = info
                sq = self._point_to_square(cx, cy)
                
                detections.append({
                    'center': (cx, cy),
                    'square': sq,
                    'label': label,
                    'conf': conf,
                    'symbol': symbol,
                    'color': color,
                })
        
        self.last_detections = detections
        return detections
    
    def _update_state(self, detections: List[dict]):
        """Update state matrix with new detections."""
        # Build update from detections
        update = np.zeros((64, 12), dtype=np.float32)
        for det in detections:
            sq = det['square']
            if sq is not None:
                update[sq, det['label']] = max(update[sq, det['label']], det['conf'])
        
        # Apply decay: new_state = decay * old + (1-decay) * update
        self.state = self.decay * self.state + (1 - self.decay) * update
    
    def _init_from_detection(self, frame: np.ndarray) -> bool:
        """Initialize state from detected pieces (should find ~32)."""
        # Run detection with lower threshold for initialization
        old_conf = self.conf_threshold
        self.conf_threshold = 0.10  # Very low for initial detection
        
        # Run detection multiple times and average
        all_detections = []
        for _ in range(10):  # More samples
            dets = self._detect(frame)
            all_detections.extend(dets)
        
        self.conf_threshold = old_conf
        
        # Build initial state
        self.state = np.zeros((64, 12), dtype=np.float32)
        for det in all_detections:
            sq = det['square']
            if sq is not None:
                self.state[sq, det['label']] = max(self.state[sq, det['label']], det['conf'])
        
        # Count detected pieces
        piece_count = 0
        for sq in range(64):
            if np.max(self.state[sq]) > 0.2:
                piece_count += 1
        
        print(f"Detected {piece_count} pieces from YOLO")
        
        # Also set state based on expected starting position
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                ptype = piece.symbol().lower()
                color = 'white' if piece.color else 'black'
                label = PIECE_TO_LABEL.get((ptype, color))
                if label is not None:
                    # Boost expected pieces
                    self.state[sq, label] = max(self.state[sq, label], 0.8)
        
        return piece_count >= 20  # At least 20 pieces detected
    
    def start(self, frame: np.ndarray):
        """Start tracking - detect initial pieces."""
        print("\nInitializing from starting position...")
        self._init_from_detection(frame)
        self.is_tracking = True
        self.board = chess.Board()
        self.moves = []
        print("\n" + "="*50)
        print("TRACKING STARTED")
        print("="*50)
    
    def _score_move(self, move: chess.Move) -> float:
        """Score how well a move matches current state."""
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Source square should be EMPTY (low max confidence)
        from_max = np.max(self.state[from_sq])
        from_empty = from_max < 0.3  # Clearly empty
        if not from_empty:
            return 0.0  # Source not vacated = not this move
        
        # Destination should have the moving piece
        piece = self.board.piece_at(from_sq)
        if not piece:
            return 0.0
        
        ptype = piece.symbol().lower()
        color = 'white' if piece.color else 'black'
        
        # Handle promotion
        if move.promotion:
            promo = {chess.QUEEN: 'q', chess.ROOK: 'r', 
                     chess.BISHOP: 'b', chess.KNIGHT: 'n'}
            ptype = promo.get(move.promotion, 'q')
        
        label = PIECE_TO_LABEL.get((ptype, color))
        if label is None:
            return 0.0
        
        to_conf = self.state[to_sq, label]
        
        # Need BOTH: source empty AND destination has piece
        score = 0.0
        score += (1.0 - from_max) * 2  # Empty source
        score += to_conf * 2  # Piece at destination
        
        # Castling
        if self.board.is_castling(move):
            castling = {
                chess.G1: (chess.H1, chess.F1, 11),
                chess.C1: (chess.A1, chess.D1, 11),
                chess.G8: (chess.H8, chess.F8, 5),
                chess.C8: (chess.A8, chess.D8, 5),
            }
            info = castling.get(move.to_square)
            if info:
                rook_from, rook_to, rook_label = info
                score += (1.0 - np.max(self.state[rook_from])) * 0.5
                score += self.state[rook_to, rook_label] * 0.5
        
        # En passant
        if self.board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
            score += (1.0 - np.max(self.state[cap_sq])) * 0.5
        
        return score
    
    def process(self, frame: np.ndarray) -> Optional[str]:
        """Process frame and detect moves."""
        if not self.is_tracking:
            return None
        
        self.frame_count += 1
        
        # Detect and update state
        detections = self._detect(frame)
        self._update_state(detections)
        
        # Cooldown after move
        if self.frame_count - self.last_move_frame < self.cooldown_frames:
            return None
        
        # Score all legal moves
        best_move = None
        best_score = 1.5  # Higher threshold - need strong evidence
        
        for move in self.board.legal_moves:
            score = self._score_move(move)
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None:
            self.pending_san = None
            self.pending_frames = 0
            return None
        
        san = self.board.san(best_move)
        
        # Confirm over multiple frames
        if san == self.pending_san:
            self.pending_frames += 1
            if self.pending_frames >= self.confirm_frames:
                return self._execute(best_move)
        else:
            self.pending_san = san
            self.pending_frames = 1
            if self.debug:
                print(f"  Pending: {san} (score={best_score:.2f})")
        
        return None
    
    def _execute(self, move: chess.Move) -> str:
        """Execute a confirmed move."""
        san = self.board.san(move)
        self.board.push(move)
        self.moves.append(san)
        self.last_move_frame = self.frame_count
        self.pending_san = None
        self.pending_frames = 0
        
        # Update state for new position
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                ptype = piece.symbol().lower()
                color = 'white' if piece.color else 'black'
                label = PIECE_TO_LABEL.get((ptype, color))
                if label is not None:
                    self.state[sq, label] = max(self.state[sq, label], 0.7)
        
        print(f"[{self.frame_count}] Move #{len(self.moves)}: {san}")
        return san
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Draw overlay on frame."""
        vis = frame.copy()
        
        # Draw board border
        if self.corners is not None:
            cv2.polylines(vis, [self.corners.astype(np.int32)], True, (0, 255, 0), 2)
        
        if self.is_tracking and self.inv_transform is not None:
            # Draw pieces
            for sq in range(64):
                piece = self.board.piece_at(sq)
                if piece:
                    f, r = sq % 8, sq // 8
                    wx = (f + 0.5) * SQUARE_SIZE
                    wy = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
                    pt = np.array([[wx, wy]], dtype=np.float32)
                    center = cv2.perspectiveTransform(pt[np.newaxis], self.inv_transform)[0, 0]
                    cx, cy = int(center[0]), int(center[1])
                    
                    ptype = piece.symbol().lower()
                    color = 'white' if piece.color else 'black'
                    label = PIECE_TO_LABEL.get((ptype, color), 0)
                    conf = self.state[sq, label]
                    
                    # Color based on confidence
                    if conf > 0.5:
                        box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                    elif conf > 0.2:
                        box_color = (0, 165, 255)  # Orange = weak
                    else:
                        box_color = (0, 0, 255)  # Red = missing
                    
                    cv2.rectangle(vis, (cx-12, cy-12), (cx+12, cy+12), box_color, 2)
                    cv2.putText(vis, piece.symbol().upper(), (cx-5, cy+4), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            
            # Debug: show raw detections
            if self.debug:
                for det in self.last_detections:
                    dcx, dcy = int(det['center'][0]), int(det['center'][1])
                    cv2.circle(vis, (dcx, dcy), 4, (0, 255, 255), -1)
        
        # Status
        turn = "White" if self.board.turn else "Black"
        cv2.putText(vis, f"Moves: {len(self.moves)} | {turn}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.moves:
            recent = self.moves[-8:]
            cv2.putText(vis, " ".join(recent), (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return vis
    
    def pgn(self, name: str) -> str:
        """Generate PGN string."""
        game = chess.pgn.Game()
        game.headers["Event"] = "ChessWorldAI Hybrid"
        game.headers["Site"] = name
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ChessWorldAI Hybrid Tracker")
    parser.add_argument('video', help='Video file')
    parser.add_argument('--output', '-o', help='Output PGN')
    parser.add_argument('--model', '-m', default='models/pieces.pt')
    parser.add_argument('--crop', type=float, default=0.45)
    parser.add_argument('--debug', '-d', action='store_true')
    
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
    
    output = args.output or f"output/pgn/{video.stem}_hybrid.pgn"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_h = int(h * args.crop)
    
    print(f"\n{'='*50}")
    print("ChessWorldAI - Hybrid Tracker")
    print(f"{'='*50}")
    print(f"Video: {video.name} ({total} frames @ {fps:.0f} FPS)")
    print(f"Model: {model_path}")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        sys.exit(1)
    frame = frame[:crop_h, :]
    
    # Initialize tracker
    tracker = HybridTracker(model_path=model_path)
    tracker.debug = args.debug
    
    # Calibrate
    corners = interactive_corners(frame)
    if corners is None:
        cap.release()
        sys.exit(0)
    tracker.calibrate(corners)
    
    # Start tracking
    tracker.start(frame)
    
    cv2.namedWindow("Hybrid Tracker", cv2.WINDOW_NORMAL)
    
    # Play at normal speed
    frame_delay = int(1000 / fps)
    
    print("\nControls: Q=quit, D=debug, SPACE=pause")
    
    paused = False
    n = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            frame = frame[:crop_h, :]
            
            # Process every frame
            tracker.process(frame)
        
        # Visualize
        vis = tracker.visualize(frame)
        
        # Progress bar
        pct = n / total if total else 0
        bar_x, bar_y = 10, vis.shape[0] - 20
        bar_w = vis.shape[1] - 20
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + int(bar_w * pct), bar_y + 10), (0, 255, 0), -1)
        
        cv2.imshow("Hybrid Tracker", vis)
        
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
    pgn_content = tracker.pgn(video.stem)
    with open(output, 'w') as f:
        f.write(pgn_content)
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Moves: {len(tracker.moves)}")
    print(f"PGN: {output}")
    print(f"\n{' '.join(tracker.moves)}")


if __name__ == "__main__":
    main()
