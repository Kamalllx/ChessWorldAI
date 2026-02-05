#!/usr/bin/env python3
"""
ChessWorldAI - Smart Hybrid Tracker
===================================
Approach:
1. ASSUME starting position (we know where 32 pieces are)
2. Run YOLO to detect pieces
3. Compare detected pieces vs expected pieces to find moves
4. Require piece to DISAPPEAR from source AND APPEAR at destination
5. Play video at normal speed

Key insight: Trust the STARTING POSITION, use detection to find CHANGES.
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

sys.path.insert(0, str(Path(__file__).parent))

from src.config import BOARD_SIZE, SQUARE_SIZE

# =============================================================================
# CONSTANTS
# =============================================================================

SQUARE_NAMES = [f"{chr(ord('a')+f)}{r+1}" for r in range(8) for f in range(8)]

# YOLO class -> (symbol, color)
YOLO_CLASS_MAP = {
    0: None,
    1: ('b', 'black'), 2: ('k', 'black'), 3: ('n', 'black'),
    4: ('p', 'black'), 5: ('q', 'black'), 6: ('r', 'black'),
    7: ('B', 'white'), 8: ('K', 'white'), 9: ('N', 'white'),
    10: ('P', 'white'), 11: ('Q', 'white'), 12: ('R', 'white'),
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


class SmartHybridTracker:
    """
    Tracker that:
    1. Trusts starting position
    2. Uses detection to find CHANGES
    3. Accumulates evidence over multiple frames
    4. Requires strong confirmation before committing
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
        
        # Expected board state: square -> (symbol, color)
        self.board_state: Dict[int, Tuple[str, str]] = {}
        
        # Detection accumulator over sliding window
        # square -> list of (was_detected, frame_num)
        self.detection_history: Dict[int, List[Tuple[bool, int]]] = defaultdict(list)
        self.window_size = 10  # Frames to consider
        
        # Chess
        self.board = chess.Board()
        self.moves: List[str] = []
        
        # State
        self.is_tracking = False
        self.frame_count = 0
        self.last_move_frame = 0
        self.cooldown = 20  # Frames between moves
        
        # Pending move
        self.pending_move: Optional[chess.Move] = None
        self.pending_count = 0
        self.confirm_threshold = 5  # Frames to confirm
        
        # Detection
        self.conf_threshold = 0.15
        self.last_detections: List[dict] = []
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
        """Start tracking from standard starting position."""
        self.board = chess.Board()
        self.moves = []
        self.board_state = {}
        self.detection_history.clear()
        
        # Initialize board state from chess.Board
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                symbol = piece.symbol()
                color = 'white' if piece.color else 'black'
                self.board_state[sq] = (symbol, color)
        
        self.is_tracking = True
        print("\n" + "="*50)
        print(f"TRACKING STARTED - {len(self.board_state)} pieces")
        print("="*50)
    
    def _detect(self, frame: np.ndarray) -> Set[int]:
        """Run detection and return set of squares with pieces."""
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        detected_squares: Set[int] = set()
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
                
                info = YOLO_CLASS_MAP.get(cls)
                if info is None:
                    continue
                
                sq = self._point_to_square(cx, cy)
                if sq is not None:
                    detected_squares.add(sq)
                    detections.append({
                        'center': (cx, cy),
                        'square': sq,
                        'symbol': info[0],
                        'color': info[1],
                        'conf': conf,
                    })
        
        self.last_detections = detections
        return detected_squares
    
    def _update_history(self, detected_squares: Set[int]):
        """Update detection history for all expected squares."""
        for sq in self.board_state.keys():
            was_detected = sq in detected_squares
            self.detection_history[sq].append((was_detected, self.frame_count))
            
            # Keep only recent history
            while len(self.detection_history[sq]) > self.window_size:
                self.detection_history[sq].pop(0)
    
    def _get_detection_rate(self, sq: int) -> float:
        """Get recent detection rate for a square (0-1)."""
        history = self.detection_history.get(sq, [])
        if not history:
            return 1.0  # Assume present if no history
        detected_count = sum(1 for d, _ in history if d)
        return detected_count / len(history)
    
    def _find_vacated_and_appeared(self, detected_squares: Set[int]) -> Tuple[Set[int], Set[int]]:
        """Find squares that appear vacated (piece left) and appeared (piece arrived)."""
        expected = set(self.board_state.keys())
        
        # Vacated: expected to have piece but detection rate is LOW
        vacated = set()
        for sq in expected:
            rate = self._get_detection_rate(sq)
            if rate < 0.3:  # Less than 30% detection = likely vacated
                vacated.add(sq)
        
        # Appeared: not expected but consistently detected
        appeared = set()
        for sq in detected_squares:
            if sq not in expected:
                # Check if this is consistent
                history = self.detection_history.get(sq, [])
                if len(history) >= 3:
                    recent_rate = sum(1 for d, _ in history[-3:] if d) / 3
                    if recent_rate > 0.5:
                        appeared.add(sq)
        
        return vacated, appeared
    
    def process(self, frame: np.ndarray) -> Optional[str]:
        """Process frame."""
        if not self.is_tracking:
            return None
        
        self.frame_count += 1
        
        # Detect
        detected_squares = self._detect(frame)
        self._update_history(detected_squares)
        
        # Cooldown
        if self.frame_count - self.last_move_frame < self.cooldown:
            return None
        
        # Find changes
        vacated, appeared = self._find_vacated_and_appeared(detected_squares)
        
        if not vacated:
            self.pending_move = None
            self.pending_count = 0
            return None
        
        # Debug
        if self.debug and vacated:
            vn = [SQUARE_NAMES[s] for s in vacated]
            an = [SQUARE_NAMES[s] for s in appeared]
            print(f"  [{self.frame_count}] Vacated: {vn}, Appeared: {an}")
        
        # Find best legal move
        best_move = None
        best_score = 0
        
        for move in self.board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            score = 0
            
            # From square must be vacated
            if from_sq in vacated:
                score += 3
            else:
                continue  # Skip moves where source isn't vacated
            
            # To square should have appeared or be in detected
            if to_sq in appeared:
                score += 3
            elif to_sq in detected_squares:
                score += 2
            
            # Castling
            if self.board.is_castling(move):
                rook_map = {
                    chess.G1: (chess.H1, chess.F1),
                    chess.C1: (chess.A1, chess.D1),
                    chess.G8: (chess.H8, chess.F8),
                    chess.C8: (chess.A8, chess.D8),
                }
                info = rook_map.get(move.to_square)
                if info:
                    if info[0] in vacated:
                        score += 2
                    if info[1] in detected_squares:
                        score += 1
            
            # En passant
            if self.board.is_en_passant(move):
                cap_sq = chess.square(chess.square_file(to_sq), 
                                     chess.square_rank(from_sq))
                if cap_sq in vacated:
                    score += 2
            
            if score > best_score:
                best_score = score
                best_move = move
        
        # Confirm move
        if best_move and best_score >= 4:
            if best_move == self.pending_move:
                self.pending_count += 1
                if self.pending_count >= self.confirm_threshold:
                    return self._execute(best_move)
            else:
                self.pending_move = best_move
                self.pending_count = 1
                if self.debug:
                    print(f"  Pending: {self.board.san(best_move)} (score={best_score})")
        else:
            self.pending_move = None
            self.pending_count = 0
        
        return None
    
    def _execute(self, move: chess.Move) -> str:
        """Execute move."""
        san = self.board.san(move)
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Update board state
        if from_sq in self.board_state:
            piece_info = self.board_state.pop(from_sq)
            
            # Handle promotion
            if move.promotion:
                promo = {chess.QUEEN: 'Q', chess.ROOK: 'R', 
                         chess.BISHOP: 'B', chess.KNIGHT: 'N'}
                new_sym = promo.get(move.promotion, 'Q')
                if piece_info[1] == 'black':
                    new_sym = new_sym.lower()
                piece_info = (new_sym, piece_info[1])
            
            # Remove captured piece
            if to_sq in self.board_state:
                del self.board_state[to_sq]
            
            self.board_state[to_sq] = piece_info
        
        # Castling rook
        if self.board.is_castling(move):
            rook_map = {
                chess.G1: (chess.H1, chess.F1),
                chess.C1: (chess.A1, chess.D1),
                chess.G8: (chess.H8, chess.F8),
                chess.C8: (chess.A8, chess.D8),
            }
            info = rook_map.get(move.to_square)
            if info and info[0] in self.board_state:
                rook = self.board_state.pop(info[0])
                self.board_state[info[1]] = rook
        
        # En passant
        if self.board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(to_sq), 
                                 chess.square_rank(from_sq))
            if cap_sq in self.board_state:
                del self.board_state[cap_sq]
        
        self.board.push(move)
        self.moves.append(san)
        self.last_move_frame = self.frame_count
        self.pending_move = None
        self.pending_count = 0
        
        # Clear history for affected squares
        self.detection_history[from_sq].clear()
        self.detection_history[to_sq].clear()
        
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
                
                # Color based on detection rate
                rate = self._get_detection_rate(sq)
                if rate > 0.6:
                    box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                elif rate > 0.3:
                    box_color = (0, 165, 255)  # Orange
                else:
                    box_color = (0, 0, 255)  # Red = likely moved
                
                cv2.rectangle(vis, (cx-12, cy-12), (cx+12, cy+12), box_color, 2)
                cv2.putText(vis, symbol.upper(), (cx-5, cy+4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            
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
        game = chess.pgn.Game()
        game.headers["Event"] = "ChessWorldAI Smart Hybrid"
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


def main():
    parser = argparse.ArgumentParser(description="ChessWorldAI Smart Hybrid")
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
    
    model_path = args.model
    if not Path(model_path).exists():
        for alt in ['models/pieces.pt', 'models/pieces_trained.pt']:
            if Path(alt).exists():
                model_path = alt
                break
    
    output = args.output or f"output/pgn/{video.stem}_smart.pgn"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_h = int(h * args.crop)
    
    print(f"\n{'='*50}")
    print("ChessWorldAI - Smart Hybrid Tracker")
    print(f"{'='*50}")
    print(f"Video: {video.name} @ {fps:.0f} FPS")
    print(f"Model: {model_path}")
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        sys.exit(1)
    frame = frame[:crop_h, :]
    
    tracker = SmartHybridTracker(model_path=model_path)
    tracker.debug = args.debug
    
    corners = interactive_corners(frame)
    if corners is None:
        cap.release()
        sys.exit(0)
    tracker.calibrate(corners)
    tracker.start()
    
    cv2.namedWindow("Smart Tracker", cv2.WINDOW_NORMAL)
    
    frame_delay = int(1000 / fps)  # Normal speed
    paused = False
    n = 0
    
    print("\nControls: Q=quit, D=debug, SPACE=pause")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            n += 1
            frame = frame[:crop_h, :]
            tracker.process(frame)
        
        vis = tracker.visualize(frame)
        
        # Progress
        pct = n / total if total else 0
        bar_x, bar_y = 10, vis.shape[0] - 20
        bar_w = vis.shape[1] - 20
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + int(bar_w * pct), bar_y + 10), (0, 255, 0), -1)
        
        cv2.imshow("Smart Tracker", vis)
        
        key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d'):
            tracker.debug = not tracker.debug
    
    cap.release()
    cv2.destroyAllWindows()
    
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
