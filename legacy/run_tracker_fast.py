#!/usr/bin/env python3
"""
ChessWorldAI - Fast Tracker
===========================
Optimized for SPEED + ACCURACY:
1. Direct YOLO inference (no wrapper overhead)
2. Multi-threaded frame reading (non-blocking I/O)
3. State-based tracking with decay (CameraChessWeb style)
4. Skip frames intelligently
5. GPU-accelerated inference

Usage:
    python run_tracker_fast.py videos/game_4.mp4 --speed 3.0
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
from threading import Thread
from queue import Queue
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BOARD_SIZE, SQUARE_SIZE

# =============================================================================
# CONSTANTS
# =============================================================================

SQUARE_NAMES = [f"{chr(ord('a')+f)}{r+1}" for r in range(8) for f in range(8)]

# YOLO class -> (symbol, color, label_index for 12-class state)
YOLO_CLASS_MAP = {
    0: None,  # generic bishop - skip
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

CASTLING_MAP = {
    chess.G1: (chess.H1, chess.F1, 11),
    chess.C1: (chess.A1, chess.D1, 11),
    chess.G8: (chess.H8, chess.F8, 5),
    chess.C8: (chess.A8, chess.D8, 5),
}


def get_square_centers() -> np.ndarray:
    centers = []
    for sq in range(64):
        f, r = sq % 8, sq // 8
        x = (f + 0.5) * SQUARE_SIZE
        y = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
        centers.append([x, y])
    return np.array(centers)


# =============================================================================
# THREADED VIDEO READER (for speed)
# =============================================================================

class ThreadedVideoReader:
    """Read frames in background thread for faster processing."""
    
    def __init__(self, video_path: str, crop_h: int, queue_size: int = 128):
        self.cap = cv2.VideoCapture(video_path)
        self.crop_h = crop_h
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.frame_count = 0
        
    def start(self):
        Thread(target=self._read, daemon=True).start()
        return self
        
    def _read(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.frame_count += 1
                frame = frame[:self.crop_h, :]
                self.queue.put((self.frame_count, frame))
            else:
                time.sleep(0.001)
                
    def read(self):
        if self.stopped and self.queue.empty():
            return None, None
        return self.queue.get()
        
    def stop(self):
        self.stopped = True
        self.cap.release()


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
                
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", mouse_cb)
    
    print("\nClick corners: h1 -> a1 -> a8 -> h8")
    print("Press 'c' to confirm, 'r' to reset, 'q' to quit")
    
    while True:
        temp = display.copy()
        if len(corners) < 4:
            cv2.putText(temp, f"Click: {names[len(corners)]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(temp, "Press 'c' to confirm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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


# =============================================================================
# FAST TRACKER
# =============================================================================

class FastTracker:
    """
    Optimized tracker combining:
    - Direct YOLO inference
    - State-based tracking with decay
    - Move scoring from CameraChessWeb
    """
    
    def __init__(self, model_path: str = 'models/pieces.pt'):
        from ultralytics import YOLO
        
        # Load model with optimizations
        self.model = YOLO(model_path)
        self.model.fuse()  # Fuse layers for speed
        print(f"✓ Model: {model_path}")
        
        # Geometry
        self.corners: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.inv_transform: Optional[np.ndarray] = None
        self.square_centers = get_square_centers()
        self.board_polygon: Optional[np.ndarray] = None
        
        # State matrix: 64 squares x 12 classes
        self.state = np.zeros((64, 12), dtype=np.float32)
        self.decay = 0.6  # Faster decay for quicker response
        
        # Thresholds (lower = more sensitive)
        self.from_thr = 0.4
        self.to_thr = 0.4
        self.conf_threshold = 0.15
        
        # Chess
        self.board = chess.Board()
        self.moves: List[str] = []
        
        # Tracking
        self.is_tracking = False
        self.frame_count = 0
        self.last_move_frame = 0
        self.cooldown = 3  # Very short cooldown
        
        # Greedy confirmation
        self.greedy_times: Dict[str, float] = {}
        self.greedy_timeout = 0.8  # seconds
        
        # Debug
        self.debug = False
        self.last_detections: List[dict] = []
        self.process_times: List[float] = []
        
    def calibrate(self, corners: np.ndarray):
        self.corners = corners.astype(np.float32)
        dst = np.array([
            [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE],
            [0, 0], [BOARD_SIZE, 0],
        ], dtype=np.float32)
        self.transform = cv2.getPerspectiveTransform(self.corners, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, self.corners)
        self.board_polygon = self.corners.reshape(-1, 1, 2).astype(np.int32)
        
    def start(self):
        self.is_tracking = True
        self.state = np.zeros((64, 12), dtype=np.float32)
        self.board = chess.Board()
        self.moves = []
        self.greedy_times = {}
        
        # Initialize state with starting position
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                ptype = piece.symbol().lower()
                color = 'white' if piece.color else 'black'
                label = PIECE_TO_LABEL.get((ptype, color))
                if label is not None:
                    self.state[sq, label] = 1.0
        
        print("\n" + "="*50)
        print("FAST TRACKER STARTED")
        print("="*50)
        
    def _detect(self, frame: np.ndarray) -> List[dict]:
        """Fast YOLO detection."""
        # Run inference with half precision if GPU available
        results = self.model.predict(
            frame, 
            verbose=False, 
            conf=self.conf_threshold,
            half=True,  # FP16 for speed
            imgsz=640,  # Smaller input = faster
        )
        
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                
                # Skip if outside board
                if self.board_polygon is not None:
                    if cv2.pointPolygonTest(self.board_polygon, (cx, cy), False) < 0:
                        continue
                
                info = YOLO_CLASS_MAP.get(cls)
                if info is None:
                    continue
                    
                symbol, color, label = info
                detections.append({
                    'center': (cx, cy),
                    'label': label,
                    'conf': conf,
                    'symbol': symbol,
                })
        
        self.last_detections = detections
        return detections
    
    def _point_to_square(self, point: Tuple[float, float]) -> Optional[int]:
        pt = np.array([[point]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.transform)[0, 0]
        if 0 <= warped[0] < BOARD_SIZE and 0 <= warped[1] < BOARD_SIZE:
            dists = np.linalg.norm(self.square_centers - warped, axis=1)
            return int(np.argmin(dists))
        return None
    
    def _update_state(self, detections: List[dict]):
        """Update state with decay."""
        # Build update matrix
        update = np.zeros((64, 12), dtype=np.float32)
        
        for det in detections:
            sq = self._point_to_square(det['center'])
            if sq is not None:
                update[sq, det['label']] = max(update[sq, det['label']], det['conf'])
        
        # Apply decay
        self.state = self.decay * self.state + (1 - self.decay) * update
    
    def _score_move(self, move: chess.Move) -> float:
        """Score how well a move matches current state."""
        score = 0.0
        
        from_sq = move.from_square
        to_sq = move.to_square
        
        # From square should be empty (low max score)
        from_max = np.max(self.state[from_sq])
        score += (1.0 - from_max - self.from_thr)
        
        # Get expected piece at destination
        piece = self.board.piece_at(from_sq)
        if piece:
            ptype = piece.symbol().lower()
            color = 'white' if piece.color else 'black'
            
            # Handle promotion
            if move.promotion:
                promo = {chess.QUEEN: 'q', chess.ROOK: 'r', 
                         chess.BISHOP: 'b', chess.KNIGHT: 'n'}
                ptype = promo.get(move.promotion, 'q')
            
            label = PIECE_TO_LABEL.get((ptype, color))
            if label is not None:
                to_score = self.state[to_sq, label]
                score += (to_score - self.to_thr)
        
        # Castling bonus
        if self.board.is_castling(move):
            info = CASTLING_MAP.get(move.to_square)
            if info:
                rook_from, rook_to, rook_label = info
                # Rook source should be empty
                score += (1.0 - np.max(self.state[rook_from]) - self.from_thr) * 0.5
                # Rook at destination
                score += (self.state[rook_to, rook_label] - self.to_thr) * 0.5
        
        # En passant
        if self.board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
            score += (1.0 - np.max(self.state[cap_sq]) - self.from_thr) * 0.5
        
        return score
    
    def process(self, frame: np.ndarray, current_time: float) -> Optional[str]:
        if not self.is_tracking or self.transform is None:
            return None
        
        t0 = time.perf_counter()
        self.frame_count += 1
        
        # Detect and update state
        detections = self._detect(frame)
        self._update_state(detections)
        
        # Skip if in cooldown
        if self.frame_count - self.last_move_frame < self.cooldown:
            self.process_times.append(time.perf_counter() - t0)
            return None
        
        # Score all legal moves
        best_move = None
        best_score = -float('inf')
        
        for move in self.board.legal_moves:
            score = self._score_move(move)
            if score > best_score:
                best_score = score
                best_move = move
        
        # Check for move confirmation
        result = None
        if best_move and best_score > 0:
            san = self.board.san(best_move)
            
            if san not in self.greedy_times:
                self.greedy_times[san] = current_time
            
            elapsed = current_time - self.greedy_times[san]
            if elapsed > self.greedy_timeout:
                result = self._execute(best_move)
                self.greedy_times = {}
        else:
            # Reset if no good move
            self.greedy_times = {}
        
        self.process_times.append(time.perf_counter() - t0)
        return result
    
    def _execute(self, move: chess.Move) -> str:
        san = self.board.san(move)
        self.board.push(move)
        self.moves.append(san)
        self.last_move_frame = self.frame_count
        
        # Update state for new position
        for sq in range(64):
            piece = self.board.piece_at(sq)
            if piece:
                ptype = piece.symbol().lower()
                color = 'white' if piece.color else 'black'
                label = PIECE_TO_LABEL.get((ptype, color))
                if label is not None:
                    # Boost expected pieces
                    self.state[sq, label] = max(self.state[sq, label], 0.7)
        
        print(f"[Frame {self.frame_count}] Move #{len(self.moves)}: {san}")
        return san
    
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        
        if self.corners is not None:
            cv2.polylines(vis, [self.corners.astype(np.int32)], True, (0, 255, 0), 2)
        
        if self.is_tracking and self.inv_transform is not None:
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
                    
                    if conf > 0.5:
                        box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                    elif conf > 0.2:
                        box_color = (0, 165, 255) if color == 'white' else (180, 100, 180)
                    else:
                        box_color = (0, 0, 255)
                    
                    cv2.rectangle(vis, (cx-14, cy-14), (cx+14, cy+14), box_color, 2)
                    cv2.putText(vis, piece.symbol().upper(), (cx-5, cy+4), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            
            if self.debug:
                for det in self.last_detections:
                    dcx, dcy = int(det['center'][0]), int(det['center'][1])
                    cv2.circle(vis, (dcx, dcy), 3, (0, 255, 255), -1)
        
        # FPS counter
        if self.process_times:
            avg_time = np.mean(self.process_times[-30:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(vis, f"FPS: {fps:.0f}", (vis.shape[1]-100, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Status
        turn = "W" if self.board.turn else "B"
        cv2.putText(vis, f"Moves: {len(self.moves)} ({turn})", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.moves:
            recent = self.moves[-8:]
            cv2.putText(vis, " ".join(recent), (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return vis
    
    def pgn(self, name: str) -> str:
        game = chess.pgn.Game()
        game.headers["Event"] = "ChessWorldAI Fast"
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
    parser = argparse.ArgumentParser(description="ChessWorldAI Fast Tracker")
    parser.add_argument('video', help='Video file')
    parser.add_argument('--output', '-o', help='Output PGN')
    parser.add_argument('--model', '-m', default='models/pieces.pt')
    parser.add_argument('--speed', '-s', type=float, default=1.0)
    parser.add_argument('--crop', type=float, default=0.45)
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--skip', type=int, default=2, help='Process every N frames')
    
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
    
    output = args.output or f"output/pgn/{video.stem}_fast.pgn"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    # Get video info
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_h = int(h * args.crop)
    
    print(f"\n{'='*50}")
    print("ChessWorldAI - Fast Tracker")
    print(f"{'='*50}")
    print(f"Video: {video.name} ({total} frames @ {fps:.1f} FPS)")
    print(f"Model: {model_path}")
    print(f"Skip: {args.skip} (process every {args.skip} frames)")
    
    # Read first frame for calibration
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        sys.exit(1)
    frame = frame[:crop_h, :]
    cap.release()
    
    # Initialize tracker
    tracker = FastTracker(model_path=model_path)
    tracker.debug = args.debug
    
    # Calibrate
    corners = interactive_corners(frame)
    if corners is None:
        print("Cancelled")
        sys.exit(0)
    tracker.calibrate(corners)
    tracker.start()
    
    # Start threaded reader
    reader = ThreadedVideoReader(str(video), crop_h)
    reader.start()
    
    cv2.namedWindow("FastTracker", cv2.WINDOW_NORMAL)
    
    frame_delay = max(1, int((1000 / fps) / args.speed))
    start_time = time.time()
    last_vis = frame
    n = 0
    
    print("\nControls: Q=quit, D=debug, +/-=speed")
    
    while True:
        result = reader.read()
        if result[0] is None:
            break
        
        n, frame = result
        
        # Process every N frames
        if n % args.skip == 0:
            current_time = time.time() - start_time
            tracker.process(frame, current_time)
            last_vis = frame
        
        # Visualize
        vis = tracker.visualize(last_vis)
        
        # Progress
        pct = 100 * n / total if total else 0
        bar_w = int(vis.shape[1] * 0.7)
        bar_x = int(vis.shape[1] * 0.15)
        bar_y = vis.shape[0] - 15
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (50, 50, 50), -1)
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + int(bar_w * pct / 100), bar_y + 8), (0, 255, 0), -1)
        cv2.putText(vis, f"{pct:.0f}%", (bar_x + bar_w + 5, bar_y + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow("FastTracker", vis)
        
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            tracker.debug = not tracker.debug
        elif key in (ord('+'), ord('=')):
            args.speed = min(10, args.speed * 1.5)
            frame_delay = max(1, int((1000 / fps) / args.speed))
            print(f"Speed: {args.speed:.1f}x")
        elif key == ord('-'):
            args.speed = max(0.25, args.speed / 1.5)
            frame_delay = max(1, int((1000 / fps) / args.speed))
            print(f"Speed: {args.speed:.1f}x")
    
    reader.stop()
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
