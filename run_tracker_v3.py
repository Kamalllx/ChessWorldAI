#!/usr/bin/env python3
"""
ChessWorldAI - Smart Tracker V3
Improvements:
1. Direct YOLO model loading (no wrapper)
2. Better confidence thresholds
3. Improved piece type matching
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple
import sys
import argparse
import chess
import chess.pgn
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.config import BOARD_SIZE, SQUARE_SIZE


SQUARE_NAMES = [f"{chr(ord('a')+f)}{r+1}" for r in range(8) for f in range(8)]

# Class mapping for pieces.pt model
CLASS_TO_PIECE = {
    0: None,     # generic bishop
    1: ('b', 'black', 'bishop'),
    2: ('k', 'black', 'king'),
    3: ('n', 'black', 'knight'),
    4: ('p', 'black', 'pawn'),
    5: ('q', 'black', 'queen'),
    6: ('r', 'black', 'rook'),
    7: ('B', 'white', 'bishop'),
    8: ('K', 'white', 'king'),
    9: ('N', 'white', 'knight'),
    10: ('P', 'white', 'pawn'),
    11: ('Q', 'white', 'queen'),
    12: ('R', 'white', 'rook'),
}

STARTING_POSITION = {}
for sq in range(8):
    pieces = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    STARTING_POSITION[sq] = (pieces[sq], 'white')
for sq in range(8, 16):
    STARTING_POSITION[sq] = ('P', 'white')
for sq in range(48, 56):
    STARTING_POSITION[sq] = ('p', 'black')
for sq in range(56, 64):
    pieces = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
    STARTING_POSITION[sq] = (pieces[sq-56], 'black')


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
                
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", mouse_cb)
    
    print("\nClick 4 corners: h1 -> a1 -> a8 -> h8")
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
    return None


class SmartTrackerV3:
    def __init__(self, model_path: str = 'models/pieces.pt'):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        print(f"✓ Loaded model: {model_path}")
        
        self.corners: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.inv_transform: Optional[np.ndarray] = None
        self.square_centers = get_square_centers()
        
        self.board_state: Dict[int, Tuple[str, str]] = {}
        self.detection_votes: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.vote_window = 5
        self.frame_in_window = 0
        
        self.board = chess.Board()
        self.moves: List[str] = []
        
        self.is_tracking = False
        self.frame_count = 0
        self.last_move_frame = 0
        self.cooldown = 8
        
        self.pending_move: Optional[chess.Move] = None
        self.pending_count = 0
        
        self.last_detected: Dict[int, Tuple[int, float]] = {}
        self.debug_mode = False
        
    def calibrate(self, corners: np.ndarray):
        self.corners = corners.astype(np.float32)
        dst = np.array([
            [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE],
            [0, 0], [BOARD_SIZE, 0],
        ], dtype=np.float32)
        self.transform = cv2.getPerspectiveTransform(self.corners, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, self.corners)
        
    def start(self):
        self.board_state = STARTING_POSITION.copy()
        self.is_tracking = True
        print("\n" + "="*50)
        print("TRACKING STARTED")
        print("="*50)
        
    def _detect(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, verbose=False, conf=0.3)
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
                    if cls != 0:  # Skip generic
                        detections.append({
                            'center': (cx, cy),
                            'class': cls,
                            'confidence': conf
                        })
        return detections
        
    def _point_to_square(self, point: Tuple[float, float]) -> Optional[int]:
        pt = np.array([[point]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.transform)[0, 0]
        if 0 <= warped[0] < BOARD_SIZE and 0 <= warped[1] < BOARD_SIZE:
            dists = np.linalg.norm(self.square_centers - warped, axis=1)
            return int(np.argmin(dists))
        return None
        
    def process(self, frame: np.ndarray) -> Optional[str]:
        if not self.is_tracking or self.transform is None:
            return None
            
        self.frame_count += 1
        detections = self._detect(frame)
        
        for det in detections:
            sq = self._point_to_square(det['center'])
            if sq is not None:
                self.detection_votes[sq].append((det['class'], det['confidence']))
        
        self.frame_in_window += 1
        
        if self.frame_in_window >= self.vote_window:
            result = self._analyze()
            self.detection_votes.clear()
            self.frame_in_window = 0
            return result
        return None
        
    def _analyze(self) -> Optional[str]:
        if self.frame_count - self.last_move_frame < self.cooldown:
            return None
            
        current_detection: Dict[int, Tuple[int, float]] = {}
        
        for sq, votes in self.detection_votes.items():
            if not votes:
                continue
            class_scores = defaultdict(float)
            class_counts = defaultdict(int)
            for cls, conf in votes:
                class_scores[cls] += conf
                class_counts[cls] += 1
            best_cls = max(class_scores.keys(), key=lambda c: class_scores[c])
            avg_conf = class_scores[best_cls] / class_counts[best_cls]
            vote_ratio = class_counts[best_cls] / len(votes)
            if vote_ratio >= 0.4 and avg_conf >= 0.3:
                current_detection[sq] = (best_cls, avg_conf)
                
        self.last_detected = current_detection
        
        expected_squares = set(self.board_state.keys())
        detected_squares = set(current_detection.keys())
        
        vacated = expected_squares - detected_squares
        appeared = detected_squares - expected_squares
        
        if not vacated:
            return None
            
        if len(vacated) <= 4 and self.debug_mode:
            vnames = sorted([SQUARE_NAMES[s] for s in vacated])
            anames = sorted([SQUARE_NAMES[s] for s in appeared])
            print(f"[{self.frame_count}] Vacated: {vnames}, Appeared: {anames}")
        
        move = self._find_best_move(vacated, appeared, current_detection)
        
        if move:
            if move == self.pending_move:
                self.pending_count += 1
                if self.pending_count >= 2:
                    return self._execute(move)
            else:
                self.pending_move = move
                self.pending_count = 1
                print(f"  Pending: {self.board.san(move)}")
        return None
        
    def _find_best_move(self, vacated: Set[int], appeared: Set[int],
                        current_detection: Dict[int, Tuple[int, float]]) -> Optional[chess.Move]:
        best_score = 0
        best_move = None
        
        for move in self.board.legal_moves:
            score = self._score_move(move, vacated, appeared, current_detection)
            if score > best_score:
                best_score = score
                best_move = move
                
        min_score = 4
        if best_move and self.board.is_castling(best_move):
            min_score = 3
        return best_move if best_score >= min_score else None
        
    def _score_move(self, move: chess.Move, vacated: Set[int], appeared: Set[int],
                    current_detection: Dict[int, Tuple[int, float]]) -> int:
        score = 0
        from_sq = move.from_square
        to_sq = move.to_square
        
        if from_sq in vacated:
            score += 3
            
        if from_sq in self.board_state:
            expected_symbol, expected_color = self.board_state[from_sq]
            if to_sq in current_detection:
                detected_cls, det_conf = current_detection[to_sq]
                piece_info = CLASS_TO_PIECE.get(detected_cls)
                if piece_info:
                    if piece_info[1] == expected_color:
                        score += 3
                        if piece_info[0].upper() == expected_symbol.upper():
                            score += 2
                    else:
                        score -= 2
            elif to_sq in appeared:
                score += 1
                
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
                if info[1] in current_detection:
                    score += 1
                    
        if self.board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
            if cap_sq in vacated:
                score += 2
                
        return score
        
    def _execute(self, move: chess.Move) -> str:
        san = self.board.san(move)
        print(f"  ==> MOVE: {san}")
        
        from_sq, to_sq = move.from_square, move.to_square
        
        if from_sq in self.board_state:
            piece, color = self.board_state.pop(from_sq)
            if move.promotion:
                promo_symbols = {chess.QUEEN: 'Q', chess.ROOK: 'R',
                                chess.BISHOP: 'B', chess.KNIGHT: 'N'}
                new_piece = promo_symbols.get(move.promotion, 'Q')
                if color == 'black':
                    new_piece = new_piece.lower()
                piece = new_piece
            if to_sq in self.board_state:
                del self.board_state[to_sq]
            self.board_state[to_sq] = (piece, color)
            
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
                
        if self.board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
            if cap_sq in self.board_state:
                del self.board_state[cap_sq]
                
        self.board.push(move)
        self.moves.append(san)
        self.last_move_frame = self.frame_count
        self.pending_move = None
        self.pending_count = 0
        return san
        
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        vis = frame.copy()
        if self.corners is not None:
            cv2.polylines(vis, [self.corners.astype(np.int32)], True, (0, 255, 0), 2)
        if self.is_tracking and self.inv_transform is not None:
            for sq, (piece, color) in self.board_state.items():
                f, r = sq % 8, sq // 8
                wx = (f + 0.5) * SQUARE_SIZE
                wy = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
                pt = np.array([[wx, wy]], dtype=np.float32)
                center = cv2.perspectiveTransform(pt[np.newaxis], self.inv_transform)[0, 0]
                cx, cy = int(center[0]), int(center[1])
                box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                if sq not in self.last_detected:
                    box_color = (0, 0, 255)
                cv2.rectangle(vis, (cx-16, cy-16), (cx+16, cy+16), box_color, 2)
                cv2.putText(vis, piece.upper(), (cx-6, cy+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        status = f"Moves: {len(self.moves)}" if self.is_tracking else "Press SPACE"
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if self.moves:
            move_str = " ".join(self.moves[-10:])
            cv2.putText(vis, move_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return vis
        
    def pgn(self, name: str) -> str:
        game = chess.pgn.Game()
        game.headers["Event"] = "ChessWorldAI V3"
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
            game.headers["Result"] = "1-0" if board.turn == chess.BLACK else "0-1"
        else:
            game.headers["Result"] = "*"
        return str(game)


def main():
    parser = argparse.ArgumentParser(description="ChessWorldAI V3 Tracker")
    parser.add_argument('video', help='Video file')
    parser.add_argument('--output', '-o', help='Output PGN')
    parser.add_argument('--model', '-m', default='models/pieces.pt')
    parser.add_argument('--speed', '-s', type=float, default=1.0)
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
                
    output = args.output or f"output/pgn/{video.stem}_v3.pgn"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_h = int(h * args.crop)
    
    print(f"\nVideo: {video.name}")
    print(f"Model: {model_path}")
    
    ret, frame = cap.read()
    if not ret:
        sys.exit(1)
    frame = frame[:crop_h, :]
    
    tracker = SmartTrackerV3(model_path=model_path)
    tracker.debug_mode = args.debug
    
    corners = interactive_corners(frame)
    if corners is None:
        cap.release()
        sys.exit(0)
    tracker.calibrate(corners)
    
    delay = max(1, int(1000 / fps / args.speed))
    sample_rate = max(1, int(fps / 15))
    
    paused = True
    n = 0
    
    try:
        while True:
            key = cv2.waitKey(delay if not paused else 30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not tracker.is_tracking:
                    tracker.start()
                    paused = False
                else:
                    paused = not paused
            elif key == ord('d'):
                tracker.debug_mode = not tracker.debug_mode
            elif key in (ord('+'), ord('=')):
                args.speed = min(10, args.speed * 1.5)
                delay = max(1, int(1000 / fps / args.speed))
            elif key == ord('-'):
                args.speed = max(0.1, args.speed / 1.5)
                delay = max(1, int(1000 / fps / args.speed))
                
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                n += 1
                frame = frame[:crop_h, :]
                if n % sample_rate == 0:
                    move = tracker.process(frame)
                    if move:
                        print(f"[{n}] #{len(tracker.moves)}: {move}")
                        
            vis = tracker.visualize(frame)
            pct = 100 * n / total if total else 0
            cv2.putText(vis, f"{n}/{total} ({pct:.0f}%) Speed:{args.speed:.1f}x",
                       (10, crop_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if paused and not tracker.is_tracking:
                cv2.putText(vis, "PRESS SPACE TO START", (w//2-130, crop_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            elif paused:
                cv2.putText(vis, "PAUSED", (w//2-50, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("ChessWorldAI V3", vis)
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    pgn_str = tracker.pgn(video.stem)
    with open(output, 'w') as f:
        f.write(pgn_str)
    print(f"\nDetected {len(tracker.moves)} moves")
    print(f"Saved: {output}")
    print(pgn_str)


if __name__ == '__main__':
    main()
