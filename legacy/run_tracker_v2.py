#!/usr/bin/env python3
"""
ChessWorldAI - Smart Tracker V2
Improvements over V1:
1. USES PIECE TYPE DETECTION - validates moves against detected piece types
2. REDUCED LATENCY - smaller vote windows, faster confirmation
3. BETTER MOVE MATCHING - scores based on piece type + position
4. TRACKS PIECE IDENTITIES - knows which piece is where
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple
import sys
import time
import argparse
import chess
import chess.pgn
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.piece_detection import PieceDetector
from src.config import PIECE_LABELS, PIECE_COLORS, BOARD_SIZE, SQUARE_SIZE


# Constants
SQUARE_NAMES = [f"{chr(ord('a')+f)}{r+1}" for r in range(8) for f in range(8)]

# Map piece labels to chess symbols
LABEL_TO_SYMBOL = {
    'b': 'b', 'k': 'k', 'n': 'n', 'p': 'p', 'q': 'q', 'r': 'r',  # black
    'B': 'B', 'K': 'K', 'N': 'N', 'P': 'P', 'Q': 'Q', 'R': 'R',  # white
}

# Standard starting position: square -> (piece_symbol, color)
# Using chess.py square indices: a1=0, b1=1, ..., h8=63
STARTING_POSITION = {}
for sq in range(8):  # White back rank
    pieces = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    STARTING_POSITION[sq] = (pieces[sq], 'white')
for sq in range(8, 16):  # White pawns
    STARTING_POSITION[sq] = ('P', 'white')
for sq in range(48, 56):  # Black pawns  
    STARTING_POSITION[sq] = ('p', 'black')
for sq in range(56, 64):  # Black back rank
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
    """Click 4 corners: h1, a1, a8, h8."""
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


class SmartTrackerV2:
    """
    Improved tracker that uses piece TYPE detection.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.detector = PieceDetector(model_path=model_path, device=device)
        
        # Geometry
        self.corners: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.inv_transform: Optional[np.ndarray] = None
        self.square_centers = get_square_centers()
        
        # Board state: square -> (piece_symbol, color)
        self.board_state: Dict[int, Tuple[str, str]] = {}
        
        # Detection accumulator: square -> {piece_type: count}
        self.detection_votes: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.vote_window = 2  # Very fast response
        self.frame_in_window = 0
        
        # Chess engine
        self.board = chess.Board()
        self.moves: List[str] = []
        
        # Tracking
        self.is_tracking = False
        self.frame_count = 0
        self.last_move_frame = 0
        self.cooldown = 4  # Fast cooldown
        
        # Pending confirmation
        self.pending_move: Optional[chess.Move] = None
        self.pending_count = 0
        
        # Debug
        self.last_detected: Dict[int, int] = {}
        
    def calibrate(self, corners: np.ndarray):
        self.corners = corners.astype(np.float32)
        dst = np.array([
            [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE],
            [0, 0], [BOARD_SIZE, 0],
        ], dtype=np.float32)
        self.transform = cv2.getPerspectiveTransform(self.corners, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, self.corners)
        
    def start(self):
        """Start tracking from assumed starting position."""
        self.board_state = STARTING_POSITION.copy()
        self.is_tracking = True
        print("\n" + "="*50)
        print("TRACKING STARTED - 32 pieces")
        print("="*50)
        
    def process(self, frame: np.ndarray) -> Optional[str]:
        if not self.is_tracking or self.transform is None:
            return None
            
        self.frame_count += 1
        
        # Detect pieces WITH their types
        detections = self._detect_with_types(frame)
        
        # Accumulate votes per square per piece type
        for sq, piece_cls in detections.items():
            self.detection_votes[sq][piece_cls] += 1
        self.frame_in_window += 1
        
        # Analyze every vote_window frames
        if self.frame_in_window >= self.vote_window:
            result = self._analyze()
            self.detection_votes.clear()
            self.frame_in_window = 0
            return result
            
        return None
        
    def _detect_with_types(self, frame: np.ndarray) -> Dict[int, int]:
        """Detect pieces and return square -> piece_class mapping."""
        raw = self.detector.detect(frame, self.corners)
        
        result = {}
        centers = raw.get('centers', [])
        classes = raw.get('classes', [])
        
        for i, center in enumerate(centers):
            pt = np.array([[center]], dtype=np.float32)
            warped = cv2.perspectiveTransform(pt, self.transform)[0, 0]
            
            if 0 <= warped[0] < BOARD_SIZE and 0 <= warped[1] < BOARD_SIZE:
                dists = np.linalg.norm(self.square_centers - warped, axis=1)
                sq = int(np.argmin(dists))
                
                if i < len(classes):
                    piece_cls = classes[i]
                    # Keep highest confidence if multiple detections per square
                    if sq not in result:
                        result[sq] = piece_cls
                        
        self.last_detected = result
        return result
        
    def _analyze(self) -> Optional[str]:
        """Analyze votes and detect move using piece types."""
        if self.frame_count - self.last_move_frame < self.cooldown:
            return None
            
        # Get most voted piece type per square (need >20% votes - more lenient)
        threshold = self.vote_window * 0.2
        current_detection: Dict[int, int] = {}
        
        for sq, votes in self.detection_votes.items():
            if votes:
                best_cls = max(votes, key=votes.get)
                if votes[best_cls] >= threshold:
                    current_detection[sq] = best_cls
        
        # Compare current detection to our board state
        expected_squares = set(self.board_state.keys())
        detected_squares = set(current_detection.keys())
        
        vacated = expected_squares - detected_squares
        appeared = detected_squares - expected_squares
        
        if not vacated:
            return None
            
        # Debug output
        if vacated or appeared:
            vnames = sorted([SQUARE_NAMES[s] for s in vacated])[:6]
            anames = sorted([SQUARE_NAMES[s] for s in appeared])[:6]
            if len(vnames) <= 4:  # Only print if not too noisy
                print(f"[{self.frame_count}] Vacated: {vnames}, Appeared: {anames}")
        
        # Find best matching legal move using PIECE TYPE validation
        move = self._find_best_move(vacated, appeared, current_detection)
        
        if move:
            if move == self.pending_move:
                self.pending_count += 1
                if self.pending_count >= 2:
                    return self._execute(move)
            else:
                self.pending_move = move
                self.pending_count = 1
                san = self.board.san(move)
                print(f"  Pending: {san}")
                
        return None
        
    def _find_best_move(self, vacated: Set[int], appeared: Set[int], 
                        current_detection: Dict[int, int]) -> Optional[chess.Move]:
        """Find legal move that best matches observations with piece type validation."""
        if not vacated:
            return None
            
        best_score = 0
        best_move = None
        
        for move in self.board.legal_moves:
            score = 0
            from_sq = move.from_square
            to_sq = move.to_square
            
            # Check if from_square is vacated
            if from_sq in vacated:
                score += 3
                
                # PIECE TYPE VALIDATION: Check if the piece we expect to move
                # matches what's now at the destination
                if from_sq in self.board_state:
                    expected_piece, color = self.board_state[from_sq]
                    
                    # Check if to_sq has correct piece type
                    if to_sq in current_detection:
                        detected_cls = current_detection[to_sq]
                        detected_label = PIECE_LABELS[detected_cls] if detected_cls < len(PIECE_LABELS) else None
                        
                        if detected_label:
                            # Match piece type (case insensitive for type, case sensitive for color)
                            if detected_label.upper() == expected_piece.upper():
                                if (detected_label.isupper() and color == 'white') or \
                                   (detected_label.islower() and color == 'black'):
                                    score += 4  # Strong match!
                                else:
                                    score += 1  # Type matches but wrong color detected
            
            # To square appeared (for non-captures)
            if to_sq in appeared:
                score += 2
            elif to_sq in current_detection:  # Square still has a piece (capture or same)
                score += 1
                
            # Castling validation
            if self.board.is_castling(move):
                rook_from, rook_to = self._get_castling_rook(move)
                if rook_from and rook_from in vacated:
                    score += 2
                if rook_to and rook_to in current_detection:
                    score += 1
                    
            if score > best_score:
                best_score = score
                best_move = move
                
        # Lower threshold for faster detection
        return best_move if best_score >= 3 else None
        
    def _get_castling_rook(self, move: chess.Move) -> Tuple[Optional[int], Optional[int]]:
        """Get rook from/to squares for castling."""
        rook_map = {
            chess.G1: (chess.H1, chess.F1),
            chess.C1: (chess.A1, chess.D1),
            chess.G8: (chess.H8, chess.F8),
            chess.C8: (chess.A8, chess.D8),
        }
        return rook_map.get(move.to_square, (None, None))
        
    def _execute(self, move: chess.Move) -> str:
        """Execute confirmed move."""
        san = self.board.san(move)
        print(f"  ==> MOVE: {san}")
        
        from_sq, to_sq = move.from_square, move.to_square
        
        # Update board state
        if from_sq in self.board_state:
            piece, color = self.board_state.pop(from_sq)
            
            # Handle promotion
            if move.promotion:
                promo_symbols = {chess.QUEEN: 'Q', chess.ROOK: 'R', 
                                chess.BISHOP: 'B', chess.KNIGHT: 'N'}
                new_piece = promo_symbols.get(move.promotion, 'Q')
                if color == 'black':
                    new_piece = new_piece.lower()
                piece = new_piece
                
            # Remove captured piece
            if to_sq in self.board_state:
                del self.board_state[to_sq]
                
            self.board_state[to_sq] = (piece, color)
            
        # Castling rook
        if self.board.is_castling(move):
            rook_from, rook_to = self._get_castling_rook(move)
            if rook_from and rook_from in self.board_state:
                rook = self.board_state.pop(rook_from)
                self.board_state[rook_to] = rook
                
        # En passant
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
            # Draw expected pieces
            for sq, (piece, color) in self.board_state.items():
                f, r = sq % 8, sq // 8
                wx = (f + 0.5) * SQUARE_SIZE
                wy = BOARD_SIZE - (r + 0.5) * SQUARE_SIZE
                pt = np.array([[wx, wy]], dtype=np.float32)
                center = cv2.perspectiveTransform(pt[np.newaxis], self.inv_transform)[0, 0]
                cx, cy = int(center[0]), int(center[1])
                
                # Color based on piece color
                box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                
                # Highlight if this square wasn't detected
                if sq not in self.last_detected:
                    box_color = (0, 0, 255)  # Red = not detected
                    
                cv2.rectangle(vis, (cx-18, cy-18), (cx+18, cy+18), box_color, 2)
                cv2.putText(vis, piece.upper(), (cx-6, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                           
        # Status
        status = f"Moves: {len(self.moves)}" if self.is_tracking else "Press SPACE"
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if self.moves:
            # Show last 10 moves
            recent = self.moves[-10:]
            move_str = " ".join(f"{i+1}.{m}" if (len(self.moves)-10+i) % 2 == 0 else m 
                               for i, m in enumerate(recent))
            cv2.putText(vis, move_str, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                       
        return vis
        
    def pgn(self, name: str) -> str:
        game = chess.pgn.Game()
        game.headers["Event"] = "ChessWorldAI"
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
    parser = argparse.ArgumentParser(description="ChessWorldAI Smart Tracker V2")
    parser.add_argument('video', help='Video file')
    parser.add_argument('--output', '-o', help='Output PGN')
    parser.add_argument('--model', '-m', default='models/pieces_trained.pt')
    parser.add_argument('--speed', '-s', type=float, default=1.0)
    parser.add_argument('--crop', type=float, default=0.45)
    
    args = parser.parse_args()
    
    video = Path(args.video)
    if not video.exists():
        print(f"Error: {video} not found")
        sys.exit(1)
        
    output = args.output or f"output/pgn/{video.stem}.pgn"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    model = str(Path(args.model)) if Path(args.model).exists() else None
    
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_h = int(h * args.crop)
    
    print(f"\nVideo: {video.name} ({w}x{h} -> {w}x{crop_h})")
    print("V2 Tracker: Using piece TYPE detection for better accuracy")
    
    ret, frame = cap.read()
    if not ret:
        sys.exit(1)
    frame = frame[:crop_h, :]
    
    tracker = SmartTrackerV2(model_path=model)
    
    corners = interactive_corners(frame)
    if corners is None:
        cap.release()
        sys.exit(0)
    tracker.calibrate(corners)
    
    delay = max(1, int(500 / fps / args.speed))  # Faster display
    sample_rate = max(1, int(fps / 20))  # Process more frames
    
    print("\nPress SPACE to start, Q to quit, +/- to change speed")
    
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
                           
            cv2.imshow("ChessWorldAI V2", vis)
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    pgn_str = tracker.pgn(video.stem)
    
    with open(output, 'w') as f:
        f.write(pgn_str)
        
    print("\n" + "="*50)
    print(f"Detected {len(tracker.moves)} moves")
    print(f"Moves: {' '.join(tracker.moves)}")
    print(f"\nSaved: {output}")
    print(pgn_str)


if __name__ == '__main__':
    main()
