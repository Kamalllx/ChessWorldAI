#!/usr/bin/env python3
"""
ChessWorldAI - Assume & Track Processor

KEY INSIGHT: We KNOW it's a standard chess game starting from move 1.
So we ASSUME the starting position and immediately start tracking.

Flow:
1. CALIBRATION: Click 4 corners, then press SPACE when board is in starting position
2. LOCK: Immediately lock standard starting position
3. TRACK: Watch for occupancy changes to detect moves

No detection needed for initialization - we just need accurate corner calibration.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Set
import sys
import time
import argparse
import chess
import chess.pgn
from datetime import datetime
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.piece_detection import PieceDetector
from src.config import (
    PIECE_LABELS, PIECE_COLORS, START_FEN, 
    BOARD_SIZE, SQUARE_SIZE
)


# ============================================================================
# CONSTANTS
# ============================================================================

PIECE_TO_INDEX = {label: idx for idx, label in enumerate(PIECE_LABELS)}

# Standard starting position
STARTING_PIECES = {
    # White back rank
    0: PIECE_TO_INDEX['R'], 1: PIECE_TO_INDEX['N'], 2: PIECE_TO_INDEX['B'], 3: PIECE_TO_INDEX['Q'],
    4: PIECE_TO_INDEX['K'], 5: PIECE_TO_INDEX['B'], 6: PIECE_TO_INDEX['N'], 7: PIECE_TO_INDEX['R'],
    # White pawns
    8: PIECE_TO_INDEX['P'], 9: PIECE_TO_INDEX['P'], 10: PIECE_TO_INDEX['P'], 11: PIECE_TO_INDEX['P'],
    12: PIECE_TO_INDEX['P'], 13: PIECE_TO_INDEX['P'], 14: PIECE_TO_INDEX['P'], 15: PIECE_TO_INDEX['P'],
    # Black pawns
    48: PIECE_TO_INDEX['p'], 49: PIECE_TO_INDEX['p'], 50: PIECE_TO_INDEX['p'], 51: PIECE_TO_INDEX['p'],
    52: PIECE_TO_INDEX['p'], 53: PIECE_TO_INDEX['p'], 54: PIECE_TO_INDEX['p'], 55: PIECE_TO_INDEX['p'],
    # Black back rank
    56: PIECE_TO_INDEX['r'], 57: PIECE_TO_INDEX['n'], 58: PIECE_TO_INDEX['b'], 59: PIECE_TO_INDEX['q'],
    60: PIECE_TO_INDEX['k'], 61: PIECE_TO_INDEX['b'], 62: PIECE_TO_INDEX['n'], 63: PIECE_TO_INDEX['r'],
}

SQUARE_NAMES = [f"{chr(ord('a')+f)}{r+1}" for r in range(8) for f in range(8)]


# ============================================================================
# GEOMETRY
# ============================================================================

def get_perspective_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def warp_point(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    pt = np.array([[point]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, matrix)[0, 0]


def get_square_centers() -> np.ndarray:
    centers = []
    for sq in range(64):
        file, rank = sq % 8, sq // 8
        x = (file + 0.5) * SQUARE_SIZE
        y = BOARD_SIZE - (rank + 0.5) * SQUARE_SIZE
        centers.append([x, y])
    return np.array(centers)


# ============================================================================
# CALIBRATION
# ============================================================================

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
    
    print("\n" + "="*60)
    print("STEP 1: Click 4 corners in order: h1 -> a1 -> a8 -> h8")
    print("STEP 2: Press 'c' to confirm when all 4 corners clicked")
    print("Press 'r' to reset, 'q' to quit")
    print("="*60)
    
    while True:
        temp = display.copy()
        if len(corners) < 4:
            cv2.putText(temp, f"Click corner: {names[len(corners)]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(temp, "Press 'c' to confirm corners", (10, 30),
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


# ============================================================================
# ASSUME & TRACK PROCESSOR
# ============================================================================

class AssumeTrackProcessor:
    """
    Assumes starting position, then tracks occupancy changes.
    Uses differential tracking - compares against a STABLE baseline that updates after each move.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.detector = PieceDetector(model_path=model_path, device=device)
        
        # Board geometry
        self.corners: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.inv_transform: Optional[np.ndarray] = None
        self.square_centers = get_square_centers()
        
        # Locked pieces: square -> piece_class
        self.locked_pieces: Dict[int, int] = {}
        
        # STABLE occupancy - what we expect based on locked pieces
        # Updated after each confirmed move
        self.stable_occupancy: Set[int] = set()
        
        # Current detection accumulator - aggregate multiple frames
        self.detection_votes: Dict[int, int] = defaultdict(int)  # square -> vote count
        self.vote_window = 10  # frames to accumulate
        self.frame_in_window = 0
        
        # Chess engine
        self.board = chess.Board()
        self.moves: List[str] = []
        
        # Tracking state
        self.is_tracking = False
        self.frame_count = 0
        self.last_move_frame = 0
        self.cooldown_frames = 15  # Wait this many frames after a move before detecting next
        
        # Pending move - need consistent detection across frames
        self.pending_move: Optional[chess.Move] = None
        self.pending_from: Optional[int] = None
        self.pending_to: Optional[int] = None
        self.pending_count = 0
        self.confirm_frames = 4  # Need this many consistent frames
        
    def calibrate(self, corners: np.ndarray):
        """Set up board transforms."""
        self.corners = corners.astype(np.float32)
        dst = np.array([
            [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE],
            [0, 0], [BOARD_SIZE, 0],
        ], dtype=np.float32)
        self.transform = get_perspective_transform(self.corners, dst)
        self.inv_transform = get_perspective_transform(dst, self.corners)
        
    def start_tracking(self):
        """Lock starting position and begin tracking."""
        self.locked_pieces = STARTING_PIECES.copy()
        self.stable_occupancy = set(STARTING_PIECES.keys())
        self.is_tracking = True
            
        print("\n" + "="*60)
        print("STARTING POSITION LOCKED - 32 PIECES")
        print("Now tracking moves...")
        print("="*60)
        
    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """Process a frame and detect moves."""
        if not self.is_tracking or self.transform is None:
            return None
            
        self.frame_count += 1
        
        # Detect which squares have pieces
        detected = self._detect_occupancy(frame)
        
        # Accumulate votes
        for sq in detected:
            self.detection_votes[sq] += 1
        self.frame_in_window += 1
        
        # Every vote_window frames, analyze the votes
        if self.frame_in_window >= self.vote_window:
            move_result = self._analyze_votes_and_detect_move()
            
            # Reset votes
            self.detection_votes.clear()
            self.frame_in_window = 0
            
            return move_result
            
        return None
        
    def _detect_occupancy(self, frame: np.ndarray) -> Set[int]:
        """Detect which squares are occupied."""
        raw = self.detector.detect(frame, self.corners)
        
        if len(raw.get('centers', [])) == 0:
            return set()
            
        occupied = set()
        for center in raw['centers']:
            point = np.array([[center]], dtype=np.float32)
            warped = cv2.perspectiveTransform(point, self.transform)[0, 0]
            
            if 0 <= warped[0] < BOARD_SIZE and 0 <= warped[1] < BOARD_SIZE:
                distances = np.linalg.norm(self.square_centers - warped, axis=1)
                sq = int(np.argmin(distances))
                occupied.add(sq)
                
        return occupied
        
    def _analyze_votes_and_detect_move(self) -> Optional[str]:
        """Analyze accumulated votes to detect a move."""
        # Skip if in cooldown
        if self.frame_count - self.last_move_frame < self.cooldown_frames:
            return None
            
        # Determine which squares are "reliably" occupied
        # Require at least 30% of votes to consider occupied
        threshold = self.vote_window * 0.3
        current_occupied = set(sq for sq, votes in self.detection_votes.items() if votes >= threshold)
        
        # Compare current vs stable (what we expect)
        vacated = self.stable_occupancy - current_occupied  # Was occupied, now empty
        appeared = current_occupied - self.stable_occupancy  # Was empty, now occupied
        
        # Debug output
        if vacated or appeared:
            vacated_names = sorted([SQUARE_NAMES[sq] for sq in vacated])
            appeared_names = sorted([SQUARE_NAMES[sq] for sq in appeared])
            print(f"[Frame {self.frame_count}] Stable={len(self.stable_occupancy)}, Current={len(current_occupied)}")
            print(f"  Vacated: {vacated_names}")
            print(f"  Appeared: {appeared_names}")
        
        # Try to find a legal move that matches the pattern
        best_move = self._find_matching_move(vacated, appeared, current_occupied)
        
        if best_move:
            # Confirm pending move
            if best_move == self.pending_move:
                self.pending_count += 1
                if self.pending_count >= 2:  # Confirmed over 2 analysis windows
                    result = self._execute_move(best_move)
                    self.pending_move = None
                    self.pending_count = 0
                    return result
            else:
                self.pending_move = best_move
                self.pending_count = 1
                print(f"  Pending: {self.board.san(best_move)}")
                
        return None
        
    def _find_matching_move(self, vacated: Set[int], appeared: Set[int], current: Set[int]) -> Optional[chess.Move]:
        """Find a legal move that matches the observed changes."""
        if not vacated:
            return None
            
        # Score each legal move based on how well it matches observations
        best_score = -1
        best_move = None
        
        for move in self.board.legal_moves:
            score = 0
            from_sq = move.from_square
            to_sq = move.to_square
            
            # From square should be vacated
            if from_sq in vacated:
                score += 3
                
            # To square should appear OR remain occupied (for captures)
            if to_sq in appeared:
                score += 3
            elif to_sq in current:  # Piece moved to an occupied square (capture)
                score += 2
                
            # For castling, check rook movement too
            if self.board.is_castling(move):
                if move.to_square == chess.G1:  # Kingside white
                    if chess.H1 in vacated and chess.F1 in (appeared | current):
                        score += 2
                elif move.to_square == chess.C1:  # Queenside white
                    if chess.A1 in vacated and chess.D1 in (appeared | current):
                        score += 2
                elif move.to_square == chess.G8:  # Kingside black
                    if chess.H8 in vacated and chess.F8 in (appeared | current):
                        score += 2
                elif move.to_square == chess.C8:  # Queenside black
                    if chess.A8 in vacated and chess.D8 in (appeared | current):
                        score += 2
                        
            if score > best_score:
                best_score = score
                best_move = move
                
        # Only return if we have reasonable confidence
        if best_score >= 4:  # At least from vacated + to appeared/current
            return best_move
            
                }
                piece = PIECE_TO_INDEX[promo_map[move.promotion]]
                
            del self.locked_pieces[from_sq]
            
            # Remove captured piece
            if to_sq in self.locked_pieces:
                del self.locked_pieces[to_sq]
                
            self.locked_pieces[to_sq] = piece
            
        # Handle castling rook
        if self.board.is_castling(move):
            self._move_castling_rook(move)
            
        # Handle en passant
        if self.board.is_en_passant(move):
            captured_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
            if captured_sq in self.locked_pieces:
                del self.locked_pieces[captured_sq]
                
        self.board.push(move)
        self.moves.append(san)
        self.last_move_frame = self.frame_count
        
        return san
        
    def _move_castling_rook(self, move: chess.Move):
        """Move rook for castling."""
        rook_moves = {
            chess.G1: (chess.H1, chess.F1),  # White kingside
            chess.C1: (chess.A1, chess.D1),  # White queenside
            chess.G8: (chess.H8, chess.F8),  # Black kingside
            chess.C8: (chess.A8, chess.D8),  # Black queenside
        }
        if move.to_square in rook_moves:
            from_rook, to_rook = rook_moves[move.to_square]
            if from_rook in self.locked_pieces:
                rook = self.locked_pieces.pop(from_rook)
                self.locked_pieces[to_rook] = rook
                
    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Create visualization."""
        vis = frame.copy()
        
        # Draw board
        if self.corners is not None:
            pts = self.corners.astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            
        # Draw pieces
        if self.is_tracking and self.inv_transform is not None:
            for sq, cls in self.locked_pieces.items():
                file, rank = sq % 8, sq // 8
                wx = (file + 0.5) * SQUARE_SIZE
                wy = BOARD_SIZE - (rank + 0.5) * SQUARE_SIZE
                center = warp_point(np.array([wx, wy]), self.inv_transform)
                cx, cy = int(center[0]), int(center[1])
                
                color = PIECE_COLORS[cls] if cls < len(PIECE_COLORS) else (128, 128, 128)
                label = PIECE_LABELS[cls] if cls < len(PIECE_LABELS) else '?'
                
                # Highlight if occupancy is low (piece might have moved)
                occ = self.occupancy[sq]
                if occ < 0.5:
                    color = (0, 0, 255)  # Red = potential move
                    
                cv2.rectangle(vis, (cx-18, cy-18), (cx+18, cy+18), color, 2)
                cv2.putText(vis, label, (cx-6, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                           
        # Status
        status = f"Tracking: {len(self.moves)} moves" if self.is_tracking else "Press SPACE to start"
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Moves
        if self.moves:
            move_text = " ".join(self.moves[-12:])
            cv2.putText(vis, move_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        return vis
        
    def generate_pgn(self, video_name: str) -> str:
        """Generate PGN."""
        game = chess.pgn.Game()
        game.headers["Event"] = "ChessWorldAI"
        game.headers["Site"] = video_name
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
            game.headers["Result"] = "1-0" if board.turn == chess.BLACK else "0-1"
        elif board.is_stalemate():
            game.headers["Result"] = "1/2-1/2"
        else:
            game.headers["Result"] = "*"
            
        return str(game)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ChessWorldAI - Assume & Track")
    parser.add_argument('video', help='Video file')
    parser.add_argument('--output', '-o', help='Output PGN')
    parser.add_argument('--model', '-m', default='models/pieces_trained.pt')
    parser.add_argument('--speed', '-s', type=float, default=1.0)
    parser.add_argument('--crop', type=float, default=0.45)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path} not found")
        sys.exit(1)
        
    output_path = args.output or f"output/pgn/{video_path.stem}.pgn"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    model_path = str(Path(args.model)) if Path(args.model).exists() else None
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_height = int(height * args.crop)
    
    print("\n" + "="*60)
    print("   ChessWorldAI - Assume & Track Processor")
    print("="*60)
    print(f"Video: {video_path.name}")
    print(f"Size: {width}x{height} -> crop to {width}x{crop_height}")
    print(f"FPS: {fps:.1f}")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        sys.exit(1)
    frame = frame[:crop_height, :]
    
    # Initialize
    processor = AssumeTrackProcessor(model_path=model_path, device=args.device)
    
    # Calibrate
    corners = interactive_corners(frame)
    if corners is None:
        cap.release()
        sys.exit(0)
    processor.calibrate(corners)
    
    # Timing
    base_delay = int(1000 / fps)
    frame_delay = max(1, int(base_delay / args.speed))
    sample_rate = max(1, int(fps / 15))
    
    print("\n" + "="*60)
    print("PRESS SPACE when the board is in STARTING POSITION")
    print("Then tracking will begin automatically")
    print("Controls: Q=quit, SPACE=start, +/-=speed")
    print("="*60)
    
    paused = True  # Start paused
    frame_num = 0
    start_time = time.time()
    
    try:
        while True:
            # Handle input first
            key = cv2.waitKey(frame_delay if not paused else 30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not processor.is_tracking:
                    processor.start_tracking()
                    paused = False
                else:
                    paused = not paused
            elif key in (ord('+'), ord('=')):
                args.speed = min(10, args.speed * 1.5)
                frame_delay = max(1, int(base_delay / args.speed))
            elif key == ord('-'):
                args.speed = max(0.1, args.speed / 1.5)
                frame_delay = max(1, int(base_delay / args.speed))
                
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_num += 1
                frame = frame[:crop_height, :]
                
                if frame_num % sample_rate == 0:
                    move = processor.process_frame(frame)
                    if move:
                        print(f"[Frame {frame_num}] Move {len(processor.moves)}: {move}")
                        
            # Visualize
            vis = processor.visualize(frame)
            
            pct = 100 * frame_num / total_frames
            cv2.putText(vis, f"Frame {frame_num}/{total_frames} ({pct:.0f}%)",
                       (10, crop_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                       
            if paused and not processor.is_tracking:
                cv2.putText(vis, "PRESS SPACE TO START", (width//2-120, crop_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif paused:
                cv2.putText(vis, "PAUSED", (width//2-50, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                           
            cv2.imshow("ChessWorldAI", vis)
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    # Generate PGN
    pgn = processor.generate_pgn(video_path.stem)
    
    with open(output_path, 'w') as f:
        f.write(pgn)
        
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print(f"Time: {elapsed:.1f}s")
    print(f"Moves detected: {len(processor.moves)}")
    print(f"\nMoves: {' '.join(processor.moves)}")
    print(f"\nPGN saved: {output_path}")
    print("-"*40)
    print(pgn)


if __name__ == '__main__':
    main()
