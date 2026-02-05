#!/usr/bin/env python3
"""
ChessWorldAI - Smart Tracker V2 Enhanced
Based on V2's solid approach with improvements:
1. Uses your trained pieces_enhanced.pt model directly
2. Clustering to avoid multiple detections per piece
3. Board boundary filtering
4. Better piece type matching with confidence weighting
5. Improved move confirmation logic
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

from src.config import BOARD_SIZE, SQUARE_SIZE


# Constants
SQUARE_NAMES = [f"{chr(ord('a')+f)}{r+1}" for r in range(8) for f in range(8)]

# Map class index to piece info (symbol, color, piece_type)
# From your pieces_enhanced.pt model
CLASS_TO_PIECE = {
    0: None,     # generic bishop - ignore
    1: ('b', 'black', 'bishop'),   # black-bishop
    2: ('k', 'black', 'king'),     # black-king
    3: ('n', 'black', 'knight'),   # black-knight
    4: ('p', 'black', 'pawn'),     # black-pawn
    5: ('q', 'black', 'queen'),    # black-queen
    6: ('r', 'black', 'rook'),     # black-rook
    7: ('B', 'white', 'bishop'),   # white-bishop
    8: ('K', 'white', 'king'),     # white-king
    9: ('N', 'white', 'knight'),   # white-knight
    10: ('P', 'white', 'pawn'),    # white-pawn
    11: ('Q', 'white', 'queen'),   # white-queen
    12: ('R', 'white', 'rook'),    # white-rook
}

# Standard starting position: square -> (piece_symbol, color)
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
    names = ["h1 (white's right)", "a1 (white's left)", "a8 (black's left)", "h8 (black's right)"]
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


class SmartTrackerV2Enhanced:
    """
    Enhanced V2 tracker with better piece recognition.
    """
    
    def __init__(self, model_path: str = 'models/pieces.pt'):
        from ultralytics import YOLO
        
        # Load model directly
        self.model = YOLO(model_path)
        print(f"✓ Loaded model: {model_path}")
        
        # Geometry
        self.corners: Optional[np.ndarray] = None
        self.transform: Optional[np.ndarray] = None
        self.inv_transform: Optional[np.ndarray] = None
        self.square_centers = get_square_centers()
        self.board_polygon: Optional[np.ndarray] = None
        
        # Board state: square -> (piece_symbol, color)
        self.board_state: Dict[int, Tuple[str, str]] = {}
        
        # Detection accumulator: square -> list of (class_id, confidence)
        self.detection_votes: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.vote_window = 4  # Frames to accumulate (faster response)
        self.frame_in_window = 0
        
        # Clustering parameters - smaller to avoid merging adjacent piece detections
        self.cluster_distance = 30  # Pixels
        
        # Persistence: remember detections for squares we expect pieces on
        self.detection_memory: Dict[int, Tuple[int, float, int]] = {}  # sq -> (class, conf, last_seen_frame)
        self.memory_decay_frames = 45  # Keep memory for this many frames (longer)
        
        # Chess engine
        self.board = chess.Board()
        self.moves: List[str] = []
        
        # Tracking
        self.is_tracking = False
        self.frame_count = 0
        self.last_move_frame = 0
        self.cooldown = 8  # Frames between moves
        
        # Pending confirmation
        self.pending_move: Optional[chess.Move] = None
        self.pending_count = 0
        self.confirm_needed = 2
        
        # Debug
        self.last_detected: Dict[int, Tuple[int, float]] = {}  # sq -> (class, conf)
        self.last_raw_detections: List[Dict] = []
        self.debug_mode = False
        
        # Piece to class mapping (reverse of CLASS_TO_PIECE)
        self.piece_to_class_map = {
            ('b', 'black'): 1, ('k', 'black'): 2, ('n', 'black'): 3,
            ('p', 'black'): 4, ('q', 'black'): 5, ('r', 'black'): 6,
            ('B', 'white'): 7, ('K', 'white'): 8, ('N', 'white'): 9,
            ('P', 'white'): 10, ('Q', 'white'): 11, ('R', 'white'): 12,
        }
        
    def _piece_to_class(self, piece: str, color: str) -> Optional[int]:
        """Convert piece symbol and color to class ID."""
        return self.piece_to_class_map.get((piece, color))
        
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
        """Start tracking from assumed starting position."""
        self.board_state = STARTING_POSITION.copy()
        self.is_tracking = True
        print("\n" + "="*50)
        print("TRACKING STARTED - Assuming standard starting position")
        print("="*50)
        print(f"Tracking {len(self.board_state)} pieces")
        print("Controls: Q=quit, SPACE=pause, D=debug, +/-=speed")
        
    def _is_inside_board(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside board boundary."""
        if self.board_polygon is None:
            return True
        result = cv2.pointPolygonTest(self.board_polygon, (float(point[0]), float(point[1])), False)
        return result >= 0
        
    def _cluster_detections(self, detections: List[Dict]) -> List[Dict]:
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
        
    def _detect_raw(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLO detection with filtering and clustering."""
        # Very low confidence - catch everything, filter by votes later
        results = self.model(frame, verbose=False, conf=0.10)
        
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
                    
                    # Filter: inside board only
                    if not self._is_inside_board((cx, cy)):
                        continue
                        
                    # Skip generic class 0
                    if cls == 0:
                        continue
                    
                    detections.append({
                        'center': (cx, cy),
                        'class': cls,
                        'confidence': conf,
                        'bbox': box
                    })
        
        # Cluster close detections
        clustered = self._cluster_detections(detections)
        self.last_raw_detections = clustered
        return clustered
        
    def process(self, frame: np.ndarray) -> Optional[str]:
        if not self.is_tracking or self.transform is None:
            return None
            
        self.frame_count += 1
        
        # Detect pieces
        raw_detections = self._detect_raw(frame)
        
        # Map to squares and accumulate votes
        for det in raw_detections:
            sq = self._point_to_square(det['center'])
            if sq is not None:
                self.detection_votes[sq].append((det['class'], det['confidence']))
        
        self.frame_in_window += 1
        
        # Analyze every vote_window frames
        if self.frame_in_window >= self.vote_window:
            result = self._analyze()
            self.detection_votes.clear()
            self.frame_in_window = 0
            return result
            
        return None
        
    def _point_to_square(self, point: Tuple[float, float]) -> Optional[int]:
        """Convert image point to square index."""
        pt = np.array([[point]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.transform)[0, 0]
        
        if 0 <= warped[0] < BOARD_SIZE and 0 <= warped[1] < BOARD_SIZE:
            dists = np.linalg.norm(self.square_centers - warped, axis=1)
            return int(np.argmin(dists))
        return None
        
    def _analyze(self) -> Optional[str]:
        """Analyze accumulated votes and detect moves."""
        if self.frame_count - self.last_move_frame < self.cooldown:
            return None
            
        # Build current detection map: sq -> (best_class, avg_confidence)
        current_detection: Dict[int, Tuple[int, float]] = {}
        
        for sq, votes in self.detection_votes.items():
            if not votes:
                continue
                
            # Weight by confidence
            class_scores = defaultdict(float)
            class_counts = defaultdict(int)
            
            for cls, conf in votes:
                class_scores[cls] += conf
                class_counts[cls] += 1
                
            # Get best class
            best_cls = max(class_scores.keys(), key=lambda c: class_scores[c])
            avg_conf = class_scores[best_cls] / class_counts[best_cls]
            
            # Need at least 25% of votes for this class (very lenient for recall)
            vote_ratio = class_counts[best_cls] / len(votes)
            if vote_ratio >= 0.25 and avg_conf >= 0.15:
                current_detection[sq] = (best_cls, avg_conf)
                # Update memory
                self.detection_memory[sq] = (best_cls, avg_conf, self.frame_count)
        
        # Add persistent detections from memory for expected squares
        for sq in self.board_state.keys():
            if sq not in current_detection:
                # Check if we have recent memory for this square
                if sq in self.detection_memory:
                    _, conf, last_seen = self.detection_memory[sq]
                    frames_ago = self.frame_count - last_seen
                    if frames_ago < self.memory_decay_frames:
                        # Use memory with decayed confidence
                        decay = 1.0 - (frames_ago / self.memory_decay_frames)
                        piece, color = self.board_state[sq]
                        # Find expected class for this piece
                        expected_cls = self._piece_to_class(piece, color)
                        if expected_cls is not None:
                            current_detection[sq] = (expected_cls, conf * decay * 0.5)
                
        self.last_detected = current_detection
        
        # Compare to expected board state
        expected_squares = set(self.board_state.keys())
        detected_squares = set(current_detection.keys())
        
        vacated = expected_squares - detected_squares
        appeared = detected_squares - expected_squares
        
        if not vacated:
            return None
            
        # Debug output (limit noise)
        if len(vacated) <= 4 and (vacated or appeared):
            vnames = sorted([SQUARE_NAMES[s] for s in vacated])
            anames = sorted([SQUARE_NAMES[s] for s in appeared])
            if self.debug_mode:
                print(f"[{self.frame_count}] Vacated: {vnames}, Appeared: {anames}")
        
        # Find best matching move with piece type validation
        move = self._find_best_move(vacated, appeared, current_detection)
        
        if move:
            if move == self.pending_move:
                self.pending_count += 1
                if self.pending_count >= self.confirm_needed:
                    return self._execute(move)
            else:
                self.pending_move = move
                self.pending_count = 1
                san = self.board.san(move)
                print(f"  Pending: {san}")
                
        return None
        
    def _find_best_move(self, vacated: Set[int], appeared: Set[int], 
                        current_detection: Dict[int, Tuple[int, float]]) -> Optional[chess.Move]:
        """Find legal move that best matches observations with piece type validation."""
        if not vacated:
            return None
            
        best_score = 0
        best_move = None
        
        for move in self.board.legal_moves:
            score = self._score_move(move, vacated, appeared, current_detection)
            if score > best_score:
                best_score = score
                best_move = move
                
        # Lower threshold - trust vacated square detection more
        min_score = 3
        if best_move and self.board.is_castling(best_move):
            min_score = 2  # Lower threshold for castling
            
        return best_move if best_score >= min_score else None
        
    def _score_move(self, move: chess.Move, vacated: Set[int], appeared: Set[int],
                    current_detection: Dict[int, Tuple[int, float]]) -> int:
        """Score how well a move matches observations using piece type."""
        score = 0
        from_sq = move.from_square
        to_sq = move.to_square
        
        # From square should be vacated - THIS IS THE KEY SIGNAL
        if from_sq in vacated:
            score += 4  # Increased from 3 - vacated source is most reliable
            
        # Check piece at destination
        if from_sq in self.board_state:
            expected_symbol, expected_color = self.board_state[from_sq]
            expected_type = expected_symbol.upper()  # P, N, B, R, Q, K
            
            # Destination has ANY detection = good sign
            if to_sq in current_detection:
                detected_cls, det_conf = current_detection[to_sq]
                piece_info = CLASS_TO_PIECE.get(detected_cls)
                
                if piece_info:
                    detected_symbol, detected_color, detected_type = piece_info
                    
                    # Color match (most important)
                    if detected_color == expected_color:
                        score += 2
                        
                        # Type match
                        if detected_symbol.upper() == expected_type:
                            score += 2
                            # Confidence bonus
                            if det_conf > 0.5:
                                score += 1
                    else:
                        # Wrong color - penalty but not too harsh
                        score -= 1
            
            # Even if destination not detected, if source is vacated that's meaningful
            # Check if destination is a valid empty square (capture or move)
            if to_sq not in current_detection and to_sq in appeared:
                # Something appeared here even if not identified
                score += 1
                        
            elif to_sq in appeared:
                # Appeared but no class info - partial credit
                score += 1
                
        # Castling bonus
        if self.board.is_castling(move):
            score += self._score_castling(move, vacated, current_detection)
            
        # En passant
        if self.board.is_en_passant(move):
            cap_sq = chess.square(chess.square_file(to_sq), chess.square_rank(from_sq))
            if cap_sq in vacated:
                score += 2
                
        return score
        
    def _score_castling(self, move: chess.Move, vacated: Set[int],
                        current_detection: Dict[int, Tuple[int, float]]) -> int:
        """Score castling move."""
        score = 0
        rook_map = {
            chess.G1: (chess.H1, chess.F1),
            chess.C1: (chess.A1, chess.D1),
            chess.G8: (chess.H8, chess.F8),
            chess.C8: (chess.A8, chess.D8),
        }
        
        info = rook_map.get(move.to_square)
        if info:
            rook_from, rook_to = info
            if rook_from in vacated:
                score += 2
            if rook_to in current_detection:
                score += 1
                
        return score
        
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
            rook_map = {
                chess.G1: (chess.H1, chess.F1),
                chess.C1: (chess.A1, chess.D1),
                chess.G8: (chess.H8, chess.F8),
                chess.C8: (chess.A8, chess.D8),
            }
            info = rook_map.get(move.to_square)
            if info:
                rook_from, rook_to = info
                if rook_from in self.board_state:
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
                
                # Color based on detection status
                if sq in self.last_detected:
                    det_cls, det_conf = self.last_detected[sq]
                    if det_conf > 0.3:
                        # Strong detection - white/black color
                        box_color = (255, 200, 100) if color == 'white' else (100, 100, 255)
                    else:
                        # Weak/memory detection - orange/purple
                        box_color = (0, 165, 255) if color == 'white' else (180, 100, 180)
                else:
                    box_color = (0, 0, 255)  # Red = not detected at all
                    
                cv2.rectangle(vis, (cx-16, cy-16), (cx+16, cy+16), box_color, 2)
                cv2.putText(vis, piece.upper(), (cx-6, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                           
        # Debug: show raw detections
        if self.debug_mode:
            for det in self.last_raw_detections:
                cx, cy = int(det['center'][0]), int(det['center'][1])
                cls = det['class']
                conf = det['confidence']
                piece_info = CLASS_TO_PIECE.get(cls)
                if piece_info:
                    label = f"{piece_info[0]}:{conf:.2f}"
                    cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.putText(vis, label, (cx+5, cy-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                           
        # Status
        turn_str = "White" if self.board.turn == chess.WHITE else "Black"
        status = f"Moves: {len(self.moves)} | {turn_str} to move" if self.is_tracking else "Press SPACE to start"
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.moves:
            recent = self.moves[-10:]
            move_str = " ".join(recent)
            cv2.putText(vis, move_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                       
        return vis
        
    def pgn(self, name: str) -> str:
        game = chess.pgn.Game()
        game.headers["Event"] = "ChessWorldAI V2 Enhanced"
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
            game.headers["Result"] = "1-0" if board.turn == chess.BLACK else "0-1"
        elif board.is_stalemate() or board.is_insufficient_material():
            game.headers["Result"] = "1/2-1/2"
        else:
            game.headers["Result"] = "*"
            
        return str(game)


def main():
    parser = argparse.ArgumentParser(description="ChessWorldAI V2 Enhanced Tracker")
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
        
    # Find model - prefer pieces.pt (original roboflow model)
    model_path = args.model
    if not Path(model_path).exists():
        alternatives = [
            'models/pieces.pt',
            'models/pieces_trained.pt',
            'models/pieces_enhanced.pt',
        ]
        for alt in alternatives:
            if Path(alt).exists():
                model_path = alt
                break
                
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
        
    output = args.output or f"output/pgn/{video.stem}_v2e.pgn"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*50}")
    print("ChessWorldAI V2 Enhanced")
    print(f"{'='*50}")
    print(f"Video: {video.name}")
    print(f"Model: {model_path}")
    
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
    
    tracker = SmartTrackerV2Enhanced(model_path=model_path)
    tracker.debug_mode = args.debug
    
    corners = interactive_corners(frame)
    if corners is None:
        cap.release()
        sys.exit(0)
    tracker.calibrate(corners)
    
    base_delay = int(1000 / fps)
    delay = max(1, int(base_delay / args.speed))
    sample_rate = max(1, int(fps / 15))
    
    paused = True
    frame_num = 0
    
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
                print(f"Debug: {'ON' if tracker.debug_mode else 'OFF'}")
            elif key in (ord('+'), ord('=')):
                args.speed = min(10, args.speed * 1.5)
                delay = max(1, int(base_delay / args.speed))
                print(f"Speed: {args.speed:.1f}x")
            elif key == ord('-'):
                args.speed = max(0.1, args.speed / 1.5)
                delay = max(1, int(base_delay / args.speed))
                print(f"Speed: {args.speed:.1f}x")
                
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video")
                    break
                frame_num += 1
                frame = frame[:crop_h, :]
                
                if frame_num % sample_rate == 0:
                    move = tracker.process(frame)
                    if move:
                        print(f"[Frame {frame_num}] Move #{len(tracker.moves)}: {move}")
                        
            vis = tracker.visualize(frame)
            
            pct = 100 * frame_num / total if total else 0
            cv2.putText(vis, f"Frame: {frame_num}/{total} ({pct:.0f}%) | Speed: {args.speed:.1f}x", 
                       (10, crop_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
            if paused and not tracker.is_tracking:
                cv2.putText(vis, "PRESS SPACE TO START", (w//2-150, crop_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif paused:
                cv2.putText(vis, "PAUSED", (w//2-50, crop_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                           
            cv2.imshow("ChessWorldAI V2 Enhanced", vis)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    pgn_str = tracker.pgn(video.stem)
    
    with open(output, 'w') as f:
        f.write(pgn_str)
        
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Total moves: {len(tracker.moves)}")
    print(f"Moves: {' '.join(tracker.moves)}")
    print(f"\nPGN saved to: {output}")
    print(f"\n{pgn_str}")


if __name__ == '__main__':
    main()
