"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    POSITION-BASED CHESS TRACKER                              ║
║                    "Perfect 32 Boxes" Approach                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This tracker TRUSTS the starting position completely and uses               ║
║  YOLO only to detect PRESENCE of pieces (not classification).               ║
║  Moves are inferred by detecting when pieces LEAVE and ARRIVE.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import cv2
import chess
import chess.pgn
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import onnxruntime as ort
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

# Constants
MODEL_WIDTH = 480
MODEL_HEIGHT = 288
SQUARE_SIZE = 128
BOARD_SIZE = 8 * SQUARE_SIZE

# Piece labels from model (we ignore these for move detection)
LABELS = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]

# Square names a1=0, b1=1, ... h8=63
SQUARE_NAMES = [
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
    'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
    'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
    'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8'
]

# Starting position: square_index -> piece_symbol
# Using chess module's square indices (a1=0, h8=63)
# Uppercase = white, lowercase = black
STARTING_POSITION = {
    # White pieces (rank 1)
    chess.A1: 'R', chess.B1: 'N', chess.C1: 'B', chess.D1: 'Q',
    chess.E1: 'K', chess.F1: 'B', chess.G1: 'N', chess.H1: 'R',
    # White pawns (rank 2)
    chess.A2: 'P', chess.B2: 'P', chess.C2: 'P', chess.D2: 'P',
    chess.E2: 'P', chess.F2: 'P', chess.G2: 'P', chess.H2: 'P',
    # Black pawns (rank 7)
    chess.A7: 'p', chess.B7: 'p', chess.C7: 'p', chess.D7: 'p',
    chess.E7: 'p', chess.F7: 'p', chess.G7: 'p', chess.H7: 'p',
    # Black pieces (rank 8)
    chess.A8: 'r', chess.B8: 'n', chess.C8: 'b', chess.D8: 'q',
    chess.E8: 'k', chess.F8: 'b', chess.G8: 'n', chess.H8: 'r',
}


@dataclass
class SquareTracker:
    """Tracks detection history for a single square."""
    square_idx: int
    piece: Optional[str] = None  # What piece SHOULD be here (None if empty)
    is_white: Optional[bool] = None
    
    # Detection history: sliding window of recent detections
    history: deque = field(default_factory=lambda: deque(maxlen=15))
    
    def add_detection(self, detected: bool):
        """Add a detection result to history."""
        self.history.append(detected)
    
    def detection_rate(self) -> float:
        """Get rate of positive detections in window."""
        if len(self.history) == 0:
            return 1.0 if self.piece else 0.0
        return sum(self.history) / len(self.history)
    
    def is_vacated(self) -> bool:
        """Check if piece has left this square."""
        if self.piece is None:
            return False
        # Must have enough history and low detection rate
        if len(self.history) < 5:
            return False
        return self.detection_rate() < 0.3
    
    def has_arrival(self) -> bool:
        """Check if a piece has arrived (was empty, now occupied)."""
        if self.piece is not None:
            return False
        if len(self.history) < 5:
            return False
        return self.detection_rate() > 0.6
    
    def clear_history(self):
        """Clear detection history after a confirmed move."""
        self.history.clear()


class PositionBasedTracker:
    """
    Chess tracker that trusts starting position and uses YOLO only for presence detection.
    """
    
    # Detection threshold - very low since we only need presence
    DETECTION_THRESHOLD = 0.15
    
    # Move detection thresholds
    VACATED_THRESHOLD = 0.3   # Below this = piece has left
    ARRIVED_THRESHOLD = 0.6   # Above this = piece has arrived
    
    # Confirmation
    CONFIRM_FRAMES = 2        # Frames to confirm a move
    COOLDOWN_FRAMES = 3      # Frames to wait after confirming a move
    
    # Minimum score to consider a move
    MIN_MOVE_SCORE = 4
    
    def __init__(self, pieces_model_path: str, corners_model_path: str):
        self.pieces_model_path = Path(pieces_model_path)
        self.corners_model_path = Path(corners_model_path)
        
        self._load_models()
        
        # Board geometry
        self.corners = None
        self.centers = None
        self.boundary = None
        self.transform = None
        self.inv_transform = None
        
        # Tracking state
        self.square_trackers: Dict[int, SquareTracker] = {}
        self.frame_count = 0
        
        # Move detection state
        self.pending_move: Optional[chess.Move] = None
        self.pending_move_san: Optional[str] = None
        self.pending_count = 0
        self.cooldown = 0
        
        # Game state
        self.board = chess.Board()
        self.moves: List[str] = []
        self.is_initialized = False
        
        # Debug info
        self.last_vacated: Set[int] = set()
        self.last_arrived: Set[int] = set()
        self.last_detected: Set[int] = set()
        self.raw_detections: List[Dict] = []
    
    def _load_models(self):
        pieces_onnx = self.pieces_model_path / "model.onnx"
        corners_onnx = self.corners_model_path / "model.onnx"
        
        if not pieces_onnx.exists():
            raise FileNotFoundError(f"Pieces model not found: {pieces_onnx}")
        if not corners_onnx.exists():
            raise FileNotFoundError(f"Corners model not found: {corners_onnx}")
        
        print("Loading ONNX models...")
        self.pieces_session = ort.InferenceSession(
            str(pieces_onnx), providers=['CPUExecutionProvider']
        )
        self.corners_session = ort.InferenceSession(
            str(corners_onnx), providers=['CPUExecutionProvider']
        )
        
        pieces_input = self.pieces_session.get_inputs()[0]
        print(f"  Pieces model input: {pieces_input.name}, shape: {pieces_input.shape}")
        corners_input = self.corners_session.get_inputs()[0]
        print(f"  Corners model input: {corners_input.name}, shape: {corners_input.shape}")
        print("Models loaded successfully")
    
    def reset(self):
        """Reset for a new game."""
        self.square_trackers = {}
        self.frame_count = 0
        self.pending_move = None
        self.pending_move_san = None
        self.pending_count = 0
        self.cooldown = 0
        self.board = chess.Board()
        self.moves = []
        self.is_initialized = False
        self.last_vacated = set()
        self.last_arrived = set()
        self.last_detected = set()
        self.raw_detections = []
    
    def initialize_board_state(self):
        """Initialize all 64 squares with starting position."""
        self.square_trackers = {}
        
        for sq in range(64):
            if sq in STARTING_POSITION:
                piece = STARTING_POSITION[sq]
                is_white = piece.isupper()
                tracker = SquareTracker(sq, piece=piece, is_white=is_white)
            else:
                tracker = SquareTracker(sq, piece=None, is_white=None)
            self.square_trackers[sq] = tracker
        
        self.is_initialized = True
        print(f"Initialized board with 32 pieces in starting position")
    
    def set_corners(self, corners: Dict[str, List[float]]):
        """Set board corners and compute transforms."""
        self.corners = corners
        self._compute_transforms()
    
    def _compute_transforms(self):
        if self.corners is None:
            return
        
        # Source points from user clicks
        src = np.array([
            self.corners['a1'],
            self.corners['h1'],
            self.corners['h8'],
            self.corners['a8']
        ], dtype=np.float32)
        
        # Destination: standard board coordinates
        dst = np.array([
            [0, BOARD_SIZE],           # a1 -> bottom-left
            [BOARD_SIZE, BOARD_SIZE],  # h1 -> bottom-right
            [BOARD_SIZE, 0],           # h8 -> top-right
            [0, 0]                     # a8 -> top-left
        ], dtype=np.float32)
        
        self.transform = cv2.getPerspectiveTransform(src, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, src)
        
        # Compute square centers in image coordinates
        centers_board = []
        for rank in range(8):  # 0 to 7
            for file in range(8):  # 0 to 7
                # Square center in board coordinates
                x = (file + 0.5) * SQUARE_SIZE
                y = (7 - rank + 0.5) * SQUARE_SIZE
                centers_board.append([x, y])
        
        centers_board = np.array(centers_board, dtype=np.float32).reshape(-1, 1, 2)
        self.centers = cv2.perspectiveTransform(centers_board, self.inv_transform).reshape(-1, 2)
        self.boundary = src.copy()
    
    def preprocess_image(self, frame: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Preprocess image for model input."""
        h, w = frame.shape[:2]
        
        if keypoints is not None and len(keypoints) >= 4:
            xs, ys = keypoints[:, 0], keypoints[:, 1]
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            padding = 12
            
            pad_left = bbox_width / padding
            pad_right = bbox_width / padding
            pad_top = bbox_height / padding
            pad_bottom = bbox_height / padding
            
            padded_width = bbox_width + pad_left + pad_right
            padded_height = bbox_height + pad_top + pad_bottom
            ratio = padded_height / padded_width
            desired_ratio = MODEL_HEIGHT / MODEL_WIDTH
            
            if ratio > desired_ratio:
                target_width = padded_height / desired_ratio
                dx = target_width - padded_width
                pad_left += dx / 2
                pad_right += dx - dx / 2
            else:
                target_height = padded_width * desired_ratio
                pad_top += target_height - padded_height
            
            roi = [
                max(0, int((xmin - pad_left) * w / MODEL_WIDTH)),
                max(0, int((ymin - pad_top) * h / MODEL_HEIGHT)),
                min(w, int((xmax + pad_right) * w / MODEL_WIDTH)),
                min(h, int((ymax + pad_bottom) * h / MODEL_HEIGHT))
            ]
        else:
            roi = [0, 0, w, h]
        
        cropped = frame[roi[1]:roi[3], roi[0]:roi[2]]
        crop_h, crop_w = cropped.shape[:2]
        
        if crop_w == 0 or crop_h == 0:
            cropped = frame
            crop_h, crop_w = frame.shape[:2]
            roi = [0, 0, w, h]
        
        ratio = crop_h / crop_w
        desired_ratio = MODEL_HEIGHT / MODEL_WIDTH
        
        if ratio > desired_ratio:
            resize_h = MODEL_HEIGHT
            resize_w = int(MODEL_HEIGHT / ratio)
        else:
            resize_w = MODEL_WIDTH
            resize_h = int(MODEL_WIDTH * ratio)
        
        resized = cv2.resize(cropped, (resize_w, resize_h))
        
        dx = MODEL_WIDTH - resize_w
        dy = MODEL_HEIGHT - resize_h
        pad_right = dx // 2
        pad_left = dx - pad_right
        pad_bottom = dy // 2
        pad_top = dy - pad_bottom
        
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(transposed, axis=0)
        
        meta = {
            'roi': roi, 'crop_w': crop_w, 'crop_h': crop_h,
            'resize_w': resize_w, 'resize_h': resize_h,
            'pad_left': pad_left, 'pad_right': pad_right,
            'pad_top': pad_top, 'pad_bottom': pad_bottom,
            'original_w': w, 'original_h': h
        }
        
        return input_tensor, meta
    
    def detect_presence(self, frame: np.ndarray) -> Set[int]:
        """
        Run YOLO and return SET of squares where ANY piece is detected.
        We ignore piece classification entirely - only care about presence.
        """
        if self.corners is None:
            return set()
        
        scale_x = MODEL_WIDTH / frame.shape[1]
        scale_y = MODEL_HEIGHT / frame.shape[0]
        
        keypoints = np.array([
            [self.corners['h1'][0] * scale_x, self.corners['h1'][1] * scale_y],
            [self.corners['a1'][0] * scale_x, self.corners['a1'][1] * scale_y],
            [self.corners['a8'][0] * scale_x, self.corners['a8'][1] * scale_y],
            [self.corners['h8'][0] * scale_x, self.corners['h8'][1] * scale_y]
        ])
        
        input_tensor, meta = self.preprocess_image(frame, keypoints)
        input_name = self.pieces_session.get_inputs()[0].name
        output = self.pieces_session.run(None, {input_name: input_tensor.astype(np.float16)})
        
        # Postprocess
        preds = np.transpose(output[0], (0, 2, 1))[0]
        
        cx, cy = preds[:, 0], preds[:, 1]
        bw, bh = preds[:, 2], preds[:, 3]
        scores = preds[:, 4:]
        
        l = cx - bw / 2 - meta['pad_left']
        t = cy - bh / 2 - meta['pad_top']
        r = cx + bw / 2 - meta['pad_left']
        b = cy + bh / 2 - meta['pad_top']
        
        scale_x = meta['crop_w'] / (MODEL_WIDTH - meta['pad_left'] - meta['pad_right'])
        scale_y = meta['crop_h'] / (MODEL_HEIGHT - meta['pad_top'] - meta['pad_bottom'])
        
        l = l * scale_x + meta['roi'][0]
        r = r * scale_x + meta['roi'][0]
        t = t * scale_y + meta['roi'][1]
        b = b * scale_y + meta['roi'][1]
        
        boxes = np.stack([l, t, r, b], axis=1)
        
        # Get max confidence across ALL classes (we don't care which piece)
        max_scores = np.max(scores, axis=1)
        mask = max_scores > self.DETECTION_THRESHOLD
        
        if not np.any(mask):
            return set()
        
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        classes = np.argmax(scores[mask], axis=1)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), max_scores.tolist(), self.DETECTION_THRESHOLD, 0.4)
        
        if len(indices) == 0:
            return set()
        
        indices = np.array(indices).flatten()
        
        # Map detections to squares
        detected_squares = set()
        self.raw_detections = []
        
        for i in indices:
            box = boxes[i]
            # Use bottom-center of box (where piece base is)
            cx = (box[0] + box[2]) / 2
            cy = box[3] - (box[2] - box[0]) / 3
            
            sq = self._get_square_at(cx, cy)
            if sq >= 0:
                detected_squares.add(sq)
                self.raw_detections.append({
                    'box': box.tolist(),
                    'piece': LABELS[classes[i]],
                    'confidence': float(max_scores[i]),
                    'square_idx': sq,
                    'square_name': SQUARE_NAMES[sq]
                })
        
        return detected_squares
    
    def _get_square_at(self, x: float, y: float) -> int:
        """Get square index at image coordinates."""
        if self.centers is None:
            return -1
        
        if not self._is_inside_board(x, y):
            return -1
        
        dists = np.sqrt((self.centers[:, 0] - x)**2 + (self.centers[:, 1] - y)**2)
        return int(np.argmin(dists))
    
    def _is_inside_board(self, x: float, y: float) -> bool:
        if self.boundary is None:
            return False
        return cv2.pointPolygonTest(self.boundary.astype(np.float32), (x, y), False) >= 0
    
    def update_tracking(self, detected_squares: Set[int]):
        """Update detection history for all squares."""
        for sq, tracker in self.square_trackers.items():
            detected = sq in detected_squares
            tracker.add_detection(detected)
        
        self.last_detected = detected_squares
    
    def find_vacated_and_arrived(self) -> Tuple[Set[int], Set[int]]:
        """Find squares where pieces have left or arrived."""
        vacated = set()
        arrived = set()
        
        for sq, tracker in self.square_trackers.items():
            if tracker.is_vacated():
                vacated.add(sq)
            elif tracker.has_arrival():
                arrived.add(sq)
        
        self.last_vacated = vacated
        self.last_arrived = arrived
        
        return vacated, arrived
    
    def score_move(self, move: chess.Move, vacated: Set[int], arrived: Set[int]) -> int:
        """Score a legal move based on detection evidence."""
        score = 0
        from_sq = move.from_square
        to_sq = move.to_square
        
        # From square should be vacated
        if from_sq in vacated:
            score += 3
        
        # To square should have arrival (or at least detection)
        if to_sq in arrived:
            score += 3
        elif to_sq in self.last_detected:
            score += 2
        
        # Special handling for castling
        if self.board.is_castling(move):
            if self.board.is_kingside_castling(move):
                # Kingside: rook h1->f1 or h8->f8
                if self.board.turn == chess.WHITE:
                    rook_from, rook_to = chess.H1, chess.F1
                else:
                    rook_from, rook_to = chess.H8, chess.F8
            else:
                # Queenside: rook a1->d1 or a8->d8
                if self.board.turn == chess.WHITE:
                    rook_from, rook_to = chess.A1, chess.D1
                else:
                    rook_from, rook_to = chess.A8, chess.D8
            
            # Rook should also vacate and arrive
            if rook_from in vacated:
                score += 2
            if rook_to in arrived or rook_to in self.last_detected:
                score += 2
        
        # Special handling for en passant
        if self.board.is_en_passant(move):
            # Captured pawn square should be vacated
            if self.board.turn == chess.WHITE:
                captured_sq = to_sq - 8
            else:
                captured_sq = to_sq + 8
            if captured_sq in vacated:
                score += 2
        
        return score
    
    def detect_move(self) -> Optional[str]:
        """Try to detect a chess move from current state."""
        # Respect cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
        
        vacated, arrived = self.find_vacated_and_arrived()
        
        # If nothing interesting happening, no move
        if not vacated and not arrived:
            self.pending_move = None
            self.pending_count = 0
            return None
        
        # Score all legal moves
        best_move = None
        best_score = 0
        
        for move in legal_moves:
            score = self.score_move(move, vacated, arrived)
            if score > best_score:
                best_score = score
                best_move = move
        
        # Need minimum score
        if best_move is None or best_score < self.MIN_MOVE_SCORE:
            self.pending_move = None
            self.pending_count = 0
            return None
        
        # Check if same as pending
        if best_move == self.pending_move:
            self.pending_count += 1
        else:
            self.pending_move = best_move
            self.pending_move_san = self.board.san(best_move)
            self.pending_count = 1
        
        # Confirm if seen enough times
        if self.pending_count >= self.CONFIRM_FRAMES:
            return self._execute_move(best_move)
        
        return None
    
    def _execute_move(self, move: chess.Move) -> str:
        """Execute a confirmed move."""
        san = self.board.san(move)
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Update board state trackers
        from_tracker = self.square_trackers[from_sq]
        to_tracker = self.square_trackers[to_sq]
        
        # Handle castling - move rook too
        if self.board.is_castling(move):
            if self.board.is_kingside_castling(move):
                if self.board.turn == chess.WHITE:
                    rook_from, rook_to = chess.H1, chess.F1
                else:
                    rook_from, rook_to = chess.H8, chess.F8
            else:
                if self.board.turn == chess.WHITE:
                    rook_from, rook_to = chess.A1, chess.D1
                else:
                    rook_from, rook_to = chess.A8, chess.D8
            
            # Move rook in state
            rook_tracker = self.square_trackers[rook_from]
            self.square_trackers[rook_to].piece = rook_tracker.piece
            self.square_trackers[rook_to].is_white = rook_tracker.is_white
            rook_tracker.piece = None
            rook_tracker.is_white = None
            rook_tracker.clear_history()
            self.square_trackers[rook_to].clear_history()
        
        # Handle en passant - remove captured pawn
        if self.board.is_en_passant(move):
            if self.board.turn == chess.WHITE:
                captured_sq = to_sq - 8
            else:
                captured_sq = to_sq + 8
            self.square_trackers[captured_sq].piece = None
            self.square_trackers[captured_sq].is_white = None
            self.square_trackers[captured_sq].clear_history()
        
        # Move piece in state
        moving_piece = from_tracker.piece
        moving_is_white = from_tracker.is_white
        
        # Handle promotion
        if move.promotion:
            piece_symbol = chess.piece_symbol(move.promotion)
            moving_piece = piece_symbol.upper() if moving_is_white else piece_symbol.lower()
        
        # Update trackers
        to_tracker.piece = moving_piece
        to_tracker.is_white = moving_is_white
        from_tracker.piece = None
        from_tracker.is_white = None
        
        # Clear histories
        from_tracker.clear_history()
        to_tracker.clear_history()
        
        # Execute on chess board
        self.board.push(move)
        self.moves.append(san)
        
        # Reset pending
        self.pending_move = None
        self.pending_move_san = None
        self.pending_count = 0
        
        # Start cooldown
        self.cooldown = self.COOLDOWN_FRAMES
        
        return san
    
    def analyze_initial_frame(self, frame: np.ndarray) -> Dict:
        """Analyze initial frame to verify starting position."""
        if self.corners is None:
            return {'success': False, 'message': 'Corners not set'}
        
        self.initialize_board_state()
        detected_squares = self.detect_presence(frame)
        
        # Compare with expected starting position
        expected_squares = set(STARTING_POSITION.keys())
        detected_correct = detected_squares & expected_squares
        not_detected = expected_squares - detected_squares
        unexpected = detected_squares - expected_squares
        
        detection_results = []
        
        # Expected pieces that were detected
        for sq in detected_correct:
            piece = STARTING_POSITION[sq]
            detection_results.append({
                'square_idx': sq,
                'square_name': SQUARE_NAMES[sq],
                'expected_piece': piece,
                'detected_piece': piece,  # We assume correct since we're using starting position
                'confidence': 1.0,
                'match': True
            })
        
        # Expected pieces NOT detected
        for sq in not_detected:
            piece = STARTING_POSITION[sq]
            detection_results.append({
                'square_idx': sq,
                'square_name': SQUARE_NAMES[sq],
                'expected_piece': piece,
                'detected_piece': None,
                'confidence': 0.0,
                'match': False
            })
        
        return {
            'success': True,
            'message': f'Starting position: {len(detected_correct)}/32 pieces detected',
            'num_expected': 32,
            'num_detected_correct': len(detected_correct),
            'num_detected_wrong': 0,  # We trust starting position
            'num_not_detected': len(not_detected),
            'detection_results': detection_results
        }
    
    def process_frame(self, frame: np.ndarray, timestamp: float = 0) -> Dict:
        """Process a single frame."""
        if self.corners is None or self.centers is None:
            return {'success': False, 'message': 'Corners not set'}
        
        if not self.is_initialized:
            self.initialize_board_state()
        
        self.frame_count += 1
        
        # Detect presence (ignoring classification)
        detected_squares = self.detect_presence(frame)
        
        # Update tracking history
        self.update_tracking(detected_squares)
        
        # Try to detect move
        new_move = self.detect_move()
        
        # Build state summary
        state_summary = {}
        for sq, tracker in self.square_trackers.items():
            if tracker.piece:
                state_summary[SQUARE_NAMES[sq]] = {
                    'piece': tracker.piece,
                    'confidence': tracker.detection_rate(),
                    'locked': not tracker.is_vacated(),
                    'empty_frames': sum(1 for d in tracker.history if not d) if tracker.history else 0
                }
        
        return {
            'success': True,
            'message': f'Frame {self.frame_count}: {len(detected_squares)} detections',
            'moves': self.moves.copy(),
            'new_move': new_move,
            'num_pieces': sum(1 for t in self.square_trackers.values() if t.piece),
            'num_detections': len(detected_squares),
            'fen': self.board.fen(),
            'state_summary': state_summary,
            'pending_move': self.pending_move_san,
            'pending_count': self.pending_count,
            'cooldown': self.cooldown,
            'vacated_squares': [SQUARE_NAMES[s] for s in self.last_vacated],
            'arrived_squares': [SQUARE_NAMES[s] for s in self.last_arrived],
            'detections': self.raw_detections
        }
    
    def draw_debug_overlay(self, frame: np.ndarray, result: Dict = None) -> np.ndarray:
        """Draw comprehensive debug overlay."""
        vis = frame.copy()
        
        if self.corners is None or self.centers is None:
            return vis
        
        # Draw grid
        self._draw_grid(vis)
        
        # Draw detection boxes
        if result and 'detections' in result:
            self._draw_detections(vis, result['detections'])
        
        # Draw tracking state
        self._draw_tracking_state(vis)
        
        # Draw mini board
        self._draw_mini_board(vis)
        
        # Draw move info
        self._draw_move_info(vis, result)
        
        return vis
    
    def _draw_grid(self, frame: np.ndarray):
        """Draw 64-square grid."""
        if self.centers is None or self.boundary is None:
            return
        
        pts = self.boundary.astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        for i, name in enumerate(['a1', 'h1', 'h8', 'a8']):
            x, y = int(self.boundary[i][0]), int(self.boundary[i][1])
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(frame, name, (x + 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for i, name in enumerate(SQUARE_NAMES):
            cx, cy = int(self.centers[i][0]), int(self.centers[i][1])
            cv2.putText(frame, name, (cx - 8, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]):
        """Draw detection boxes."""
        for det in detections:
            box = det['box']
            piece = det['piece']
            conf = det['confidence']
            sq_name = det.get('square_name', '?')
            
            is_white = piece.isupper()
            color = (255, 255, 255) if is_white else (100, 100, 100)
            l, t, r, b = [int(v) for v in box]
            
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            
            label = f"{sq_name} {conf:.0%}"
            cv2.putText(frame, label, (l + 2, t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    def _draw_tracking_state(self, frame: np.ndarray):
        """Draw tracking indicators at each square."""
        if self.centers is None:
            return
        
        for sq, tracker in self.square_trackers.items():
            cx, cy = int(self.centers[sq][0]), int(self.centers[sq][1])
            
            if tracker.piece:
                rate = tracker.detection_rate()
                
                if sq in self.last_vacated:
                    # Red for vacated
                    cv2.circle(frame, (cx, cy - 15), 6, (0, 0, 255), -1)
                elif rate > 0.7:
                    # Green for well-tracked
                    cv2.circle(frame, (cx, cy - 15), 4, (0, 255, 0), -1)
                elif rate > 0.4:
                    # Yellow for uncertain
                    cv2.circle(frame, (cx, cy - 15), 4, (0, 255, 255), -1)
                else:
                    # Orange for low confidence
                    cv2.circle(frame, (cx, cy - 15), 4, (0, 165, 255), -1)
            
            elif sq in self.last_arrived:
                # Cyan for new arrival
                cv2.circle(frame, (cx, cy - 15), 6, (255, 255, 0), -1)
    
    def _draw_mini_board(self, frame: np.ndarray):
        """Draw mini board showing current state."""
        h, w = frame.shape[:2]
        size = min(180, w // 5)
        sq_size = size // 8
        ox, oy = w - size - 10, 10
        
        cv2.rectangle(frame, (ox - 2, oy - 2), (ox + size + 2, oy + size + 2), (40, 40, 40), -1)
        
        for rank in range(8):
            for file in range(8):
                x = ox + file * sq_size
                y = oy + (7 - rank) * sq_size
                is_light = (file + rank) % 2 == 1
                color = (240, 217, 181) if is_light else (181, 136, 99)
                cv2.rectangle(frame, (x, y), (x + sq_size, y + sq_size), color, -1)
        
        for sq, tracker in self.square_trackers.items():
            if tracker.piece is None:
                continue
            
            file_idx = sq % 8
            rank_idx = sq // 8
            x = ox + file_idx * sq_size + sq_size // 2
            y = oy + (7 - rank_idx) * sq_size + sq_size // 2
            
            is_white = tracker.is_white
            text_color = (255, 255, 255) if is_white else (0, 0, 0)
            bg_color = (0, 0, 0) if is_white else (255, 255, 255)
            
            cv2.circle(frame, (x, y), sq_size // 3, bg_color, -1)
            cv2.putText(frame, tracker.piece.upper(), (x - 4, y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
    
    def _draw_move_info(self, frame: np.ndarray, result: Dict = None):
        """Draw move info panel."""
        h, w = frame.shape[:2]
        
        # Info panel
        y_start = 200
        cv2.rectangle(frame, (w - 140, y_start - 25), (w - 5, y_start + 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 140, y_start - 25), (w - 5, y_start + 120), (100, 100, 100), 1)
        
        cv2.putText(frame, "Move Detection", (w - 135, y_start - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        y = y_start + 10
        
        # Pending move
        if self.pending_move_san:
            cv2.putText(frame, f"Pending: {self.pending_move_san}", (w - 135, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            y += 15
            cv2.putText(frame, f"Confirm: {self.pending_count}/{self.CONFIRM_FRAMES}", (w - 135, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
            y += 15
        
        # Cooldown
        if self.cooldown > 0:
            cv2.putText(frame, f"Cooldown: {self.cooldown}", (w - 135, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 255), 1)
            y += 15
        
        # Last moves
        y += 5
        cv2.putText(frame, "Moves:", (w - 135, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y += 15
        
        recent = self.moves[-4:] if len(self.moves) > 4 else self.moves
        for i, move in enumerate(recent):
            move_num = len(self.moves) - len(recent) + i + 1
            prefix = f"{(move_num + 1) // 2}." if move_num % 2 == 1 else "  "
            cv2.putText(frame, f"{prefix}{move}", (w - 135, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
            y += 12
    
    def generate_pgn(self) -> str:
        """Generate PGN from detected moves."""
        game = chess.pgn.Game()
        game.headers["Event"] = "Chess Video Analysis"
        game.headers["Site"] = "PositionBasedTracker"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "Player 1"
        game.headers["Black"] = "Player 2"
        game.headers["Result"] = "*"
        
        node = game
        temp_board = chess.Board()
        
        for san in self.moves:
            try:
                move = temp_board.parse_san(san)
                node = node.add_variation(move)
                temp_board.push(move)
            except:
                pass
        
        return str(game)
