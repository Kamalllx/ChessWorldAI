"""
Chess Video Analyzer - Hybrid Tracking Module
Combines:
1. Known starting position (32 pieces)
2. ONNX model detection for confirmation
3. Visual tracking (bounding box tracking) for movement
4. Chess logic for move validation (castling, en passant, captures)
"""

import numpy as np
import cv2
import chess
import chess.pgn
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import onnxruntime as ort
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque

# Constants
MODEL_WIDTH = 480
MODEL_HEIGHT = 288
SQUARE_SIZE = 128
BOARD_SIZE = 8 * SQUARE_SIZE

# Piece labels - lowercase=black, uppercase=white
LABELS = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]

# Square names
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

# Known starting position - maps square index to piece label
STARTING_POSITION = {
    0: 'R', 1: 'N', 2: 'B', 3: 'Q', 4: 'K', 5: 'B', 6: 'N', 7: 'R',
    8: 'P', 9: 'P', 10: 'P', 11: 'P', 12: 'P', 13: 'P', 14: 'P', 15: 'P',
    48: 'p', 49: 'p', 50: 'p', 51: 'p', 52: 'p', 53: 'p', 54: 'p', 55: 'p',
    56: 'r', 57: 'n', 58: 'b', 59: 'q', 60: 'k', 61: 'b', 62: 'n', 63: 'r',
}


@dataclass
class TrackedPiece:
    """A tracked piece with visual and logical state."""
    piece_id: int  # Unique ID for this piece
    piece_type: str  # 'K', 'Q', 'R', 'B', 'N', 'P' or lowercase for black
    square_idx: int  # Current square (0-63)
    
    # Visual tracking
    bbox: List[float] = field(default_factory=list)  # [l, t, r, b]
    center: Tuple[float, float] = (0, 0)
    
    # Confidence
    detection_conf: float = 0.0
    tracking_conf: float = 1.0  # Visual tracking confidence
    
    # State
    is_captured: bool = False
    last_seen_frame: int = 0
    frames_not_seen: int = 0
    
    # Movement tracking
    prev_center: Tuple[float, float] = (0, 0)
    is_moving: bool = False
    movement_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def is_white(self) -> bool:
        return self.piece_type.isupper()
    
    def update_center(self, bbox: List[float], frame_num: int):
        """Update piece center from bounding box."""
        self.prev_center = self.center
        cx = (bbox[0] + bbox[2]) / 2
        cy = bbox[3] - (bbox[2] - bbox[0]) / 3  # Use bottom-center
        self.center = (cx, cy)
        self.bbox = bbox
        self.last_seen_frame = frame_num
        self.frames_not_seen = 0
        
        # Track movement
        if self.prev_center != (0, 0):
            dist = np.sqrt((self.center[0] - self.prev_center[0])**2 + 
                          (self.center[1] - self.prev_center[1])**2)
            self.movement_history.append(dist)
            self.is_moving = dist > 5  # Threshold for movement


class HybridChessAnalyzer:
    """
    Hybrid Chess Analyzer combining:
    - Known starting position
    - ONNX model detection
    - Visual bounding box tracking
    - Chess logic validation
    """
    
    # Thresholds
    DETECTION_THRESHOLD = 0.08  # Very low for maximum recall
    MATCH_THRESHOLD = 0.25  # To match detection to tracked piece
    MOVE_CONFIRM_FRAMES = 3  # Frames to confirm a move
    SQUARE_EMPTY_FRAMES = 4  # Frames before considering square empty
    
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
        self.square_size_pixels = 0  # Average square size in image pixels
        
        # Tracking state
        self.tracked_pieces: Dict[int, TrackedPiece] = {}  # piece_id -> TrackedPiece
        self.square_to_piece: Dict[int, int] = {}  # square_idx -> piece_id
        self.next_piece_id = 0
        
        # Detection history
        self.raw_detections = []
        self.frame_count = 0
        
        # Game state
        self.board = chess.Board()
        self.moves = []
        self.pending_move = None
        self.pending_move_frame = 0
        self.is_initialized = False
        
        # Move detection state
        self.squares_changed: Dict[int, int] = {}  # square_idx -> frames_changed
        self.last_move_frame = 0
    
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
        self.tracked_pieces = {}
        self.square_to_piece = {}
        self.next_piece_id = 0
        self.raw_detections = []
        self.frame_count = 0
        self.board = chess.Board()
        self.moves = []
        self.pending_move = None
        self.pending_move_frame = 0
        self.is_initialized = False
        self.squares_changed = {}
        self.last_move_frame = 0
    
    def initialize_with_starting_position(self):
        """Initialize all 32 pieces in starting position."""
        self.tracked_pieces = {}
        self.square_to_piece = {}
        
        for sq_idx, piece_type in STARTING_POSITION.items():
            piece_id = self.next_piece_id
            self.next_piece_id += 1
            
            # Get center for this square
            center = (0, 0)
            if self.centers is not None:
                center = (float(self.centers[sq_idx][0]), float(self.centers[sq_idx][1]))
            
            tracked = TrackedPiece(
                piece_id=piece_id,
                piece_type=piece_type,
                square_idx=sq_idx,
                center=center,
                tracking_conf=1.0,
                last_seen_frame=0
            )
            
            self.tracked_pieces[piece_id] = tracked
            self.square_to_piece[sq_idx] = piece_id
        
        self.is_initialized = True
        print(f"Initialized with {len(self.tracked_pieces)} pieces in starting position")
    
    def set_corners(self, corners: Dict[str, List[float]]):
        """Set board corners and compute transforms."""
        self.corners = corners
        self._compute_transforms()
    
    def _compute_transforms(self):
        if self.corners is None:
            return
        
        src = np.array([
            self.corners['a1'],
            self.corners['h1'],
            self.corners['h8'],
            self.corners['a8']
        ], dtype=np.float32)
        
        dst = np.array([
            [0, BOARD_SIZE],
            [BOARD_SIZE, BOARD_SIZE],
            [BOARD_SIZE, 0],
            [0, 0]
        ], dtype=np.float32)
        
        self.transform = cv2.getPerspectiveTransform(src, dst)
        self.inv_transform = cv2.getPerspectiveTransform(dst, src)
        
        # Compute square centers in image coordinates
        centers_board = []
        for rank in range(8):
            for file in range(8):
                x = (file + 0.5) * SQUARE_SIZE
                y = (7 - rank + 0.5) * SQUARE_SIZE
                centers_board.append([x, y])
        
        centers_board = np.array(centers_board, dtype=np.float32).reshape(-1, 1, 2)
        self.centers = cv2.perspectiveTransform(centers_board, self.inv_transform).reshape(-1, 2)
        self.boundary = src.copy()
        
        # Estimate square size in pixels (average)
        edge1 = np.linalg.norm(src[1] - src[0])  # a1 to h1
        edge2 = np.linalg.norm(src[3] - src[0])  # a1 to a8
        self.square_size_pixels = (edge1 + edge2) / 16
        
        # Update piece centers if already initialized
        if self.is_initialized:
            for piece_id, piece in self.tracked_pieces.items():
                sq_idx = piece.square_idx
                piece.center = (float(self.centers[sq_idx][0]), float(self.centers[sq_idx][1]))
    
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
    
    def postprocess_detections(self, output: np.ndarray, meta: Dict) -> List[Dict]:
        """Postprocess to get ALL detections."""
        preds = np.transpose(output, (0, 2, 1))[0]
        
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
        
        max_scores = np.max(scores, axis=1)
        mask = max_scores > self.DETECTION_THRESHOLD
        
        if not np.any(mask):
            return []
        
        boxes = boxes[mask]
        scores = scores[mask]
        classes = np.argmax(scores, axis=1)
        max_scores = max_scores[mask]
        
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), max_scores.tolist(), self.DETECTION_THRESHOLD, 0.4)
        
        if len(indices) == 0:
            return []
        
        indices = np.array(indices).flatten()
        
        detections = []
        for i in indices:
            det = {
                'box': boxes[i].tolist(),
                'class_idx': int(classes[i]),
                'piece': LABELS[classes[i]],
                'confidence': float(max_scores[i]),
                'center': ((boxes[i][0] + boxes[i][2]) / 2, 
                          boxes[i][3] - (boxes[i][2] - boxes[i][0]) / 3)
            }
            detections.append(det)
        
        return detections
    
    def detect_pieces_raw(self, frame: np.ndarray) -> List[Dict]:
        """Run piece detection and return raw results."""
        if self.corners is None:
            return []
        
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
        
        detections = self.postprocess_detections(output[0], meta)
        
        # Assign each detection to a square
        for det in detections:
            sq_idx = self._get_square_at(*det['center'])
            det['square_idx'] = sq_idx
            det['square_name'] = SQUARE_NAMES[sq_idx] if sq_idx >= 0 else 'outside'
        
        self.raw_detections = detections
        return detections
    
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
    
    def analyze_initial_frame(self, frame: np.ndarray) -> Dict:
        """Analyze initial frame."""
        if self.corners is None:
            return {'success': False, 'message': 'Corners not set'}
        
        self.initialize_with_starting_position()
        detections = self.detect_pieces_raw(frame)
        
        # Match detections to expected pieces and update bbox
        detection_results = []
        matched_squares = set()
        
        for det in detections:
            sq_idx = det['square_idx']
            if sq_idx < 0:
                continue
            
            expected_piece = STARTING_POSITION.get(sq_idx)
            detected_piece = det['piece']
            confidence = det['confidence']
            
            result = {
                'square_idx': sq_idx,
                'square_name': SQUARE_NAMES[sq_idx],
                'detected_piece': detected_piece,
                'expected_piece': expected_piece,
                'confidence': confidence,
                'box': det['box'],
                'match': expected_piece == detected_piece if expected_piece else (expected_piece is None)
            }
            
            # Update tracked piece with initial bbox
            if sq_idx in self.square_to_piece:
                piece_id = self.square_to_piece[sq_idx]
                piece = self.tracked_pieces[piece_id]
                piece.bbox = det['box']
                piece.update_center(det['box'], 0)
                piece.detection_conf = confidence
            
            detection_results.append(result)
            matched_squares.add(sq_idx)
        
        # Add expected pieces that weren't detected
        for sq_idx, piece in STARTING_POSITION.items():
            if sq_idx not in matched_squares:
                detection_results.append({
                    'square_idx': sq_idx,
                    'square_name': SQUARE_NAMES[sq_idx],
                    'detected_piece': None,
                    'expected_piece': piece,
                    'confidence': 0.0,
                    'box': None,
                    'match': False
                })
        
        total_expected = len(STARTING_POSITION)
        detected_correct = sum(1 for r in detection_results if r['match'] and r['detected_piece'])
        detected_wrong = sum(1 for r in detection_results if not r['match'] and r['detected_piece'])
        not_detected = sum(1 for r in detection_results if r['expected_piece'] and not r['detected_piece'])
        
        return {
            'success': True,
            'message': f'Expected {total_expected} pieces, detected {detected_correct} correctly',
            'num_expected': total_expected,
            'num_detected_correct': detected_correct,
            'num_detected_wrong': detected_wrong,
            'num_not_detected': not_detected,
            'detection_results': detection_results
        }
    
    def match_detections_to_pieces(self, detections: List[Dict]) -> Dict[int, Dict]:
        """Match detections to tracked pieces using position and type."""
        matches = {}  # piece_id -> detection
        used_detections = set()
        
        # For each tracked piece, find best matching detection
        for piece_id, piece in self.tracked_pieces.items():
            if piece.is_captured:
                continue
            
            best_det = None
            best_score = 0
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                
                # Calculate match score
                score = 0
                
                # Type match (most important)
                if det['piece'] == piece.piece_type:
                    score += 2.0
                elif det['piece'].upper() == piece.piece_type.upper():
                    # Same piece type, wrong color - less likely
                    score += 0.5
                
                # Position proximity
                dist = np.sqrt((det['center'][0] - piece.center[0])**2 + 
                              (det['center'][1] - piece.center[1])**2)
                
                # Allow larger distance for movement
                max_dist = self.square_size_pixels * 2 if self.square_size_pixels > 0 else 100
                if dist < max_dist:
                    score += 1.0 * (1 - dist / max_dist)
                
                # Confidence bonus
                score += det['confidence'] * 0.3
                
                if score > best_score and score > self.MATCH_THRESHOLD:
                    best_score = score
                    best_det = (i, det)
            
            if best_det:
                used_detections.add(best_det[0])
                matches[piece_id] = best_det[1]
        
        return matches
    
    def update_tracking(self, detections: List[Dict]) -> List[int]:
        """Update piece tracking with new detections. Returns list of changed squares."""
        self.frame_count += 1
        changed_squares = []
        
        # Match detections to pieces
        matches = self.match_detections_to_pieces(detections)
        
        # Update matched pieces
        for piece_id, det in matches.items():
            piece = self.tracked_pieces[piece_id]
            piece.update_center(det['box'], self.frame_count)
            piece.detection_conf = det['confidence']
            
            # Check if piece moved to different square
            new_sq = det['square_idx']
            if new_sq >= 0 and new_sq != piece.square_idx:
                changed_squares.append(piece.square_idx)
                changed_squares.append(new_sq)
        
        # Track pieces not detected
        for piece_id, piece in self.tracked_pieces.items():
            if piece_id not in matches and not piece.is_captured:
                piece.frames_not_seen += 1
                
                if piece.frames_not_seen >= self.SQUARE_EMPTY_FRAMES:
                    changed_squares.append(piece.square_idx)
        
        return list(set(changed_squares))
    
    def detect_move(self) -> Optional[str]:
        """Detect chess move from tracking state."""
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
        
        # Don't detect moves too frequently
        if self.frame_count - self.last_move_frame < self.MOVE_CONFIRM_FRAMES:
            return None
        
        # Find squares where pieces might have moved
        from_candidates = []  # Squares that might now be empty
        to_candidates = []    # Squares that might now have a piece
        
        for piece_id, piece in self.tracked_pieces.items():
            if piece.is_captured:
                continue
            
            # Piece not seen = might have moved away
            if piece.frames_not_seen >= self.SQUARE_EMPTY_FRAMES:
                from_candidates.append((piece.square_idx, piece))
            
            # Check if piece bbox moved to different square
            if piece.bbox:
                current_sq = self._get_square_at(*piece.center)
                if current_sq >= 0 and current_sq != piece.square_idx:
                    to_candidates.append((current_sq, piece))
        
        # Also check raw detections for pieces on new squares
        for det in self.raw_detections:
            sq_idx = det['square_idx']
            if sq_idx >= 0 and sq_idx not in self.square_to_piece:
                # Detection on empty square
                to_candidates.append((sq_idx, det))
        
        # Score legal moves
        best_move = None
        best_score = 0
        
        for move in legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            score = 0
            
            # From square should show piece leaving
            for sq, _ in from_candidates:
                if sq == from_sq:
                    score += 2
                    break
            
            # To square should show piece arriving
            for sq, info in to_candidates:
                if sq == to_sq:
                    score += 2
                    # Bonus if piece type matches
                    moving_piece = self.board.piece_at(from_sq)
                    if moving_piece:
                        if isinstance(info, TrackedPiece):
                            if info.piece_type == moving_piece.symbol():
                                score += 1
                        elif isinstance(info, dict) and info.get('piece') == moving_piece.symbol():
                            score += 1
                    break
            
            # Special moves get bonus
            if self.board.is_castling(move):
                score += 0.5
            if self.board.is_en_passant(move):
                score += 0.5
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None or best_score < 3:
            return None
        
        # Pending move confirmation
        san = self.board.san(best_move)
        
        if self.pending_move != san:
            self.pending_move = san
            self.pending_move_frame = self.frame_count
            return None
        
        if self.frame_count - self.pending_move_frame < self.MOVE_CONFIRM_FRAMES:
            return None
        
        # Execute the move
        return self._execute_move(best_move)
    
    def _execute_move(self, move: chess.Move) -> str:
        """Execute a chess move, handling all special cases."""
        san = self.board.san(move)
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Handle castling
        if self.board.is_castling(move):
            self._handle_castling(move)
        # Handle en passant
        elif self.board.is_en_passant(move):
            self._handle_en_passant(move)
        # Handle capture
        elif self.board.is_capture(move):
            self._handle_capture(move)
        # Normal move
        else:
            self._handle_normal_move(move)
        
        # Handle promotion
        if move.promotion:
            self._handle_promotion(move)
        
        self.board.push(move)
        self.moves.append(san)
        self.pending_move = None
        self.last_move_frame = self.frame_count
        
        return san
    
    def _handle_normal_move(self, move: chess.Move):
        """Handle a normal non-capture move."""
        from_sq = move.from_square
        to_sq = move.to_square
        
        if from_sq in self.square_to_piece:
            piece_id = self.square_to_piece[from_sq]
            piece = self.tracked_pieces[piece_id]
            
            # Update piece position
            piece.square_idx = to_sq
            
            # Update mapping
            del self.square_to_piece[from_sq]
            self.square_to_piece[to_sq] = piece_id
            
            # Update center to new square
            if self.centers is not None:
                piece.center = (float(self.centers[to_sq][0]), float(self.centers[to_sq][1]))
            
            piece.frames_not_seen = 0
    
    def _handle_capture(self, move: chess.Move):
        """Handle a capture."""
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Mark captured piece
        if to_sq in self.square_to_piece:
            captured_id = self.square_to_piece[to_sq]
            self.tracked_pieces[captured_id].is_captured = True
            del self.square_to_piece[to_sq]
        
        # Move capturing piece
        if from_sq in self.square_to_piece:
            piece_id = self.square_to_piece[from_sq]
            piece = self.tracked_pieces[piece_id]
            
            piece.square_idx = to_sq
            del self.square_to_piece[from_sq]
            self.square_to_piece[to_sq] = piece_id
            
            if self.centers is not None:
                piece.center = (float(self.centers[to_sq][0]), float(self.centers[to_sq][1]))
            
            piece.frames_not_seen = 0
    
    def _handle_castling(self, move: chess.Move):
        """Handle castling - move both king and rook."""
        from_sq = move.from_square
        to_sq = move.to_square
        
        is_kingside = to_sq > from_sq
        
        if self.board.turn == chess.WHITE:
            if is_kingside:  # O-O
                king_from, king_to = chess.E1, chess.G1
                rook_from, rook_to = chess.H1, chess.F1
            else:  # O-O-O
                king_from, king_to = chess.E1, chess.C1
                rook_from, rook_to = chess.A1, chess.D1
        else:
            if is_kingside:  # O-O
                king_from, king_to = chess.E8, chess.G8
                rook_from, rook_to = chess.H8, chess.F8
            else:  # O-O-O
                king_from, king_to = chess.E8, chess.C8
                rook_from, rook_to = chess.A8, chess.D8
        
        # Move king
        if king_from in self.square_to_piece:
            king_id = self.square_to_piece[king_from]
            self.tracked_pieces[king_id].square_idx = king_to
            del self.square_to_piece[king_from]
            self.square_to_piece[king_to] = king_id
            if self.centers is not None:
                self.tracked_pieces[king_id].center = (
                    float(self.centers[king_to][0]), 
                    float(self.centers[king_to][1])
                )
        
        # Move rook
        if rook_from in self.square_to_piece:
            rook_id = self.square_to_piece[rook_from]
            self.tracked_pieces[rook_id].square_idx = rook_to
            del self.square_to_piece[rook_from]
            self.square_to_piece[rook_to] = rook_id
            if self.centers is not None:
                self.tracked_pieces[rook_id].center = (
                    float(self.centers[rook_to][0]), 
                    float(self.centers[rook_to][1])
                )
    
    def _handle_en_passant(self, move: chess.Move):
        """Handle en passant capture."""
        from_sq = move.from_square
        to_sq = move.to_square
        
        # The captured pawn is on a different square
        if self.board.turn == chess.WHITE:
            captured_sq = to_sq - 8  # One rank below
        else:
            captured_sq = to_sq + 8  # One rank above
        
        # Mark captured pawn
        if captured_sq in self.square_to_piece:
            captured_id = self.square_to_piece[captured_sq]
            self.tracked_pieces[captured_id].is_captured = True
            del self.square_to_piece[captured_sq]
        
        # Move capturing pawn
        if from_sq in self.square_to_piece:
            piece_id = self.square_to_piece[from_sq]
            piece = self.tracked_pieces[piece_id]
            
            piece.square_idx = to_sq
            del self.square_to_piece[from_sq]
            self.square_to_piece[to_sq] = piece_id
            
            if self.centers is not None:
                piece.center = (float(self.centers[to_sq][0]), float(self.centers[to_sq][1]))
    
    def _handle_promotion(self, move: chess.Move):
        """Handle pawn promotion."""
        to_sq = move.to_square
        if to_sq in self.square_to_piece:
            piece_id = self.square_to_piece[to_sq]
            # Update piece type to promoted piece
            promo_piece = chess.piece_symbol(move.promotion)
            if self.tracked_pieces[piece_id].is_white():
                self.tracked_pieces[piece_id].piece_type = promo_piece.upper()
            else:
                self.tracked_pieces[piece_id].piece_type = promo_piece.lower()
    
    def process_frame(self, frame: np.ndarray, timestamp: float = 0) -> Dict:
        """Process a single frame with hybrid tracking."""
        if self.corners is None or self.centers is None:
            return {'success': False, 'message': 'Corners not set'}
        
        if not self.is_initialized:
            self.initialize_with_starting_position()
        
        # Detect pieces
        detections = self.detect_pieces_raw(frame)
        
        # Update tracking
        changed_squares = self.update_tracking(detections)
        
        # Try to detect move
        new_move = self.detect_move()
        
        # Build state summary
        state_summary = {}
        for piece_id, piece in self.tracked_pieces.items():
            if not piece.is_captured:
                sq_name = SQUARE_NAMES[piece.square_idx]
                state_summary[sq_name] = {
                    'piece': piece.piece_type,
                    'confidence': piece.detection_conf,
                    'locked': piece.frames_not_seen == 0,
                    'empty_frames': piece.frames_not_seen
                }
        
        return {
            'success': True,
            'message': f'Tracking {sum(1 for p in self.tracked_pieces.values() if not p.is_captured)} pieces',
            'moves': self.moves.copy(),
            'new_move': new_move,
            'num_pieces': sum(1 for p in self.tracked_pieces.values() if not p.is_captured),
            'num_detections': len(detections),
            'fen': self.board.fen(),
            'state_summary': state_summary,
            'detections': [{
                'box': d['box'],
                'piece': d['piece'],
                'confidence': d['confidence'],
                'square_idx': d['square_idx'],
                'square_name': d['square_name']
            } for d in detections]
        }
    
    def draw_debug_overlay(self, frame: np.ndarray, result: Dict = None) -> np.ndarray:
        """Draw comprehensive debug overlay."""
        vis = frame.copy()
        
        if self.corners is None or self.centers is None:
            return vis
        
        # Draw grid
        self._draw_grid(vis)
        
        # Draw detections
        if result and 'detections' in result:
            self._draw_detections(vis, result['detections'])
        
        # Draw tracker state
        self._draw_tracker_state(vis, result.get('state_summary', {}) if result else {})
        
        # Draw mini board
        self._draw_mini_board(vis)
        
        # Draw move info
        if self.moves:
            self._draw_move_info(vis)
        
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
            file_idx, rank_idx = i % 8, i // 8
            is_light = (file_idx + rank_idx) % 2 == 1
            
            color = (200, 200, 200) if is_light else (100, 100, 100)
            cv2.circle(frame, (cx, cy), 2, color, -1)
            cv2.putText(frame, name, (cx - 8, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict]):
        """Draw detection boxes."""
        piece_colors = {
            'b': (139, 69, 19), 'k': (0, 0, 139), 'n': (101, 67, 33),
            'p': (80, 80, 80), 'q': (128, 0, 128), 'r': (0, 0, 200),
            'B': (0, 215, 255), 'K': (255, 255, 255), 'N': (32, 165, 218),
            'P': (200, 200, 200), 'Q': (255, 0, 255), 'R': (0, 100, 255)
        }
        
        for det in detections:
            box = det['box']
            piece = det['piece']
            conf = det['confidence']
            sq_name = det.get('square_name', '?')
            
            color = piece_colors.get(piece, (0, 255, 0))
            l, t, r, b = [int(v) for v in box]
            
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            
            label = f"{piece}@{sq_name} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (l, t - lh - 4), (l + lw + 4, t), color, -1)
            cv2.putText(frame, label, (l + 2, t - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    def _draw_tracker_state(self, frame: np.ndarray, state_summary: Dict):
        """Draw tracker indicators at each square."""
        if not state_summary or self.centers is None:
            return
        
        for sq_name, info in state_summary.items():
            sq_idx = SQUARE_NAMES.index(sq_name)
            cx, cy = int(self.centers[sq_idx][0]), int(self.centers[sq_idx][1])
            
            if info['locked']:
                cv2.circle(frame, (cx, cy - 15), 4, (0, 255, 0), -1)
            elif info['empty_frames'] > 0:
                cv2.circle(frame, (cx, cy - 15), 4, (0, 0, 255), -1)
    
    def _draw_mini_board(self, frame: np.ndarray):
        """Draw mini board showing tracked pieces."""
        h, w = frame.shape[:2]
        size = min(180, w // 5)
        sq = size // 8
        ox, oy = w - size - 10, 10
        
        cv2.rectangle(frame, (ox - 2, oy - 2), (ox + size + 2, oy + size + 2), (40, 40, 40), -1)
        
        for rank in range(8):
            for file in range(8):
                x = ox + file * sq
                y = oy + (7 - rank) * sq
                is_light = (file + rank) % 2 == 1
                color = (240, 217, 181) if is_light else (181, 136, 99)
                cv2.rectangle(frame, (x, y), (x + sq, y + sq), color, -1)
        
        for piece_id, piece in self.tracked_pieces.items():
            if piece.is_captured:
                continue
            
            file_idx = piece.square_idx % 8
            rank_idx = piece.square_idx // 8
            x = ox + file_idx * sq + sq // 2
            y = oy + (7 - rank_idx) * sq + sq // 2
            
            is_white = piece.is_white()
            text_color = (255, 255, 255) if is_white else (0, 0, 0)
            
            cv2.circle(frame, (x, y), sq // 3, (0, 0, 0) if is_white else (255, 255, 255), -1)
            cv2.putText(frame, piece.piece_type.upper(), (x - 4, y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
    
    def _draw_move_info(self, frame: np.ndarray):
        """Draw last few moves on screen."""
        h, w = frame.shape[:2]
        
        # Show last 5 moves
        recent = self.moves[-5:] if len(self.moves) > 5 else self.moves
        
        y = 200
        cv2.rectangle(frame, (w - 120, y - 20), (w - 10, y + len(recent) * 20 + 5), (0, 0, 0), -1)
        cv2.putText(frame, "Moves:", (w - 115, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        for i, move in enumerate(recent):
            move_num = len(self.moves) - len(recent) + i + 1
            prefix = f"{(move_num + 1) // 2}." if move_num % 2 == 1 else "  "
            cv2.putText(frame, f"{prefix}{move}", (w - 115, y + 20 + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    def generate_pgn(self) -> str:
        """Generate PGN from detected moves."""
        game = chess.pgn.Game()
        game.headers["Event"] = "Chess Video Analysis"
        game.headers["Site"] = "HybridChessAnalyzer"
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
