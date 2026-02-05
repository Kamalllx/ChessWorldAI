"""
Chess Video Analyzer - Smart Tracking Module
Uses known initial state + detection confirmation + piece tracking

Key improvements over basic detection:
1. Initialize with KNOWN starting position (all 32 pieces)
2. Lock-in pieces once confirmed with threshold
3. Track movements rather than re-detecting every frame
4. Combine model detection with logical reasoning
"""

import numpy as np
import cv2
import chess
import chess.pgn
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import onnxruntime as ort
from pathlib import Path

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
    # White pieces (bottom rows)
    0: 'R', 1: 'N', 2: 'B', 3: 'Q', 4: 'K', 5: 'B', 6: 'N', 7: 'R',  # a1-h1
    8: 'P', 9: 'P', 10: 'P', 11: 'P', 12: 'P', 13: 'P', 14: 'P', 15: 'P',  # a2-h2
    # Black pieces (top rows)
    48: 'p', 49: 'p', 50: 'p', 51: 'p', 52: 'p', 53: 'p', 54: 'p', 55: 'p',  # a7-h7
    56: 'r', 57: 'n', 58: 'b', 59: 'q', 60: 'k', 61: 'b', 62: 'n', 63: 'r',  # a8-h8
}


class PieceTracker:
    """Tracks a single piece's state."""
    def __init__(self, square_idx: int, piece: str, confidence: float = 0.0):
        self.square_idx = square_idx
        self.piece = piece  # Label like 'K', 'p', etc.
        self.confidence = confidence
        self.locked = False  # Once locked, we trust this piece exists
        self.lock_frames = 0  # Frames piece has been consistently detected
        self.empty_frames = 0  # Frames square has been empty (piece moved?)
        self.last_seen_confidence = 0.0
    
    def is_white(self) -> bool:
        return self.piece.isupper()
    
    def get_piece_index(self) -> int:
        try:
            return LABELS.index(self.piece)
        except ValueError:
            return -1


class SmartChessAnalyzer:
    """
    Chess analyzer with smart tracking.
    Uses known initial state and tracks piece movements.
    """
    
    # Thresholds
    DETECTION_THRESHOLD = 0.10  # Very low - show all detections
    LOCK_THRESHOLD = 0.40  # Confidence to lock a piece
    LOCK_FRAMES = 3  # Frames piece must be seen to lock
    EMPTY_THRESHOLD = 0.15  # Below this = square is empty
    EMPTY_FRAMES = 5  # Frames empty before considering piece moved
    MOVE_CONFIRM_MS = 400  # Milliseconds to confirm a move
    
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
        self.trackers: Dict[int, PieceTracker] = {}  # square_idx -> PieceTracker
        self.raw_detections = []  # Latest raw detections for visualization
        
        # Game state
        self.board = chess.Board()
        self.moves = []
        self.pending_move = None
        self.pending_move_time = 0
        self.is_initialized = False
    
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
        self.trackers = {}
        self.raw_detections = []
        self.board = chess.Board()
        self.moves = []
        self.pending_move = None
        self.pending_move_time = 0
        self.is_initialized = False
    
    def initialize_with_starting_position(self):
        """Initialize trackers with known starting position."""
        self.trackers = {}
        for sq_idx, piece in STARTING_POSITION.items():
            self.trackers[sq_idx] = PieceTracker(sq_idx, piece, confidence=1.0)
            self.trackers[sq_idx].locked = True  # We KNOW these pieces are there
        self.is_initialized = True
        print(f"Initialized with {len(self.trackers)} pieces in starting position")
    
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
        """Postprocess to get ALL detections (very low threshold)."""
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
        
        # Scale to full image coordinates
        boxes = np.stack([l, t, r, b], axis=1)
        
        # Filter by very low threshold
        max_scores = np.max(scores, axis=1)
        mask = max_scores > self.DETECTION_THRESHOLD
        
        if not np.any(mask):
            return []
        
        boxes = boxes[mask]
        scores = scores[mask]
        classes = np.argmax(scores, axis=1)
        max_scores = max_scores[mask]
        
        # NMS
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
                'scores': scores[i].tolist()
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
            box = det['box']
            # Use bottom-center of box (where piece base is)
            cx = (box[0] + box[2]) / 2
            cy = box[3] - (box[2] - box[0]) / 3
            
            sq_idx = self._get_square_at(cx, cy)
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
        """
        Analyze initial frame combining detection with known starting position.
        Returns detailed info for user confirmation.
        """
        if self.corners is None:
            return {'success': False, 'message': 'Corners not set'}
        
        # Initialize with known starting position
        self.initialize_with_starting_position()
        
        # Run detection
        detections = self.detect_pieces_raw(frame)
        
        # Match detections to expected pieces
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
            
            # Update tracker if detection matches or improves
            if sq_idx in self.trackers:
                tracker = self.trackers[sq_idx]
                # If detection matches expected piece, boost confidence
                if detected_piece == tracker.piece:
                    tracker.confidence = max(tracker.confidence, confidence)
                    tracker.last_seen_confidence = confidence
            
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
        
        # Count stats
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
            'detection_results': detection_results,
            'all_trackers': {sq: {'piece': t.piece, 'locked': t.locked, 'conf': t.confidence} 
                            for sq, t in self.trackers.items()}
        }
    
    def update_trackers(self, detections: List[Dict], timestamp: float):
        """Update piece trackers based on detections."""
        # Build detection map: square -> best detection for that square
        detection_map = {}
        for det in detections:
            sq = det['square_idx']
            if sq < 0:
                continue
            if sq not in detection_map or det['confidence'] > detection_map[sq]['confidence']:
                detection_map[sq] = det
        
        # Update each tracker
        for sq_idx, tracker in list(self.trackers.items()):
            if sq_idx in detection_map:
                det = detection_map[sq_idx]
                # Detection on this square
                if det['piece'] == tracker.piece:
                    # Matches expected piece
                    tracker.confidence = max(tracker.confidence, det['confidence'])
                    tracker.last_seen_confidence = det['confidence']
                    tracker.lock_frames += 1
                    tracker.empty_frames = 0
                    
                    if not tracker.locked and tracker.lock_frames >= self.LOCK_FRAMES and tracker.confidence >= self.LOCK_THRESHOLD:
                        tracker.locked = True
                else:
                    # Different piece detected - could be wrong detection or capture
                    tracker.empty_frames += 1
            else:
                # No detection on this square
                tracker.empty_frames += 1
                tracker.last_seen_confidence = 0
        
        # Check for pieces that might have moved
        squares_possibly_empty = [sq for sq, t in self.trackers.items() if t.empty_frames >= self.EMPTY_FRAMES]
        
        return squares_possibly_empty
    
    def detect_move(self, timestamp: float) -> Optional[str]:
        """Try to detect a chess move from tracker states."""
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
        
        # Find squares that are now empty (piece left)
        empty_squares = [sq for sq, t in self.trackers.items() 
                        if t.empty_frames >= self.EMPTY_FRAMES]
        
        # Find squares where piece appeared
        # Check raw detections for new pieces
        occupied_by_detection = {}
        for det in self.raw_detections:
            sq = det['square_idx']
            if sq >= 0 and det['confidence'] > self.LOCK_THRESHOLD:
                if sq not in self.trackers or self.trackers[sq].empty_frames > 0:
                    occupied_by_detection[sq] = det
        
        # Score each legal move
        best_move = None
        best_score = 0
        
        for move in legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            score = 0
            
            # From square should be empty or emptying
            if from_sq in self.trackers and self.trackers[from_sq].empty_frames > 0:
                score += 1 + self.trackers[from_sq].empty_frames / 10
            
            # To square should have the moving piece
            if to_sq in occupied_by_detection:
                moving_piece = self.board.piece_at(from_sq)
                if moving_piece and occupied_by_detection[to_sq]['piece'] == moving_piece.symbol():
                    score += 2
            
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None or best_score < 1.5:
            return None
        
        # Pending move confirmation
        san = self.board.san(best_move)
        
        if self.pending_move != san:
            self.pending_move = san
            self.pending_move_time = timestamp
            return None
        
        if timestamp - self.pending_move_time < self.MOVE_CONFIRM_MS:
            return None
        
        # Confirm the move
        from_sq = best_move.from_square
        to_sq = best_move.to_square
        
        # Update trackers
        if from_sq in self.trackers:
            moving_piece = self.trackers[from_sq]
            del self.trackers[from_sq]
            
            # If capture, remove captured piece
            if to_sq in self.trackers:
                del self.trackers[to_sq]
            
            # Place piece at new square
            self.trackers[to_sq] = PieceTracker(to_sq, moving_piece.piece, moving_piece.confidence)
            self.trackers[to_sq].locked = True
        
        self.board.push(best_move)
        self.moves.append(san)
        self.pending_move = None
        
        return san
    
    def process_frame(self, frame: np.ndarray, timestamp: float = 0) -> Dict:
        """Process a single frame with smart tracking."""
        if self.corners is None or self.centers is None:
            return {'success': False, 'message': 'Corners not set'}
        
        if not self.is_initialized:
            self.initialize_with_starting_position()
        
        # Detect pieces
        detections = self.detect_pieces_raw(frame)
        
        # Update trackers
        self.update_trackers(detections, timestamp)
        
        # Try to detect move
        new_move = self.detect_move(timestamp)
        
        # Build state summary for visualization
        state_summary = {}
        for sq, tracker in self.trackers.items():
            state_summary[SQUARE_NAMES[sq]] = {
                'piece': tracker.piece,
                'confidence': tracker.confidence,
                'locked': tracker.locked,
                'empty_frames': tracker.empty_frames
            }
        
        return {
            'success': True,
            'message': f'Tracking {len(self.trackers)} pieces',
            'moves': self.moves.copy(),
            'new_move': new_move,
            'num_pieces': len(self.trackers),
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
        
        return vis
    
    def _draw_grid(self, frame: np.ndarray):
        """Draw 64-square grid."""
        if self.centers is None or self.boundary is None:
            return
        
        # Draw boundary
        pts = self.boundary.astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        # Draw corner labels
        for i, name in enumerate(['a1', 'h1', 'h8', 'a8']):
            x, y = int(self.boundary[i][0]), int(self.boundary[i][1])
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(frame, name, (x + 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw square centers with names
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
                # Green dot for locked pieces
                cv2.circle(frame, (cx, cy - 15), 4, (0, 255, 0), -1)
            elif info['empty_frames'] > 0:
                # Red dot for possibly moved
                cv2.circle(frame, (cx, cy - 15), 4, (0, 0, 255), -1)
    
    def _draw_mini_board(self, frame: np.ndarray):
        """Draw mini board showing tracked pieces."""
        h, w = frame.shape[:2]
        size = min(180, w // 5)
        sq = size // 8
        ox, oy = w - size - 10, 10
        
        # Background
        cv2.rectangle(frame, (ox - 2, oy - 2), (ox + size + 2, oy + size + 2), (40, 40, 40), -1)
        
        # Squares
        for rank in range(8):
            for file in range(8):
                x = ox + file * sq
                y = oy + (7 - rank) * sq
                is_light = (file + rank) % 2 == 1
                color = (240, 217, 181) if is_light else (181, 136, 99)
                cv2.rectangle(frame, (x, y), (x + sq, y + sq), color, -1)
        
        # Pieces
        symbols = {'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
                   'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙'}
        
        for sq_idx, tracker in self.trackers.items():
            file_idx = sq_idx % 8
            rank_idx = sq_idx // 8
            x = ox + file_idx * sq + sq // 2
            y = oy + (7 - rank_idx) * sq + sq // 2
            
            piece = tracker.piece
            is_white = piece.isupper()
            text_color = (255, 255, 255) if is_white else (0, 0, 0)
            
            cv2.circle(frame, (x, y), sq // 3, (0, 0, 0) if is_white else (255, 255, 255), -1)
            cv2.putText(frame, piece.upper(), (x - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
    
    def generate_pgn(self) -> str:
        """Generate PGN from detected moves."""
        game = chess.pgn.Game()
        game.headers["Event"] = "Chess Video Analysis"
        game.headers["Site"] = "SmartChessAnalyzer"
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
