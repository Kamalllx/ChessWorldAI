"""
Chess Video Analyzer - Core Module
Exact port of the React CameraChessWeb app's functionality to Python

Key functions from React:
- getInput() -> preprocess_image()
- getBoxesAndScores() -> postprocess_detections()
- detect() -> detect_pieces()
- findPieces loop -> process_frame()
- calculateScore() + processState() -> detect_move()
"""

import numpy as np
import cv2
import chess
import chess.pgn
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import onnxruntime as ort
from pathlib import Path

# Constants matching React constants.tsx
MODEL_WIDTH = 480
MODEL_HEIGHT = 288
SQUARE_SIZE = 128
BOARD_SIZE = 8 * SQUARE_SIZE

# Piece labels - lowercase=black, uppercase=white
LABELS = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]

# Square names in order (a1=0, b1=1, ... h8=63)
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


def get_piece_index(piece: chess.Piece) -> int:
    """Convert chess.Piece to label index (0-11)."""
    symbol = piece.symbol()
    try:
        return LABELS.index(symbol)
    except ValueError:
        return -1


class ChessAnalyzer:
    """
    Main class for analyzing chess videos.
    Exact port of React CameraChessWeb functionality.
    """
    
    def __init__(self, pieces_model_path: str, corners_model_path: str):
        """Initialize the analyzer with model paths."""
        self.pieces_model_path = Path(pieces_model_path)
        self.corners_model_path = Path(corners_model_path)
        
        # Load ONNX models
        self._load_models()
        
        # State for video processing
        self.corners = None
        self.centers = None  # 64x2 array of square centers in image coords
        self.boundary = None  # 4x2 array of board corners
        self.inv_transform = None
        
        # Game state
        self.board = chess.Board()
        self.moves = []
        self.state = np.zeros((64, 12), dtype=np.float32)  # Detection state
        self.greedy_move_times = {}  # For move confirmation timing
        self.last_move_lan = None
        
    def _load_models(self):
        """Load ONNX models for piece detection."""
        pieces_onnx = self.pieces_model_path / "model.onnx"
        corners_onnx = self.corners_model_path / "model.onnx"
        
        if not pieces_onnx.exists():
            raise FileNotFoundError(f"Pieces model not found: {pieces_onnx}")
        if not corners_onnx.exists():
            raise FileNotFoundError(f"Corners model not found: {corners_onnx}")
        
        print("Loading ONNX models...")
        self.pieces_session = ort.InferenceSession(
            str(pieces_onnx),
            providers=['CPUExecutionProvider']
        )
        self.corners_session = ort.InferenceSession(
            str(corners_onnx),
            providers=['CPUExecutionProvider']
        )
        
        # Get input info
        pieces_input = self.pieces_session.get_inputs()[0]
        print(f"  Pieces model input: {pieces_input.name}, shape: {pieces_input.shape}")
        
        corners_input = self.corners_session.get_inputs()[0]
        print(f"  Corners model input: {corners_input.name}, shape: {corners_input.shape}")
        
        print("Models loaded successfully")
    
    def preprocess_image(self, frame: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for model input.
        Matches React's getInput() function exactly.
        
        Args:
            frame: BGR image (H, W, 3)
            keypoints: Optional 4x2 array of corner points in MODEL coordinates
        
        Returns:
            input_tensor: (1, 3, 288, 480) float32
            meta: dict with preprocessing metadata
        """
        h, w = frame.shape[:2]
        
        # Calculate ROI based on keypoints
        if keypoints is not None and len(keypoints) >= 4:
            xs = keypoints[:, 0]
            ys = keypoints[:, 1]
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            
            # Padding (1/12 of bbox size)
            padding_ratio = 12
            pad_left = bbox_width / padding_ratio
            pad_right = bbox_width / padding_ratio
            pad_top = bbox_height / padding_ratio
            pad_bottom = bbox_height / padding_ratio
            
            # Adjust padding to maintain aspect ratio
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
            
            # Convert to image coordinates
            roi = [
                max(0, int((xmin - pad_left) * w / MODEL_WIDTH)),
                max(0, int((ymin - pad_top) * h / MODEL_HEIGHT)),
                min(w, int((xmax + pad_right) * w / MODEL_WIDTH)),
                min(h, int((ymax + pad_bottom) * h / MODEL_HEIGHT))
            ]
        else:
            roi = [0, 0, w, h]
        
        # Crop
        cropped = frame[roi[1]:roi[3], roi[0]:roi[2]]
        crop_h, crop_w = cropped.shape[:2]
        
        if crop_w == 0 or crop_h == 0:
            cropped = frame
            crop_h, crop_w = frame.shape[:2]
            roi = [0, 0, w, h]
        
        # Calculate resize dimensions maintaining aspect ratio
        ratio = crop_h / crop_w
        desired_ratio = MODEL_HEIGHT / MODEL_WIDTH
        
        if ratio > desired_ratio:
            resize_h = MODEL_HEIGHT
            resize_w = int(MODEL_HEIGHT / ratio)
        else:
            resize_w = MODEL_WIDTH
            resize_h = int(MODEL_WIDTH * ratio)
        
        # Resize
        resized = cv2.resize(cropped, (resize_w, resize_h))
        
        # Calculate padding
        dx = MODEL_WIDTH - resize_w
        dy = MODEL_HEIGHT - resize_h
        pad_right = dx // 2
        pad_left = dx - pad_right
        pad_bottom = dy // 2
        pad_top = dy - pad_bottom
        
        # Pad with gray (114)
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose HWC to CHW and add batch dimension
        transposed = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(transposed, axis=0)
        
        meta = {
            'roi': roi,
            'crop_w': crop_w,
            'crop_h': crop_h,
            'resize_w': resize_w,
            'resize_h': resize_h,
            'pad_left': pad_left,
            'pad_right': pad_right,
            'pad_top': pad_top,
            'pad_bottom': pad_bottom,
            'original_w': w,
            'original_h': h
        }
        
        return input_tensor, meta
    
    def postprocess_detections(self, output: np.ndarray, meta: Dict,
                               conf_threshold: float = 0.25) -> Dict:
        """
        Postprocess model output to get bounding boxes and scores.
        Matches React's getBoxesAndScores() function.
        
        Returns dict with:
            boxes: Nx4 array (l, t, r, b) in MODEL coordinates
            scores: Nx12 array of class scores
            classes: N array of predicted classes
        """
        # Output shape: (1, 16, 2100) -> transpose to (1, 2100, 16)
        preds = np.transpose(output, (0, 2, 1))
        preds = preds[0]  # Remove batch dimension: (2100, 16)
        
        # Extract bbox and scores
        # Format: [cx, cy, w, h, score0, score1, ..., score11]
        cx = preds[:, 0]
        cy = preds[:, 1]
        bw = preds[:, 2]
        bh = preds[:, 3]
        scores = preds[:, 4:]  # (N, 12)
        
        # Convert to l, t, r, b
        l = cx - bw / 2
        t = cy - bh / 2
        r = cx + bw / 2
        b = cy + bh / 2
        
        # Remove padding
        l = l - meta['pad_left']
        r = r - meta['pad_left']
        t = t - meta['pad_top']
        b = b - meta['pad_top']
        
        # Scale to crop size
        scale_x = meta['crop_w'] / (MODEL_WIDTH - meta['pad_left'] - meta['pad_right'])
        scale_y = meta['crop_h'] / (MODEL_HEIGHT - meta['pad_top'] - meta['pad_bottom'])
        l = l * scale_x
        r = r * scale_x
        t = t * scale_y
        b = b * scale_y
        
        # Add ROI offset
        l = l + meta['roi'][0]
        r = r + meta['roi'][0]
        t = t + meta['roi'][1]
        b = b + meta['roi'][1]
        
        # Scale to MODEL coordinates
        l = l * MODEL_WIDTH / meta['original_w']
        r = r * MODEL_WIDTH / meta['original_w']
        t = t * MODEL_HEIGHT / meta['original_h']
        b = b * MODEL_HEIGHT / meta['original_h']
        
        boxes = np.stack([l, t, r, b], axis=1)
        
        # Filter by confidence
        max_scores = np.max(scores, axis=1)
        mask = max_scores > conf_threshold
        
        if not np.any(mask):
            return {'boxes': np.array([]), 'scores': np.array([]), 'classes': np.array([])}
        
        boxes = boxes[mask]
        scores = scores[mask]
        classes = np.argmax(scores, axis=1)
        
        # Apply NMS
        max_score_values = np.max(scores, axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            max_score_values.tolist(),
            conf_threshold,
            0.3  # NMS threshold
        )
        
        if len(indices) == 0:
            return {'boxes': np.array([]), 'scores': np.array([]), 'classes': np.array([])}
        
        indices = np.array(indices).flatten()
        
        return {
            'boxes': boxes[indices],
            'scores': scores[indices],
            'classes': classes[indices]
        }
    
    def detect_pieces(self, frame: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Dict:
        """Detect chess pieces in a frame."""
        input_tensor, meta = self.preprocess_image(frame, keypoints)
        
        input_name = self.pieces_session.get_inputs()[0].name
        output = self.pieces_session.run(None, {input_name: input_tensor.astype(np.float16)})
        
        return self.postprocess_detections(output[0], meta)
    
    def detect_xcorners(self, frame: np.ndarray, keypoints: np.ndarray) -> Dict:
        """Detect X-corners (internal grid intersections) in a frame."""
        input_tensor, meta = self.preprocess_image(frame, keypoints)
        
        input_name = self.corners_session.get_inputs()[0].name
        output = self.corners_session.run(None, {input_name: input_tensor.astype(np.float16)})
        
        # X-corners model has 1 class
        result = self.postprocess_detections(output[0], meta, conf_threshold=0.1)
        
        if len(result['boxes']) == 0:
            return {'centers': np.array([])}
        
        # Get centers of detected corners
        boxes = result['boxes']
        centers = np.stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2
        ], axis=1)
        
        return {'centers': centers}
    
    def find_corners_auto(self, frame: np.ndarray) -> Dict:
        """
        Automatically find board corners from a frame.
        Uses piece positions to guide corner detection.
        """
        # Step 1: Detect pieces
        pieces = self.detect_pieces(frame)
        
        if len(pieces['boxes']) == 0:
            return {'success': False, 'message': 'No pieces detected in frame'}
        
        boxes = pieces['boxes']
        classes = pieces['classes']
        
        # Get piece centers
        piece_centers = np.stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2
        ], axis=1)
        
        # Separate black and white pieces
        black_mask = classes <= 5  # b, k, n, p, q, r
        white_mask = classes > 5   # B, K, N, P, Q, R
        
        black_centers = piece_centers[black_mask]
        white_centers = piece_centers[white_mask]
        
        if len(black_centers) == 0 or len(white_centers) == 0:
            return {'success': False, 'message': 'Need both black and white pieces'}
        
        # Step 2: Detect X-corners
        xcorners = self.detect_xcorners(frame, piece_centers)
        
        if len(xcorners['centers']) < 5:
            return {
                'success': False,
                'message': f'Need at least 5 X-corners, found {len(xcorners["centers"])}'
            }
        
        # Step 3: Find board corners from X-corners
        corners = self._find_corners_from_xcorners(xcorners['centers'], frame.shape)
        
        if corners is None:
            return {'success': False, 'message': 'Failed to calculate board corners'}
        
        # Step 4: Orient corners (white at bottom)
        oriented = self._orient_corners(corners, black_centers, white_centers)
        
        # Scale to frame coordinates
        scale_x = frame.shape[1] / MODEL_WIDTH
        scale_y = frame.shape[0] / MODEL_HEIGHT
        
        result = {}
        for key, pt in oriented.items():
            result[key] = [pt[0] * scale_x, pt[1] * scale_y]
        
        return {
            'success': True,
            'message': f'Found corners ({len(xcorners["centers"])} X-corners)',
            'corners': result,
            'xcorners': (xcorners['centers'] * np.array([scale_x, scale_y])).tolist()
        }
    
    def _find_corners_from_xcorners(self, xcorners: np.ndarray, frame_shape: tuple) -> Optional[np.ndarray]:
        """Find 4 board corners from X-corner detections."""
        if len(xcorners) < 4:
            return None
        
        # Use convex hull to find extreme points
        hull = cv2.convexHull(xcorners.astype(np.float32))
        hull = hull.reshape(-1, 2)
        
        if len(hull) < 4:
            return None
        
        # Find 4 corners by finding points furthest from centroid in each quadrant
        centroid = hull.mean(axis=0)
        
        # Sort by angle
        angles = np.arctan2(hull[:, 1] - centroid[1], hull[:, 0] - centroid[0])
        order = np.argsort(angles)
        hull = hull[order]
        
        # Sample 4 points roughly equally spaced
        n = len(hull)
        indices = [0, n//4, n//2, 3*n//4]
        corners = hull[indices]
        
        # Expand corners outward to get board edges (X-corners are internal)
        # Each X-corner is 1 square inside the board
        corner_expansion = self._estimate_square_size(xcorners)
        
        expanded = []
        for i, corner in enumerate(corners):
            direction = corner - centroid
            direction = direction / np.linalg.norm(direction)
            expanded.append(corner + direction * corner_expansion)
        
        return np.array(expanded)
    
    def _estimate_square_size(self, xcorners: np.ndarray) -> float:
        """Estimate square size from X-corner spacing."""
        if len(xcorners) < 2:
            return 20
        
        # Find minimum distances between corners
        dists = []
        for i in range(len(xcorners)):
            for j in range(i + 1, len(xcorners)):
                d = np.linalg.norm(xcorners[i] - xcorners[j])
                if d > 5:  # Ignore very close points
                    dists.append(d)
        
        if not dists:
            return 20
        
        # Square size is approximately the minimum distance
        return np.median(sorted(dists)[:max(1, len(dists)//4)])
    
    def _orient_corners(self, corners: np.ndarray, black_centers: np.ndarray,
                       white_centers: np.ndarray) -> Dict[str, List[float]]:
        """Orient corners so white is at bottom (a1-h1 side)."""
        black_centroid = black_centers.mean(axis=0)
        white_centroid = white_centers.mean(axis=0)
        
        best_shift = 0
        best_score = float('-inf')
        
        for shift in range(4):
            # White edge midpoint
            white_edge = (corners[shift % 4] + corners[(shift + 1) % 4]) / 2
            # Black edge midpoint (opposite side)
            black_edge = (corners[(shift + 2) % 4] + corners[(shift + 3) % 4]) / 2
            
            # Score: white pieces should be near white edge, black near black edge
            dist_w = np.linalg.norm(white_centroid - white_edge)
            dist_b = np.linalg.norm(black_centroid - black_edge)
            score = -dist_w - dist_b
            
            if score > best_score:
                best_score = score
                best_shift = shift
        
        return {
            'a1': corners[best_shift % 4].tolist(),
            'h1': corners[(best_shift + 1) % 4].tolist(),
            'h8': corners[(best_shift + 2) % 4].tolist(),
            'a8': corners[(best_shift + 3) % 4].tolist()
        }
    
    def set_corners(self, corners: Dict[str, List[float]]):
        """Set board corners and compute transforms."""
        self.corners = corners
        self._compute_transforms()
    
    def _compute_transforms(self):
        """Compute perspective transforms and square centers."""
        if self.corners is None:
            return
        
        # Source points (board corners in image coords)
        src = np.array([
            self.corners['a1'],
            self.corners['h1'],
            self.corners['h8'],
            self.corners['a8']
        ], dtype=np.float32)
        
        # Destination points (ideal board coordinates)
        # a1 at bottom-left, h8 at top-right
        dst = np.array([
            [0, BOARD_SIZE],        # a1
            [BOARD_SIZE, BOARD_SIZE],  # h1
            [BOARD_SIZE, 0],        # h8
            [0, 0]                  # a8
        ], dtype=np.float32)
        
        # Forward: image -> board
        self.transform = cv2.getPerspectiveTransform(src, dst)
        # Inverse: board -> image
        self.inv_transform = cv2.getPerspectiveTransform(dst, src)
        
        # Compute square centers in image coordinates
        centers_board = []
        for rank in range(8):  # 0=rank1, 7=rank8
            for file in range(8):  # 0=file a, 7=file h
                x = (file + 0.5) * SQUARE_SIZE
                y = (7 - rank + 0.5) * SQUARE_SIZE
                centers_board.append([x, y])
        
        centers_board = np.array(centers_board, dtype=np.float32).reshape(-1, 1, 2)
        self.centers = cv2.perspectiveTransform(centers_board, self.inv_transform).reshape(-1, 2)
        
        # Board boundary in image coordinates
        self.boundary = src.copy()
    
    def get_box_centers(self, boxes: np.ndarray) -> np.ndarray:
        """Get centers of boxes (bottom-center for chess pieces)."""
        l = boxes[:, 0]
        r = boxes[:, 2]
        b = boxes[:, 3]
        cx = (l + r) / 2
        cy = b - (r - l) / 3  # Use bottom-center adjusted for piece height
        return np.stack([cx, cy], axis=1)
    
    def get_squares(self, box_centers: np.ndarray) -> np.ndarray:
        """Assign boxes to squares. Returns square indices (-1 if outside board)."""
        squares = np.full(len(box_centers), -1, dtype=np.int32)
        
        for i, center in enumerate(box_centers):
            # Check if inside board
            if not self._is_inside_board(center):
                continue
            
            # Find closest square center
            dists = np.linalg.norm(self.centers - center, axis=1)
            squares[i] = np.argmin(dists)
        
        return squares
    
    def _is_inside_board(self, point: np.ndarray) -> bool:
        """Check if point is inside the board polygon."""
        result = cv2.pointPolygonTest(self.boundary.astype(np.float32), tuple(point), False)
        return result >= 0
    
    def update_state(self, scores: np.ndarray, squares: np.ndarray, decay: float = 0.5):
        """Update detection state with exponential moving average."""
        update = np.zeros((64, 12), dtype=np.float32)
        
        for i, square in enumerate(squares):
            if square == -1:
                continue
            for j in range(12):
                update[square][j] = max(update[square][j], scores[i][j])
        
        self.state = decay * self.state + (1 - decay) * update
    
    def calculate_move_score(self, move: chess.Move, from_thr: float = 0.6, to_thr: float = 0.6) -> float:
        """
        Calculate how well a move matches the current detection state.
        Matches React's calculateScore() function.
        """
        from_square = move.from_square
        to_square = move.to_square
        piece = self.board.piece_at(from_square)
        
        if piece is None:
            return float('-inf')
        
        piece_idx = get_piece_index(piece)
        if piece_idx == -1:
            return float('-inf')
        
        # From square should be empty (low scores)
        from_score = 1 - np.max(self.state[from_square]) - from_thr
        
        # To square should have the piece
        to_score = self.state[to_square][piece_idx] - to_thr
        
        return from_score + to_score
    
    def detect_move(self, current_time: float) -> Optional[str]:
        """
        Try to detect a move from the current state.
        Uses greedy detection with 1-second confirmation.
        """
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
        
        best_move = None
        best_score = 0
        
        for move in legal_moves:
            score = self.calculate_move_score(move)
            if score > best_score:
                best_score = score
                best_move = move
        
        if best_move is None:
            return None
        
        san = self.board.san(best_move)
        lan = best_move.uci()
        
        # Check if this is a new move
        if lan == self.last_move_lan:
            return None
        
        # Greedy move confirmation with timing
        if san not in self.greedy_move_times:
            self.greedy_move_times[san] = current_time
        
        # Require move to be stable for 500ms
        elapsed = current_time - self.greedy_move_times[san]
        if elapsed < 500:
            return None
        
        # Confirm the move
        self.board.push(best_move)
        self.moves.append(san)
        self.last_move_lan = lan
        self.greedy_move_times = {}  # Reset timings
        
        return san
    
    def process_frame(self, frame: np.ndarray, timestamp: float = 0) -> Dict:
        """
        Process a single frame and try to detect moves.
        This is the main loop function matching React's findPieces loop.
        """
        if self.corners is None or self.centers is None:
            return {'success': False, 'message': 'Corners not set'}
        
        # Get keypoints for cropping (in MODEL coordinates)
        scale_x = MODEL_WIDTH / frame.shape[1]
        scale_y = MODEL_HEIGHT / frame.shape[0]
        
        keypoints = np.array([
            [self.corners['h1'][0] * scale_x, self.corners['h1'][1] * scale_y],
            [self.corners['a1'][0] * scale_x, self.corners['a1'][1] * scale_y],
            [self.corners['a8'][0] * scale_x, self.corners['a8'][1] * scale_y],
            [self.corners['h8'][0] * scale_x, self.corners['h8'][1] * scale_y]
        ])
        
        # Detect pieces
        pieces = self.detect_pieces(frame, keypoints)
        
        if len(pieces['boxes']) == 0:
            return {
                'success': True,
                'message': 'No pieces detected',
                'moves': self.moves.copy(),
                'new_move': None,
                'num_pieces': 0
            }
        
        # Scale boxes to image coordinates
        boxes = pieces['boxes'].copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * frame.shape[1] / MODEL_WIDTH
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * frame.shape[0] / MODEL_HEIGHT
        
        # Get box centers and assign to squares
        box_centers = self.get_box_centers(boxes)
        squares = self.get_squares(box_centers)
        
        # Update state
        self.update_state(pieces['scores'], squares)
        
        # Try to detect move
        new_move = self.detect_move(timestamp)
        
        return {
            'success': True,
            'message': f'Detected {len(pieces["boxes"])} pieces',
            'moves': self.moves.copy(),
            'new_move': new_move,
            'num_pieces': len(pieces['boxes']),
            'fen': self.board.fen(),
            'detections': {
                'boxes': boxes.tolist(),
                'classes': pieces['classes'].tolist(),
                'scores': np.max(pieces['scores'], axis=1).tolist(),
                'squares': squares.tolist()
            },
            'state_summary': self._get_state_summary()
        }
    
    def _get_state_summary(self) -> Dict:
        """Get a summary of the current state for visualization."""
        board_state = {}
        for sq in range(64):
            max_score = np.max(self.state[sq])
            if max_score > 0.3:  # Threshold for having a piece
                piece_idx = np.argmax(self.state[sq])
                board_state[SQUARE_NAMES[sq]] = {
                    'piece': LABELS[piece_idx],
                    'confidence': float(max_score)
                }
        return board_state
    
    def analyze_initial_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze the initial frame thoroughly for user confirmation.
        Returns detailed detection info for all 32 pieces and 64 squares.
        """
        if self.corners is None or self.centers is None:
            return {'success': False, 'message': 'Corners not set'}
        
        # Get keypoints for cropping (in MODEL coordinates)
        scale_x = MODEL_WIDTH / frame.shape[1]
        scale_y = MODEL_HEIGHT / frame.shape[0]
        
        keypoints = np.array([
            [self.corners['h1'][0] * scale_x, self.corners['h1'][1] * scale_y],
            [self.corners['a1'][0] * scale_x, self.corners['a1'][1] * scale_y],
            [self.corners['a8'][0] * scale_x, self.corners['a8'][1] * scale_y],
            [self.corners['h8'][0] * scale_x, self.corners['h8'][1] * scale_y]
        ])
        
        # Detect pieces with lower threshold for initial analysis
        input_tensor, meta = self.preprocess_image(frame, keypoints)
        input_name = self.pieces_session.get_inputs()[0].name
        output = self.pieces_session.run(None, {input_name: input_tensor.astype(np.float16)})
        pieces = self.postprocess_detections(output[0], meta, conf_threshold=0.15)
        
        if len(pieces['boxes']) == 0:
            return {
                'success': False,
                'message': 'No pieces detected in initial frame'
            }
        
        # Scale boxes to image coordinates
        boxes = pieces['boxes'].copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * frame.shape[1] / MODEL_WIDTH
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * frame.shape[0] / MODEL_HEIGHT
        
        # Get box centers and assign to squares
        box_centers = self.get_box_centers(boxes)
        squares = self.get_squares(box_centers)
        
        # Build detailed detection list
        detections = []
        for i in range(len(boxes)):
            det = {
                'box': boxes[i].tolist(),
                'center': box_centers[i].tolist(),
                'class_idx': int(pieces['classes'][i]),
                'class_name': LABELS[pieces['classes'][i]],
                'confidence': float(np.max(pieces['scores'][i])),
                'square_idx': int(squares[i]),
                'square_name': SQUARE_NAMES[squares[i]] if squares[i] >= 0 else 'outside'
            }
            detections.append(det)
        
        # Count pieces by type
        white_pieces = sum(1 for d in detections if d['class_idx'] >= 6)
        black_pieces = sum(1 for d in detections if d['class_idx'] < 6)
        
        # Get square centers in image coordinates for visualization
        square_centers = {}
        for i, name in enumerate(SQUARE_NAMES):
            square_centers[name] = self.centers[i].tolist()
        
        # Get board boundary
        boundary = self.boundary.tolist() if self.boundary is not None else None
        
        return {
            'success': True,
            'message': f'Detected {len(detections)} pieces ({white_pieces} white, {black_pieces} black)',
            'num_pieces': len(detections),
            'white_pieces': white_pieces,
            'black_pieces': black_pieces,
            'detections': detections,
            'square_centers': square_centers,
            'boundary': boundary,
            'frame_shape': list(frame.shape)
        }
    
    def draw_debug_overlay(self, frame: np.ndarray, detections: Dict = None) -> np.ndarray:
        """Draw comprehensive debug overlay on frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        if self.corners is None or self.centers is None:
            return vis
        
        # Draw board grid (64 squares)
        self._draw_grid_overlay(vis)
        
        # Draw piece detections if provided
        if detections and 'boxes' in detections:
            self._draw_detections(vis, detections)
        
        # Draw state summary
        self._draw_state_overlay(vis)
        
        return vis
    
    def _draw_grid_overlay(self, frame: np.ndarray):
        """Draw the 64-square grid overlay."""
        if self.centers is None:
            return
        
        # Draw all 64 square centers and labels
        for i, name in enumerate(SQUARE_NAMES):
            center = self.centers[i]
            x, y = int(center[0]), int(center[1])
            
            # Determine square color (light/dark)
            file_idx = i % 8  # 0-7 for a-h
            rank_idx = i // 8  # 0-7 for ranks 1-8
            is_light = (file_idx + rank_idx) % 2 == 1
            
            # Draw square center
            color = (200, 200, 200) if is_light else (100, 100, 100)
            cv2.circle(frame, (x, y), 3, color, -1)
            
            # Draw square name (small)
            cv2.putText(frame, name, (x - 10, y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw board boundary
        if self.boundary is not None:
            pts = self.boundary.astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            # Draw corner labels
            corner_names = ['a1', 'h1', 'h8', 'a8']
            for i, name in enumerate(corner_names):
                x, y = int(self.boundary[i][0]), int(self.boundary[i][1])
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
                cv2.putText(frame, name, (x + 12, y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _draw_detections(self, frame: np.ndarray, detections: Dict):
        """Draw piece detection boxes and labels."""
        boxes = detections.get('boxes', [])
        classes = detections.get('classes', [])
        scores = detections.get('scores', [])
        squares = detections.get('squares', [])
        
        # Color map for pieces
        colors = {
            'b': (139, 69, 19), 'k': (0, 0, 0), 'n': (101, 67, 33),
            'p': (50, 50, 50), 'q': (75, 0, 130), 'r': (128, 0, 0),
            'B': (255, 215, 0), 'K': (255, 255, 255), 'N': (218, 165, 32),
            'P': (245, 245, 220), 'Q': (255, 105, 180), 'R': (255, 99, 71)
        }
        
        for i in range(len(boxes)):
            if i >= len(classes):
                break
            
            box = boxes[i]
            class_idx = classes[i]
            score = scores[i] if i < len(scores) else 0
            sq = squares[i] if i < len(squares) else -1
            
            piece_name = LABELS[class_idx]
            color = colors.get(piece_name, (0, 255, 0))
            
            # Brighten colors for visibility
            color = tuple(min(255, c + 50) for c in color)
            
            # Draw box
            l, t, r, b = [int(v) for v in box]
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            
            # Draw label
            sq_name = SQUARE_NAMES[sq] if 0 <= sq < 64 else '?'
            label = f"{piece_name}@{sq_name} {score:.0%}"
            
            # Label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (l, t - lh - 4), (l + lw + 4, t), color, -1)
            cv2.putText(frame, label, (l + 2, t - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    def _draw_state_overlay(self, frame: np.ndarray):
        """Draw the current board state as a mini-board overlay."""
        h, w = frame.shape[:2]
        
        # Mini-board in top-right corner
        board_size = min(200, w // 4)
        sq_size = board_size // 8
        offset_x = w - board_size - 10
        offset_y = 10
        
        # Draw mini-board background
        cv2.rectangle(frame, (offset_x - 2, offset_y - 2), 
                     (offset_x + board_size + 2, offset_y + board_size + 2), 
                     (50, 50, 50), -1)
        
        # Draw squares and pieces
        for rank in range(8):
            for file in range(8):
                sq_idx = rank * 8 + file
                x = offset_x + file * sq_size
                y = offset_y + (7 - rank) * sq_size
                
                # Square color
                is_light = (file + rank) % 2 == 1
                sq_color = (240, 217, 181) if is_light else (181, 136, 99)
                cv2.rectangle(frame, (x, y), (x + sq_size, y + sq_size), sq_color, -1)
                
                # Check if piece detected
                max_score = np.max(self.state[sq_idx])
                if max_score > 0.3:
                    piece_idx = np.argmax(self.state[sq_idx])
                    piece = LABELS[piece_idx]
                    
                    # Draw piece symbol
                    is_white = piece.isupper()
                    text_color = (255, 255, 255) if not is_white else (0, 0, 0)
                    bg_color = (0, 0, 0) if not is_white else (255, 255, 255)
                    
                    # Circle for piece
                    cx, cy = x + sq_size // 2, y + sq_size // 2
                    cv2.circle(frame, (cx, cy), sq_size // 3, bg_color, -1)
                    cv2.putText(frame, piece.upper(), (cx - 5, cy + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    def generate_pgn(self) -> str:
        """Generate PGN from detected moves."""
        game = chess.pgn.Game()
        game.headers["Event"] = "Chess Video Analysis"
        game.headers["Site"] = "ChessAnalyzer"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = "?"
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
    
    def reset(self):
        """Reset the analyzer state for a new video."""
        self.board = chess.Board()
        self.moves = []
        self.state = np.zeros((64, 12), dtype=np.float32)
        self.greedy_move_times = {}
        self.last_move_lan = None


def process_video(video_path: str, corners: Dict[str, List[float]], 
                  analyzer: ChessAnalyzer, 
                  progress_callback=None) -> Dict:
    """
    Process a complete video with given corners.
    
    Args:
        video_path: Path to video file
        corners: Dict with a1, h1, h8, a8 coordinates
        analyzer: ChessAnalyzer instance
        progress_callback: Optional callback(progress, frame_data)
    
    Returns:
        Dict with moves, pgn, and statistics
    """
    analyzer.reset()
    analyzer.set_corners(corners)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'success': False, 'message': 'Failed to open video'}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        fps = 30
    
    frame_count = 0
    # Process ~2 frames per second for efficiency
    process_interval = max(1, int(fps / 2))
    
    print(f"Processing video: {total_frames} frames, {fps:.1f} fps")
    print(f"Processing every {process_interval} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % process_interval == 0:
            timestamp = frame_count / fps * 1000  # ms
            result = analyzer.process_frame(frame, timestamp)
            
            if result.get('new_move'):
                print(f"  Move detected: {result['new_move']} (frame {frame_count})")
            
            if progress_callback:
                progress = frame_count / total_frames * 100
                progress_callback(progress, result)
        
        frame_count += 1
    
    cap.release()
    
    pgn = analyzer.generate_pgn()
    
    return {
        'success': True,
        'moves': analyzer.moves,
        'pgn': pgn,
        'total_frames': total_frames,
        'frames_processed': frame_count // process_interval,
        'fen': analyzer.board.fen()
    }
