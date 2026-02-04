"""
ChessWorldAI - Piece Detection Module

This module handles chess piece detection and classification using deep learning.
Uses YOLOv8 for real-time piece detection with support for various lighting conditions.

Key Features:
- Real-time piece detection using YOLOv8
- Support for both black and white pieces (12 classes)
- Confidence-based filtering and NMS
- Batch processing for efficiency
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("Warning: ultralytics not installed. Using fallback detection.")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import (
    PIECE_LABELS, LABEL_TO_PIECE, PIECE_COLORS,
    PIECE_CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
    MODEL_WIDTH, MODEL_HEIGHT, SQUARE_SIZE, BOARD_SIZE,
    SQUARE_NAMES, TRAINED_CLASS_INDEX_MAPPING
)


class PieceDetector:
    """
    Detects and classifies chess pieces in video frames.
    
    Uses YOLOv8 for detection with fallback to color-based detection
    when neural network is not available.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the piece detector.
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.model = None
        self.device = self._select_device(device)
        self.class_mapping = None  # Will be set when model loads
        
        if model_path and HAS_ULTRALYTICS:
            self._load_model(model_path)
        elif HAS_ULTRALYTICS:
            # Try to load default model
            self._load_default_model()
            
    def _select_device(self, device: str) -> str:
        """Select computation device."""
        if device == 'auto':
            if HAS_TORCH and torch.cuda.is_available():
                return 'cuda'
            return 'cpu'
        return device
        
    def _load_model(self, model_path: str):
        """Load YOLO model from file."""
        try:
            self.model = YOLO(model_path)
            print(f"Loaded piece detection model from {model_path}")
            # Check if this is the trained model with descriptive names
            if self.model.names and 'black-bishop' in self.model.names.values():
                print("  Using trained model class mapping")
                self.class_mapping = TRAINED_CLASS_INDEX_MAPPING
            else:
                print("  Using direct class mapping")
                self.class_mapping = None
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            
    def _load_default_model(self):
        """Try to load default pretrained model."""
        default_paths = [
            'models/pieces_trained.pt',   # Our trained model (preferred)
            'runs/detect/runs/chess-pieces/train/weights/best.pt',  # Training output
            'models/pieces_yolov8.pt',
            'models/pieces.pt',
            'weights/pieces_yolov8.pt',
        ]
        
        for path in default_paths:
            if Path(path).exists():
                self._load_model(path)
                if self.model is not None:
                    break
                
    def detect(
        self, 
        frame: np.ndarray,
        corners: Optional[np.ndarray] = None,
        conf_threshold: float = PIECE_CONFIDENCE_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Detect all chess pieces in the frame.
        
        Args:
            frame: Input video frame (BGR format)
            corners: Optional board corners for region-of-interest cropping
            conf_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary with detection results:
            - boxes: Bounding boxes (N, 4) in xyxy format
            - scores: Confidence scores (N,)
            - classes: Class indices (N,)
            - centers: Box centers (N, 2)
        """
        if self.model is not None:
            return self._detect_neural(frame, corners, conf_threshold)
        else:
            return self._detect_fallback(frame, corners)
            
    def _detect_neural(
        self, 
        frame: np.ndarray,
        corners: Optional[np.ndarray],
        conf_threshold: float
    ) -> Dict[str, Any]:
        """
        Detect pieces using YOLO neural network.
        """
        # Crop to ROI if corners provided
        if corners is not None:
            frame, offset = self._crop_to_roi(frame, corners)
        else:
            offset = (0, 0)
            
        # Run inference
        results = self.model(
            frame,
            conf=conf_threshold,
            iou=NMS_IOU_THRESHOLD,
            verbose=False,
            device=self.device
        )
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return self._empty_result()
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        # Adjust boxes for ROI offset
        boxes[:, [0, 2]] += offset[0]
        boxes[:, [1, 3]] += offset[1]
        
        # Calculate centers
        centers = np.column_stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2
        ])
        
        # Apply class mapping if needed (for trained model with descriptive names)
        mapped_classes = classes.copy()
        valid_mask = np.ones(len(classes), dtype=bool)
        
        if self.class_mapping is not None:
            for i, cls in enumerate(classes):
                mapped = self.class_mapping.get(int(cls))
                if mapped is None:
                    # Invalid class (like generic 'bishop'), skip it
                    valid_mask[i] = False
                else:
                    mapped_classes[i] = mapped
                    
            # Filter out invalid detections
            if not np.all(valid_mask):
                boxes = boxes[valid_mask]
                scores = scores[valid_mask]
                mapped_classes = mapped_classes[valid_mask]
                centers = centers[valid_mask]
                
                if len(boxes) == 0:
                    return self._empty_result()
        
        # Generate labels from mapped classes
        labels = [PIECE_LABELS[c] if 0 <= c < len(PIECE_LABELS) else '?' for c in mapped_classes]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': mapped_classes,
            'centers': centers,
            'labels': labels
        }
        
    def _detect_fallback(
        self, 
        frame: np.ndarray,
        corners: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Fallback detection using color-based methods.
        This is a simplified backup when YOLO is not available.
        """
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for white and black pieces
        # These are approximate and may need tuning
        white_lower = np.array([0, 0, 150])
        white_upper = np.array([180, 60, 255])
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 80])
        
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        
        # Find contours
        boxes = []
        scores = []
        classes = []
        
        for mask, is_white in [(white_mask, True), (black_mask, False)]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100 or area > 10000:  # Filter by size
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # Chess pieces are typically taller than wide
                if aspect_ratio < 0.8:
                    continue
                    
                boxes.append([x, y, x + w, y + h])
                scores.append(0.5)  # Default confidence
                # Default to pawn (most common piece)
                classes.append(9 if is_white else 3)
                
        if not boxes:
            return self._empty_result()
            
        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array(classes)
        
        centers = np.column_stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2
        ])
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'centers': centers,
            'labels': [PIECE_LABELS[c] for c in classes]
        }
        
    def _crop_to_roi(
        self, 
        frame: np.ndarray, 
        corners: np.ndarray,
        padding_ratio: float = 0.1
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop frame to region of interest around the board.
        """
        h, w = frame.shape[:2]
        
        # Get bounding box of corners with padding
        x_min = max(0, int(corners[:, 0].min() - w * padding_ratio))
        x_max = min(w, int(corners[:, 0].max() + w * padding_ratio))
        y_min = max(0, int(corners[:, 1].min() - h * padding_ratio))
        y_max = min(h, int(corners[:, 1].max() + h * padding_ratio))
        
        cropped = frame[y_min:y_max, x_min:x_max]
        
        return cropped, (x_min, y_min)
        
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty detection result."""
        return {
            'boxes': np.array([]).reshape(0, 4),
            'scores': np.array([]),
            'classes': np.array([]),
            'centers': np.array([]).reshape(0, 2),
            'labels': []
        }


class BoardStateEstimator:
    """
    Estimates the state of each square on the chess board.
    
    Uses piece detections to build a probability distribution
    over piece types for each of the 64 squares.
    """
    
    def __init__(self, decay: float = 0.7):
        """
        Initialize the board state estimator.
        
        Args:
            decay: Exponential decay factor for temporal smoothing
        """
        self.decay = decay
        self.state = np.zeros((64, 12))  # 64 squares, 12 piece types
        self.square_centers = self._compute_square_centers()
        
    def _compute_square_centers(self) -> np.ndarray:
        """
        Compute centers of all 64 squares in normalized coordinates.
        """
        centers = []
        for row in range(8):  # 0 = rank 1, 7 = rank 8
            for col in range(8):  # 0 = file a, 7 = file h
                x = (col + 0.5) * SQUARE_SIZE
                y = (7 - row + 0.5) * SQUARE_SIZE
                centers.append([x, y])
        return np.array(centers)
        
    def update(
        self,
        detections: Dict[str, Any],
        transform: np.ndarray
    ) -> np.ndarray:
        """
        Update board state based on new detections.
        
        Args:
            detections: Detection results from PieceDetector
            transform: Perspective transform matrix (original -> warped)
            
        Returns:
            Updated state matrix (64, 12)
        """
        if len(detections['boxes']) == 0:
            # Decay existing state
            self.state *= self.decay
            return self.state
            
        # Transform detection centers to warped coordinates
        centers = detections['centers']
        
        # Add homogeneous coordinate
        n = len(centers)
        homogeneous = np.column_stack([centers, np.ones(n)])
        transformed = (transform @ homogeneous.T).T
        warped_centers = transformed[:, :2] / transformed[:, 2:3]
        
        # Create update matrix
        update = np.zeros((64, 12))
        
        for i, (center, cls, score) in enumerate(zip(
            warped_centers,
            detections['classes'],
            detections['scores']
        )):
            # Check if center is within board bounds
            if not (0 <= center[0] < BOARD_SIZE and 0 <= center[1] < BOARD_SIZE):
                continue
                
            # Find nearest square
            # Adjust y-coordinate for piece center (pieces are taller than their base)
            adjusted_center = center.copy()
            box = detections['boxes'][i]
            box_height = box[3] - box[1]
            
            # Use bottom third of bounding box for square assignment
            adjusted_y = center[1] + box_height * 0.2
            adjusted_center[1] = min(adjusted_y, BOARD_SIZE - 1)
            
            # Calculate distances to all square centers
            distances = np.linalg.norm(self.square_centers - adjusted_center, axis=1)
            nearest_square = np.argmin(distances)
            
            # Update state for this square
            update[nearest_square, cls] = max(update[nearest_square, cls], score)
            
        # Apply exponential moving average
        self.state = self.decay * self.state + (1 - self.decay) * update
        
        return self.state
        
    def get_board_fen_array(self, threshold: float = 0.3) -> List[Optional[str]]:
        """
        Convert state matrix to FEN-style array.
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            List of 64 piece symbols (or None for empty squares)
        """
        result = []
        for i in range(64):
            max_score = np.max(self.state[i])
            if max_score < threshold:
                result.append(None)
            else:
                piece_idx = np.argmax(self.state[i])
                result.append(PIECE_LABELS[piece_idx])
        return result
        
    def get_piece_positions(self, threshold: float = 0.3) -> Dict[str, str]:
        """
        Get dictionary mapping squares to pieces.
        
        Args:
            threshold: Minimum confidence threshold
            
        Returns:
            Dictionary like {'e2': 'P', 'e7': 'p', ...}
        """
        positions = {}
        for i, square_name in enumerate(SQUARE_NAMES):
            max_score = np.max(self.state[i])
            if max_score >= threshold:
                piece_idx = np.argmax(self.state[i])
                positions[square_name] = PIECE_LABELS[piece_idx]
        return positions
        
    def reset(self):
        """Reset state to zeros."""
        self.state = np.zeros((64, 12))


def assign_pieces_to_squares(
    detections: Dict[str, Any],
    transform: np.ndarray,
    square_centers: np.ndarray
) -> List[Tuple[int, int, float]]:
    """
    Assign detected pieces to board squares.
    
    Args:
        detections: Detection results
        transform: Perspective transform matrix
        square_centers: Centers of all 64 squares in warped coordinates
        
    Returns:
        List of (square_index, class_index, confidence) tuples
    """
    if len(detections['boxes']) == 0:
        return []
        
    # Transform detection centers
    centers = detections['centers']
    n = len(centers)
    homogeneous = np.column_stack([centers, np.ones(n)])
    transformed = (transform @ homogeneous.T).T
    warped_centers = transformed[:, :2] / transformed[:, 2:3]
    
    assignments = []
    
    for i, (center, cls, score) in enumerate(zip(
        warped_centers,
        detections['classes'],
        detections['scores']
    )):
        # Check bounds
        if not (0 <= center[0] < BOARD_SIZE and 0 <= center[1] < BOARD_SIZE):
            continue
            
        # Find nearest square
        distances = np.linalg.norm(square_centers - center, axis=1)
        nearest = np.argmin(distances)
        
        assignments.append((nearest, cls, score))
        
    return assignments


def visualize_detections(
    frame: np.ndarray,
    detections: Dict[str, Any],
    corners: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draw detection results on frame.
    
    Args:
        frame: Input frame
        detections: Detection results
        corners: Optional board corners to draw
        
    Returns:
        Annotated frame
    """
    result = frame.copy()
    
    # Draw corners
    if corners is not None:
        for i, corner in enumerate(corners):
            cv2.circle(result, tuple(corner.astype(int)), 8, (0, 0, 255), -1)
            cv2.putText(
                result,
                ['h1', 'a1', 'a8', 'h8'][i],
                tuple((corner + [5, -5]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
            
        # Draw board outline
        pts = corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(result, [pts], True, (0, 255, 0), 2)
        
    # Draw detections
    for box, score, cls, label in zip(
        detections['boxes'],
        detections['scores'],
        detections['classes'],
        detections['labels']
    ):
        x1, y1, x2, y2 = box.astype(int)
        color = PIECE_COLORS[cls]
        
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        text = f"{label} {score:.2f}"
        cv2.putText(
            result,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        
    return result
