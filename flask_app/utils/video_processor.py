import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from utils.detector import ChessDetector
from utils.board_tracker import BoardTracker

class VideoProcessor:
    """Process chess game videos frame by frame."""
    
    def __init__(self, detector: ChessDetector):
        """
        Initialize video processor.
        
        Args:
            detector: ChessDetector instance
        """
        self.detector = detector
        self.tracker = BoardTracker()
        self.board_corners = None
        self.corner_detection_attempts = 0
        self.max_corner_attempts = 30
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process a single video frame.
        
        Args:
            frame: Video frame (BGR)
        
        Returns:
            Detection results or None if processing failed
        """
        # Detect corners if not yet found
        if self.board_corners is None and self.corner_detection_attempts < self.max_corner_attempts:
            corners = self.detector.detect_corners(frame)
            self.corner_detection_attempts += 1
            
            if len(corners) >= 4:
                # Find the 4 best corner candidates
                self.board_corners = self._select_board_corners(corners)
                print(f"✓ Board corners detected: {self.board_corners.shape}")
        
        # Detect pieces
        pieces = self.detector.detect_pieces(frame, self.board_corners)
        
        # Update tracker with detections
        if pieces and len(pieces.get('boxes', [])) > 0:
            self.tracker.update(pieces)
        
        return {
            'corners': self.board_corners,
            'pieces': pieces,
            'frame_shape': frame.shape,
            'moves': self.tracker.moves
        }
    
    def _select_board_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Select the 4 corners that best represent the board corners.
        
        Args:
            corners: Array of detected corner points
        
        Returns:
            4 corner points in order: [top-left, top-right, bottom-right, bottom-left]
        """
        if len(corners) < 4:
            return corners
        
        # Find extreme points
        # Top-left: minimum sum of coordinates
        # Top-right: maximum x - y
        # Bottom-right: maximum sum of coordinates
        # Bottom-left: minimum x - y
        
        sum_coords = corners[:, 0] + corners[:, 1]
        diff_coords = corners[:, 0] - corners[:, 1]
        
        tl_idx = np.argmin(sum_coords)
        br_idx = np.argmax(sum_coords)
        tr_idx = np.argmax(diff_coords)
        bl_idx = np.argmin(diff_coords)
        
        board_corners = np.array([
            corners[tl_idx],  # top-left
            corners[tr_idx],  # top-right
            corners[br_idx],  # bottom-right
            corners[bl_idx]   # bottom-left
        ])
        
        return board_corners
    
    def draw_detections(
        self,
        frame: np.ndarray,
        result: Dict,
        draw_corners: bool = True,
        draw_pieces: bool = True
    ) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Original frame
            result: Detection results
            draw_corners: Whether to draw corners
            draw_pieces: Whether to draw piece detections
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Scale factors (detections are in model coordinates)
        scale_x = w / self.detector.MODEL_WIDTH
        scale_y = h / self.detector.MODEL_HEIGHT
        
        # Draw corners
        if draw_corners and result['corners'] is not None:
            corners = result['corners']
            for i, corner in enumerate(corners):
                x, y = int(corner[0] * scale_x), int(corner[1] * scale_y)
                cv2.circle(annotated, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(
                    annotated,
                    str(i),
                    (x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
            
            # Draw board outline
            if len(corners) == 4:
                pts = (corners * [scale_x, scale_y]).astype(np.int32)
                cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        
        # Draw pieces
        if draw_pieces and len(result['pieces']['boxes']) > 0:
            boxes = result['pieces']['boxes']
            scores = result['pieces']['scores']
            classes = result['pieces']['classes']
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                
                # Safety check for invalid coordinates
                if np.isnan([x1, y1, x2, y2]).any() or np.isinf([x1, y1, x2, y2]).any():
                    continue
                
                # Safety check for invalid class index
                cls_idx = int(cls)
                if cls_idx < 0 or cls_idx >= len(self.detector.LABELS):
                    continue
                
                x1, y1 = int(np.clip(x1 * scale_x, 0, w)), int(np.clip(y1 * scale_y, 0, h))
                x2, y2 = int(np.clip(x2 * scale_x, 0, w)), int(np.clip(y2 * scale_y, 0, h))
                
                # Color based on piece color
                label = self.detector.LABELS[cls_idx]
                color = (255, 255, 0) if label.isupper() else (0, 255, 255)
                
                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_text = f"{label} {score:.2f}"
                cv2.putText(
                    annotated,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
        
        return annotated
    
    def reset_corners(self):
        """Reset corner detection state."""
        self.board_corners = None
        self.corner_detection_attempts = 0
