"""
ChessWorldAI - Configuration Constants

This module contains all configuration constants for the chess video to PGN converter.
Optimized for over-the-board chess video analysis.
"""

import numpy as np

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Input dimensions for the piece detection model
MODEL_WIDTH = 640
MODEL_HEIGHT = 640

# Confidence thresholds for detection
PIECE_CONFIDENCE_THRESHOLD = 0.15  # Lower threshold for piece detection (was 0.35)
CORNER_CONFIDENCE_THRESHOLD = 0.20  # Threshold for corner/xcorner detection
NMS_IOU_THRESHOLD = 0.50  # Non-maximum suppression IoU threshold (higher = more overlapping detections allowed)

# =============================================================================
# CHESS PIECE LABELS
# =============================================================================

# Standard piece labels (lowercase = black, uppercase = white)
# Order matches the state tracker: b, k, n, p, q, r, B, K, N, P, Q, R
PIECE_LABELS = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']
PIECE_SYMBOLS = ['b', 'k', 'n', 'p', 'q', 'r']  # Bishop, King, Knight, Pawn, Queen, Rook

# Mapping from label index to piece info
LABEL_TO_PIECE = {
    0: ('b', 'black'),   # Black Bishop
    1: ('k', 'black'),   # Black King
    2: ('n', 'black'),   # Black Knight
    3: ('p', 'black'),   # Black Pawn
    4: ('q', 'black'),   # Black Queen
    5: ('r', 'black'),   # Black Rook
    6: ('B', 'white'),   # White Bishop
    7: ('K', 'white'),   # White King
    8: ('N', 'white'),   # White Knight
    9: ('P', 'white'),   # White Pawn
    10: ('Q', 'white'),  # White Queen
    11: ('R', 'white'),  # White Rook
}

# Mapping from Roboflow/trained model class names to our standard indices
# The trained model has: bishop(0), black-bishop(1), black-king(2), ...
# We need to map these to our 12-class scheme
TRAINED_MODEL_CLASS_MAPPING = {
    'bishop': None,  # Generic bishop - ignore
    'black-bishop': 0,  # b
    'black-king': 1,    # k
    'black-knight': 2,  # n
    'black-pawn': 3,    # p
    'black-queen': 4,   # q
    'black-rook': 5,    # r
    'white-bishop': 6,  # B
    'white-king': 7,    # K
    'white-knight': 8,  # N
    'white-pawn': 9,    # P
    'white-queen': 10,  # Q
    'white-rook': 11,   # R
}

# Mapping from trained model class index to our standard index
# Trained model: 0=bishop, 1=black-bishop, 2=black-king, ...
TRAINED_CLASS_INDEX_MAPPING = {
    0: None,  # Generic 'bishop' - ignore
    1: 0,     # black-bishop -> b
    2: 1,     # black-king -> k
    3: 2,     # black-knight -> n
    4: 3,     # black-pawn -> p
    5: 4,     # black-queen -> q
    6: 5,     # black-rook -> r
    7: 6,     # white-bishop -> B
    8: 7,     # white-king -> K
    9: 8,     # white-knight -> N
    10: 9,    # white-pawn -> P
    11: 10,   # white-queen -> Q
    12: 11,   # white-rook -> R
}

# =============================================================================
# CHESS BOARD CONFIGURATION
# =============================================================================

# Square size for warped board (virtual board)
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE  # 400 pixels

# Square names in algebraic notation (a1-h8)
SQUARE_NAMES = [
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
    'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
    'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
    'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8',
]

# Corner keys for board orientation
CORNER_KEYS = ['h1', 'a1', 'a8', 'h8']

# Starting FEN position
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# =============================================================================
# VIDEO PROCESSING CONFIGURATION
# =============================================================================

# Frame sampling rate (process every Nth frame)
# Lower = more accurate but slower
FRAME_SAMPLE_RATE = 2

# Minimum frames between move detection to avoid false positives
MIN_FRAMES_BETWEEN_MOVES = 5

# State smoothing parameters (exponential moving average)
STATE_DECAY = 0.7  # How quickly old observations fade

# Move detection thresholds
MOVE_DETECTION_THRESHOLD = 0.6  # Confidence for detecting a move
PIECE_PRESENCE_THRESHOLD = 0.5  # Threshold for piece presence on square

# =============================================================================
# VISUALIZATION COLORS (BGR format for OpenCV)
# =============================================================================

COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'orange': (0, 165, 255),
}

# Piece colors for visualization
PIECE_COLORS = [
    (56, 56, 255),    # Black Bishop - Red
    (151, 151, 255),  # Black King - Light Red
    (31, 112, 255),   # Black Knight - Orange
    (29, 178, 255),   # Black Pawn - Yellow
    (49, 210, 207),   # Black Queen - Light Green
    (10, 249, 72),    # Black Rook - Green
    (23, 204, 146),   # White Bishop - Teal
    (134, 219, 61),   # White King - Cyan
    (255, 109, 58),   # White Knight - Blue
    (255, 64, 64),    # White Pawn - Blue-Purple
    (203, 86, 255),   # White Queen - Purple
    (255, 56, 151),   # White Rook - Magenta
]

# =============================================================================
# BOARD DETECTION PARAMETERS
# =============================================================================

# Hough transform parameters for line detection
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 100

# Canny edge detection thresholds
CANNY_LOW = 50
CANNY_HIGH = 150

# Corner refinement parameters
CORNER_SEARCH_WINDOW = 11
CORNER_ZERO_ZONE = -1
CORNER_CRITERIA_ITERATIONS = 30
CORNER_CRITERIA_EPS = 0.001

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Model paths (relative to project root)
MODELS_DIR = "models"
PIECES_MODEL_PATH = f"{MODELS_DIR}/pieces_yolov8.pt"
XCORNERS_MODEL_PATH = f"{MODELS_DIR}/xcorners_yolov8.pt"

# Output paths
OUTPUT_DIR = "output"
PGN_OUTPUT_DIR = f"{OUTPUT_DIR}/pgn"
VIDEO_OUTPUT_DIR = f"{OUTPUT_DIR}/videos"
