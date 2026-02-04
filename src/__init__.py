"""
ChessWorldAI - Chess Video to PGN Converter

A state-of-the-art computer vision system for converting over-the-board
chess game videos into PGN notation.

Author: ChessWorldAI Team
License: MIT
"""

from .config import (
    MODEL_WIDTH, MODEL_HEIGHT, PIECE_LABELS, SQUARE_NAMES,
    START_FEN, BOARD_SIZE, SQUARE_SIZE
)

from .piece_detection import PieceDetector

__version__ = '1.1.0'
__author__ = 'ChessWorldAI Team'

__all__ = [
    # Config
    'MODEL_WIDTH', 'MODEL_HEIGHT', 'PIECE_LABELS', 'SQUARE_NAMES',
    'START_FEN', 'BOARD_SIZE', 'SQUARE_SIZE',
    # Piece Detection
    'PieceDetector',
]
