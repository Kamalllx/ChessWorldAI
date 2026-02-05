"""
Process a single video file and generate PGN
Used by batch_runner.py - Simplified standalone version
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analyzer import ChessAnalyzer

def process_video(video_path: str, output_path: str):
    """Process a single video and save PGN"""
    
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Failed to open video: {video_path}")
        return 1
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Total frames: {total_frames}, FPS: {fps:.2f}")
    
    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Failed to read first frame")
        cap.release()
        return 1
    
    # Initialize analyzer
    print("Initializing chess analyzer...")
    pieces_path = "480M_pieces_float16"
    corners_path = "480L_xcorners_float16"
    
    analyzer = ChessAnalyzer(pieces_path, corners_path)
    
    # Auto-detect corners
    print("Detecting corners...")
    corner_result = analyzer.find_corners_auto(first_frame)
    
    if not corner_result['success']:
        print(f"ERROR: Corner detection failed: {corner_result['message']}")
        cap.release()
        return 1
    
    analyzer.set_corners(corner_result['corners'])
    print(f"OK: Corners detected: {list(corner_result['corners'].keys())}")
    
    # Analyze initial position
    print("Analyzing initial position...")
    init_result = analyzer.analyze_initial_frame(first_frame)
    
    if not init_result['success']:
        print(f"ERROR: Initial analysis failed: {init_result['message']}")
        cap.release()
        return 1
    
    print(f"OK: Initial position analyzed")
    
    # Process remaining frames
    print("Processing frames...")
    frame_count = 1  # Already processed first frame
    moves_detected = 0
    
    # Skip to frame 30 to avoid initial noise
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        result = analyzer.process_frame(frame)
        
        if result['success'] and result.get('move'):
            moves_detected += 1
            print(f"   Move {moves_detected}: {result['move']} (frame {frame_count})")
        
        # Progress update every 200 frames
        if frame_count % 200 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {frame_count}/{total_frames} ({progress:.1f}%) - {moves_detected} moves")
    
    cap.release()
    
    # Get PGN
    print(f"Saving PGN to: {output_path}")
    pgn_content = analyzer.generate_pgn()
    
    with open(output_path, 'w') as f:
        f.write(pgn_content)
    
    print(f"OK: Complete! {moves_detected} moves detected, {frame_count} frames processed")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_single_video.py <video_path> <output_path>")
        exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2]
    
    exit(process_video(video_path, output_path))
