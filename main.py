#!/usr/bin/env python3
"""
ChessWorldAI - Main Entry Point

Command-line interface for processing chess game videos into PGN format.

Usage:
    python main.py <video_path> [options]
    
Examples:
    python main.py game1.mp4
    python main.py game1.mp4 --output game1.pgn
    python main.py game1.mp4 --no-viz --batch
    python main.py videos/ --batch --output pgns/
    python main.py game1.mp4 --v2          Use improved processor (recommended)
    python main.py game1.mp4 --speed 0.5   Process at half speed
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import START_FEN

# Import both processor versions
try:
    from src.video_processor import ChessVideoProcessor, interactive_calibration
except ImportError:
    ChessVideoProcessor = None
    interactive_calibration = None

try:
    from src.video_processor_v2 import ImprovedChessVideoProcessor, process_video_simple
except ImportError:
    ImprovedChessVideoProcessor = None
    process_video_simple = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ChessWorldAI - Convert chess game videos to PGN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s game.mp4                    Process single video with visualization
  %(prog)s game.mp4 -o game.pgn        Process and save PGN to file
  %(prog)s videos/ --batch             Process all videos in directory
  %(prog)s game.mp4 --calibrate        Manually select board corners first
  %(prog)s game.mp4 --no-crop          Don't crop to top half
        """
    )
    
    parser.add_argument(
        'input',
        help='Input video file or directory of videos'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output PGN file or directory'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all videos in input directory'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization window'
    )
    
    parser.add_argument(
        '--calibrate',
        action='store_true',
        help='Manually calibrate board corners before processing'
    )
    
    parser.add_argument(
        '--no-crop',
        action='store_true',
        help='Process full frame (don\'t crop to top half)'
    )
    
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save annotated output video'
    )
    
    parser.add_argument(
        '--fen',
        default=START_FEN,
        help='Starting position FEN (default: standard chess starting position)'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Computation device (default: auto)'
    )
    
    parser.add_argument(
        '--pieces-model',
        help='Path to piece detection model weights'
    )
    
    parser.add_argument(
        '--corners-model',
        help='Path to corner detection model weights'
    )
    
    # New V2 options
    parser.add_argument(
        '--v2', '--improved',
        action='store_true',
        dest='use_v2',
        help='Use improved processor v2 (recommended for OTB videos)'
    )
    
    parser.add_argument(
        '--speed',
        type=float,
        default=0.5,
        help='Playback speed for processing (default: 0.5 = half speed)'
    )
    
    parser.add_argument(
        '--crop-ratio',
        type=float,
        default=0.45,
        help='Ratio of frame to keep from top (default: 0.45)'
    )
    
    parser.add_argument(
        '--stabilize',
        type=int,
        default=5,
        help='Frames to stabilize before confirming move (default: 5)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode with manual board selection'
    )

    return parser.parse_args()


def process_single_video_v2(
    video_path: Path,
    output_path: Optional[Path],
    args: argparse.Namespace
) -> dict:
    """Process a single video file with V2 processor."""
    print(f"\nProcessing (V2): {video_path}")
    
    if ImprovedChessVideoProcessor is None:
        print("Error: V2 processor not available. Falling back to V1.")
        return process_single_video(video_path, output_path, args, None)
    
    # Create improved processor
    processor = ImprovedChessVideoProcessor(
        pieces_model_path=args.pieces_model,
        xcorners_model_path=args.corners_model,
        device=args.device,
        visualize=not args.no_viz,
        save_annotated_video=args.save_video,
        crop_ratio=args.crop_ratio,
        playback_speed=args.speed,
        stabilization_frames=args.stabilize
    )
    
    # Determine output path
    pgn_path = None
    if output_path:
        pgn_path = str(output_path)
    elif args.batch:
        pgn_path = str(video_path.with_suffix('.pgn'))
        
    video_output_path = None
    if args.save_video:
        video_output_path = str(video_path.with_suffix('.annotated.mp4'))
        
    # Process video
    result = processor.process_video(
        video_path=str(video_path),
        output_pgn_path=pgn_path,
        output_video_path=video_output_path,
        starting_fen=args.fen,
        interactive_calibration=args.interactive or args.calibrate
    )
    
    return result


def process_single_video(
    video_path: Path,
    output_path: Optional[Path],
    args: argparse.Namespace,
    corners: Optional[any] = None
) -> dict:
    """Process a single video file."""
    print(f"\nProcessing: {video_path}")
    
    # Create processor
    processor = ChessVideoProcessor(
        pieces_model_path=args.pieces_model,
        xcorners_model_path=args.corners_model,
        device=args.device,
        visualize=not args.no_viz,
        save_annotated_video=args.save_video
    )
    
    # Determine output paths
    pgn_path = None
    video_output_path = None
    
    if output_path:
        pgn_path = str(output_path)
    elif args.batch:
        # Default output in same directory with .pgn extension
        pgn_path = str(video_path.with_suffix('.pgn'))
        
    if args.save_video:
        video_output_path = str(video_path.with_suffix('.annotated.mp4'))
        
    # Process video
    result = processor.process_video(
        video_path=str(video_path),
        output_pgn_path=pgn_path,
        output_video_path=video_output_path,
        starting_fen=args.fen,
        manual_corners=corners,
        crop_top_half=not args.no_crop
    )
    
    return result


def get_video_files(input_path: Path) -> List[Path]:
    """Get list of video files from path."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        files = []
        for ext in video_extensions:
            files.extend(input_path.glob(f'*{ext}'))
            files.extend(input_path.glob(f'*{ext.upper()}'))
        return sorted(files)
    else:
        return []


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get input path
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
        
    # Get video files to process
    video_files = get_video_files(input_path)
    
    if not video_files:
        print(f"Error: No video files found at: {input_path}")
        sys.exit(1)
        
    print(f"Found {len(video_files)} video(s) to process")
    
    if args.use_v2:
        print("Using IMPROVED processor (V2)")
    
    # Handle output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
        if args.batch and not output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            
    # Calibration if requested (for V1 only)
    corners = None
    if args.calibrate and not args.use_v2 and len(video_files) > 0:
        if interactive_calibration is not None:
            print("\nStarting calibration...")
            corners = interactive_calibration(str(video_files[0]))
            if corners is None:
                print("Calibration cancelled")
                sys.exit(0)
            print("Calibration complete!")
        
    # Process videos
    results = []
    
    for i, video_file in enumerate(video_files):
        try:
            # Determine output for this video
            if output_path:
                if output_path.is_dir():
                    file_output = output_path / video_file.with_suffix('.pgn').name
                else:
                    file_output = output_path
            else:
                file_output = None
                
            # Use V2 or V1 processor
            if args.use_v2:
                result = process_single_video_v2(video_file, file_output, args)
            else:
                result = process_single_video(
                    video_file,
                    file_output,
                    args,
                    corners=corners
                )
            
            results.append({
                'video': str(video_file),
                'pgn': result['pgn'],
                'moves': len(result['moves']),
                'success': True
            })
            
        except Exception as e:
            import traceback
            print(f"Error processing {video_file}: {e}")
            traceback.print_exc()
            results.append({
                'video': str(video_file),
                'error': str(e),
                'success': False
            })
            
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    print(f"Processed: {successful}/{len(results)} videos successfully")
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        video_name = Path(result['video']).name
        if result['success']:
            print(f"  {status} {video_name}: {result['moves']} moves detected")
        else:
            print(f"  {status} {video_name}: {result.get('error', 'Unknown error')}")
            
    print()
    
    return 0 if successful == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
