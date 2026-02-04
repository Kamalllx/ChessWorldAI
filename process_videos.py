#!/usr/bin/env python3
"""
ChessWorldAI - Batch Processing Script for Assignment Videos

This script processes all 5 chess game videos from the ChessWorldAI assignment
and generates PGN output files.

Videos are expected to be downloaded from:
https://drive.google.com/drive/folders/1Cc8raUBp4XRVhU1VZGAC44hRlVnNmtjS

USAGE:
    python process_videos.py videos/              # Uses V3 processor (recommended)
    python process_videos.py videos/ --v2         # Use V2 processor 
    python process_videos.py videos/ --v1         # Use original V1 processor
    python process_videos.py videos/ --speed 0.3  # Slower processing for accuracy
    python process_videos.py videos/ --interactive  # Interactive board selection

V3 FEATURES (NEW):
    - Initial position awareness (uses known starting positions)
    - State-based piece tracking (tracks pieces between frames)
    - Two-move lookahead for better accuracy
    - Greedy mode for fast moves
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import V3 processor first (recommended)
try:
    from src.video_processor_v3 import ChessVideoProcessorV3
    HAS_V3 = True
except ImportError:
    HAS_V3 = False
    print("Note: V3 processor not available")

# Try to import V2 processor 
try:
    from src.video_processor_v2 import ImprovedChessVideoProcessor
    HAS_V2 = True
except ImportError:
    HAS_V2 = False

try:
    from src.video_processor import ChessVideoProcessor
    HAS_V1 = True
except ImportError:
    HAS_V1 = False


def process_assignment_videos(
    videos_dir: str, 
    output_dir: str = "output/pgn",
    processor_version: str = "v3",  # "v1", "v2", or "v3"
    speed: float = 0.5,
    crop_ratio: float = 0.45,
    interactive: bool = False,
    state_decay: float = 0.5,  # For V3: how much old state persists
    greedy_mode: bool = False  # For V3: allow faster single-move detection
):
    """
    Process all assignment videos and generate PGN files.
    
    Args:
        videos_dir: Directory containing the downloaded videos
        output_dir: Directory to save PGN outputs
        processor_version: Which processor to use ("v1", "v2", or "v3")
        speed: Playback speed (lower = more accurate, slower)
        crop_ratio: How much of frame to keep from top (0.45 = top 45%)
        interactive: Enable interactive board selection
        state_decay: V3 only - how much old state persists (0-1)
        greedy_mode: V3 only - allow faster single-move detection
    """
    videos_path = Path(videos_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_path.glob(f'*{ext}'))
        video_files.extend(videos_path.glob(f'*{ext.upper()}'))
    
    video_files = sorted(set(video_files))
    
    if not video_files:
        print(f"No video files found in: {videos_path}")
        print("\nPlease download the videos from:")
        print("https://drive.google.com/drive/folders/1Cc8raUBp4XRVhU1VZGAC44hRlVnNmtjS")
        return
    
    # Determine which processor to use
    if processor_version == "v3" and HAS_V3:
        processor_name = "V3 (State-Based Tracking)"
    elif processor_version == "v2" and HAS_V2:
        processor_name = "V2 (Improved)"
    elif HAS_V1:
        processor_name = "V1 (Original)"
    else:
        print("ERROR: No video processor available!")
        return
    
    print(f"Found {len(video_files)} video(s)")
    print(f"Processor: {processor_name}")
    print(f"Speed: {speed}x, Crop ratio: {crop_ratio}")
    if processor_version == "v3":
        print(f"State decay: {state_decay}, Greedy mode: {greedy_mode}")
    print("=" * 60)
    
    # Create processor
    if processor_version == "v3" and HAS_V3:
        processor = ChessVideoProcessorV3(
            visualize=True,
            crop_ratio=crop_ratio,
            playback_speed=speed,
            state_decay=state_decay,
            greedy_mode=greedy_mode
        )
    elif processor_version == "v2" and HAS_V2:
        processor = ImprovedChessVideoProcessor(
            visualize=True,
            crop_ratio=crop_ratio,
            playback_speed=speed,
            stabilization_frames=5
        )
    elif HAS_V1:
        processor = ChessVideoProcessor(
            visualize=True,
            save_annotated_video=False
        )
    else:
        print("ERROR: No video processor available!")
        print("Please check src/video_processor.py, video_processor_v2.py, or video_processor_v3.py")
        return
    
    results = []
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {video_file.name}")
        print("-" * 60)
        
        # Output file path
        pgn_file = output_path / video_file.with_suffix('.pgn').name
        
        try:
            if processor_version == "v3" and HAS_V3:
                result = processor.process_video(
                    video_path=str(video_file),
                    output_pgn_path=str(pgn_file),
                    require_calibration=interactive
                )
            elif processor_version == "v2" and HAS_V2:
                result = processor.process_video(
                    video_path=str(video_file),
                    output_pgn_path=str(pgn_file),
                    require_calibration=interactive
                )
            else:
                result = processor.process_video(
                    video_path=str(video_file),
                    output_pgn_path=str(pgn_file),
                    crop_top_half=True
                )
            
            results.append({
                'video': video_file.name,
                'pgn_file': str(pgn_file),
                'moves': len(result['moves']),
                'success': True,
                'pgn': result['pgn']
            })
            
            print(f"✓ Saved: {pgn_file}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                'video': video_file.name,
                'error': str(e),
                'success': False
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    print(f"Successfully processed: {successful}/{len(results)} videos\n")
    
    for r in results:
        status = "✓" if r['success'] else "✗"
        if r['success']:
            print(f"{status} {r['video']}: {r['moves']} moves → {r['pgn_file']}")
        else:
            print(f"{status} {r['video']}: {r.get('error', 'Unknown error')}")
    
    # Print all PGNs
    print("\n" + "=" * 60)
    print("PGN OUTPUTS")
    print("=" * 60)
    
    for r in results:
        if r['success']:
            print(f"\n--- {r['video']} ---")
            print(r['pgn'])
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process ChessWorldAI assignment videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s videos/                     Process all videos with V3 (state tracking)
  %(prog)s videos/ --v2                Use V2 processor instead
  %(prog)s videos/ --v1                Use original V1 processor
  %(prog)s videos/ --speed 0.3         Process slower for better accuracy
  %(prog)s videos/ --interactive       Interactive board selection
  %(prog)s videos/ --decay 0.7         More persistent state tracking
  %(prog)s videos/ --greedy            Enable greedy mode for fast moves
        """
    )
    parser.add_argument(
        'videos_dir',
        nargs='?',
        default='videos',
        help='Directory containing downloaded videos (default: videos/)'
    )
    parser.add_argument(
        '-o', '--output',
        default='output/pgn',
        help='Output directory for PGN files (default: output/pgn/)'
    )
    parser.add_argument(
        '--v1',
        action='store_true',
        help='Use original V1 processor'
    )
    parser.add_argument(
        '--v2',
        action='store_true',
        help='Use V2 processor (without state tracking)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=0.5,
        help='Playback speed (default: 0.5 = half speed for accuracy)'
    )
    parser.add_argument(
        '--crop-ratio',
        type=float,
        default=0.45,
        help='Fraction of frame to keep from top (default: 0.45)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode - manually select board corners'
    )
    parser.add_argument(
        '--decay',
        type=float,
        default=0.5,
        help='V3 only: State decay factor 0-1 (default: 0.5)'
    )
    parser.add_argument(
        '--greedy',
        action='store_true',
        help='V3 only: Enable greedy mode for faster move detection'
    )
    
    args = parser.parse_args()
    
    # Determine processor version
    if args.v1:
        processor_version = "v1"
    elif args.v2:
        processor_version = "v2"
    else:
        processor_version = "v3"  # Default to V3
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║          ChessWorldAI - Assignment Video Processor           ║
║                                                              ║
║  V3: State-Based Piece Tracking (NEW)                        ║
║  - Uses known starting position                              ║
║  - Tracks pieces instead of detecting from scratch           ║
║  - Two-move lookahead for accuracy                           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    process_assignment_videos(
        args.videos_dir, 
        args.output,
        processor_version=processor_version,
        speed=args.speed,
        crop_ratio=args.crop_ratio,
        interactive=args.interactive,
        state_decay=args.decay,
        greedy_mode=args.greedy
    )


if __name__ == '__main__':
    main()
