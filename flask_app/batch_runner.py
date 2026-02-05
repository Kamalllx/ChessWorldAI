"""
Batch process all videos and generate PGN files for submission
Run this to test both approaches on all 5 videos
"""

import os
import sys
import subprocess
from pathlib import Path

# Video files to process
VIDEOS = ["game_1.mp4", "game_2.mp4", "game_3.mp4", "game_4.mp4", "game_5.mp4"]
VIDEO_DIR = Path("videos")
OUTPUT_DIR = Path("output")

def main():
    print("=" * 60)
    print("BATCH VIDEO PROCESSING - Flask Approach")
    print("=" * 60)
    print()
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Check if videos exist
    missing_videos = []
    for video in VIDEOS:
        video_path = VIDEO_DIR / video
        if not video_path.exists():
            missing_videos.append(video)
    
    if missing_videos:
        print(f"❌ Missing videos: {', '.join(missing_videos)}")
        print(f"   Please place videos in: {VIDEO_DIR.absolute()}")
        return 1
    
    print(f"✅ Found all {len(VIDEOS)} videos")
    print()
    
    # Process each video
    for i, video in enumerate(VIDEOS, 1):
        video_path = VIDEO_DIR / video
        output_name = f"game_{i}.pgn"
        output_path = OUTPUT_DIR / output_name
        
        print(f"[{i}/{len(VIDEOS)}] Processing: {video}")
        print(f"    Video: {video_path}")
        print(f"    Output: {output_path}")
        
        try:
            # Run the video processor
            result = subprocess.run(
                [sys.executable, "process_single_video.py", str(video_path), str(output_path)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per video
            )
            
            if result.returncode == 0:
                print(f"    ✅ Success! PGN saved to {output_path}")
            else:
                print(f"    ❌ Failed with error:")
                print(f"       {result.stderr}")
        
        except subprocess.TimeoutExpired:
            print(f"    ⏱️  Timeout - video took too long to process")
        except Exception as e:
            print(f"    ❌ Error: {e}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    
    pgn_files = list(OUTPUT_DIR.glob("*.pgn"))
    print(f"\n📁 Generated {len(pgn_files)} PGN files:")
    for pgn in sorted(pgn_files):
        size = pgn.stat().st_size
        print(f"   - {pgn.name} ({size} bytes)")
    
    print(f"\n📂 Output directory: {OUTPUT_DIR.absolute()}")
    print("\n✅ All done! Ready for submission.")
    
    return 0

if __name__ == "__main__":
    exit(main())
