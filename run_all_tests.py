"""
Batch test runner for all chess video trackers.

Runs all 11 tracker implementations on all 5 test videos and generates PGNs.
Creates a results matrix showing which combinations succeeded.

Usage:
    python run_all_tests.py
    python run_all_tests.py --trackers v1,v3  # Only run specific trackers
    python run_all_tests.py --videos game_1,game_4  # Only run specific videos
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import time
from datetime import datetime

# Define all trackers
NEW_TRACKERS = [
    ("V1_Position", "trackers/v1_position_based.py"),
    ("V2_Model", "trackers/v2_model_based.py"),
    ("V3_Hybrid", "trackers/v3_hybrid.py"),
    ("V4_OpticalFlow", "trackers/v4_optical_flow.py"),
]

LEGACY_TRACKERS = [
    ("CameraChess", "legacy/run_camerachess.py"),
    ("V2_Fast", "legacy/run_tracker_v2.py"),
    ("V3_Direct", "legacy/run_tracker_v3.py"),
    ("V4_Ensemble", "legacy/run_tracker_v4.py"),
    ("V5_Phases", "legacy/run_tracker_v5.py"),
    ("V2_Enhanced", "legacy/run_tracker_v2_enhanced.py"),
]

ALL_TRACKERS = NEW_TRACKERS + LEGACY_TRACKERS

TEST_VIDEOS = [
    "videos/game_1.mp4",
    "videos/game_2.mp4",
    "videos/game_3.mp4",
    "videos/game_4.mp4",
    "videos/game_5.mp4",
]


class TestRunner:
    """Manages batch testing of all trackers."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results: Dict[str, Dict[str, Tuple[bool, float, str]]] = {}
        
    def run_tracker(self, tracker_name: str, tracker_path: str, video_path: str) -> Tuple[bool, float, str]:
        """
        Run a single tracker on a single video.
        
        Returns:
            (success, duration, error_msg)
        """
        full_tracker_path = self.base_dir / tracker_path
        full_video_path = self.base_dir / video_path
        
        if not full_tracker_path.exists():
            return False, 0.0, f"Tracker not found: {tracker_path}"
        
        if not full_video_path.exists():
            return False, 0.0, f"Video not found: {video_path}"
        
        print(f"  Running {tracker_name} on {video_path}...")
        
        start_time = time.time()
        
        try:
            # Run the tracker (suppress video window)
            result = subprocess.run(
                [sys.executable, str(full_tracker_path), str(full_video_path), "--no-display"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(self.base_dir)
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return True, duration, ""
            else:
                error = result.stderr[-200:] if result.stderr else "Unknown error"
                return False, duration, error
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return False, duration, "Timeout (>5 minutes)"
            
        except Exception as e:
            duration = time.time() - start_time
            return False, duration, str(e)
    
    def run_all_tests(self, tracker_filter: List[str] = None, video_filter: List[str] = None):
        """Run all combinations of trackers and videos."""
        
        # Filter trackers if specified
        if tracker_filter:
            trackers = [(name, path) for name, path in ALL_TRACKERS 
                       if any(f.lower() in name.lower() for f in tracker_filter)]
        else:
            trackers = ALL_TRACKERS
        
        # Filter videos if specified
        if video_filter:
            videos = [v for v in TEST_VIDEOS 
                     if any(f.lower() in v.lower() for f in video_filter)]
        else:
            videos = TEST_VIDEOS
        
        total_tests = len(trackers) * len(videos)
        current_test = 0
        
        print(f"\n{'='*80}")
        print(f"ChessWorldAI Batch Test Runner")
        print(f"{'='*80}")
        print(f"Trackers: {len(trackers)}")
        print(f"Videos: {len(videos)}")
        print(f"Total tests: {total_tests}")
        print(f"{'='*80}\n")
        
        # Initialize results dictionary
        for tracker_name, _ in trackers:
            self.results[tracker_name] = {}
        
        # Run all combinations
        for video in videos:
            video_name = Path(video).stem
            print(f"\n[{video_name}]")
            
            for tracker_name, tracker_path in trackers:
                current_test += 1
                print(f"[{current_test}/{total_tests}] ", end="")
                
                success, duration, error = self.run_tracker(tracker_name, tracker_path, video)
                self.results[tracker_name][video_name] = (success, duration, error)
                
                if success:
                    print(f"    ✓ Success ({duration:.1f}s)")
                else:
                    print(f"    ✗ Failed ({duration:.1f}s): {error[:50]}")
        
        print(f"\n{'='*80}")
        print("All tests completed!")
        print(f"{'='*80}\n")
    
    def print_results(self):
        """Print a results matrix."""
        
        # Get all video names
        video_names = sorted(set(
            video_name 
            for tracker_results in self.results.values() 
            for video_name in tracker_results.keys()
        ))
        
        if not video_names:
            print("No results to display.")
            return
        
        print("\n" + "="*100)
        print("RESULTS MATRIX")
        print("="*100)
        print(f"{'Tracker':<20}", end="")
        for video_name in video_names:
            print(f"{video_name:>12}", end="")
        print(f"  {'Success Rate':>12}  {'Avg Time':>10}")
        print("-"*100)
        
        for tracker_name in self.results:
            print(f"{tracker_name:<20}", end="")
            
            successes = 0
            total_time = 0.0
            count = 0
            
            for video_name in video_names:
                if video_name in self.results[tracker_name]:
                    success, duration, _ = self.results[tracker_name][video_name]
                    symbol = "✓" if success else "✗"
                    print(f"{symbol:>12}", end="")
                    
                    if success:
                        successes += 1
                    total_time += duration
                    count += 1
                else:
                    print(f"{'—':>12}", end="")
            
            if count > 0:
                success_rate = f"{successes}/{count}"
                avg_time = total_time / count
                print(f"  {success_rate:>12}  {avg_time:>9.1f}s")
            else:
                print(f"  {'0/0':>12}  {'—':>10}")
        
        print("="*100)
        print()
    
    def save_report(self):
        """Save a detailed report to file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.base_dir / "output" / f"test_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write(f"ChessWorldAI Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
            
            for tracker_name, video_results in self.results.items():
                f.write(f"\n{tracker_name}\n")
                f.write("-" * 50 + "\n")
                
                for video_name, (success, duration, error) in video_results.items():
                    status = "SUCCESS" if success else "FAILED"
                    f.write(f"  {video_name}: {status} ({duration:.1f}s)\n")
                    if error:
                        f.write(f"    Error: {error}\n")
                
                # Summary
                successes = sum(1 for s, _, _ in video_results.values() if s)
                total = len(video_results)
                avg_time = sum(d for _, d, _ in video_results.values()) / total if total > 0 else 0
                
                f.write(f"\n  Summary: {successes}/{total} successful, avg {avg_time:.1f}s\n")
        
        print(f"Detailed report saved to: {report_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all chess video trackers on all test videos")
    parser.add_argument("--trackers", type=str, help="Comma-separated list of tracker names to run (e.g., 'v1,v3,camerachess')")
    parser.add_argument("--videos", type=str, help="Comma-separated list of video names to test (e.g., 'game_1,game_4')")
    parser.add_argument("--no-report", action="store_true", help="Don't save detailed report")
    
    args = parser.parse_args()
    
    # Parse filters
    tracker_filter = args.trackers.split(',') if args.trackers else None
    video_filter = args.videos.split(',') if args.videos else None
    
    # Get base directory
    base_dir = Path(__file__).parent
    
    # Create test runner
    runner = TestRunner(base_dir)
    
    # Run tests
    runner.run_all_tests(tracker_filter, video_filter)
    
    # Print results
    runner.print_results()
    
    # Save report
    if not args.no_report:
        runner.save_report()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
