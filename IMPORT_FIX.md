# Quick Fix Guide - Import Issues After Organization

## Issue
After moving trackers to `legacy/` subdirectory, import paths broke with error:
```
ModuleNotFoundError: No module named 'src'
```

## Root Cause
Legacy files used `sys.path.insert(0, str(Path(__file__).parent))` which added the `legacy/` directory to Python path. After moving files, this pointed to the wrong directory.

## Solution Applied
Changed all legacy files from:
```python
sys.path.insert(0, str(Path(__file__).parent))  # Points to legacy/
```

To:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))  # Points to project root
```

## Files Fixed
✅ All 10 legacy tracker files:
- run_tracker_v2.py
- run_tracker_v3.py
- run_tracker_v4.py
- run_tracker_v5.py
- run_tracker_v2_enhanced.py
- run_camerachess.py
- run_assume.py
- run_smart.py
- run_hybrid.py
- run_tracker_fast.py

## Verification
Run any legacy tracker:
```bash
python legacy/run_tracker_v2.py videos/game_1.mp4
python legacy/run_camerachess.py videos/game_4.mp4 --debug
```

Should work without import errors.

## Why This Matters
When organizing code into subdirectories, relative imports must be updated to maintain correct path resolution. Using `.parent.parent` ensures we always reference the project root where `src/` lives.
