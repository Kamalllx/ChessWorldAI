"""
Chess Video Analyzer - Flask App with Position-Based Tracking
Trusts starting position, uses YOLO only for presence detection.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
import json
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
import queue

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.position_tracker import PositionBasedTracker, SQUARE_NAMES, STARTING_POSITION

app = Flask(__name__)
CORS(app)

# Configuration
OUTPUT_FOLDER = 'output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Global state
analyzer = None
processing_queue = queue.Queue()
stop_event = Event()


def get_analyzer():
    """Get or create the analyzer instance."""
    global analyzer
    if analyzer is None:
        pieces_path = "480M_pieces_float16"
        corners_path = "480L_xcorners_float16"
        
        if not os.path.exists(os.path.join(pieces_path, "model.onnx")):
            raise FileNotFoundError(f"Pieces model not found at {pieces_path}")
        if not os.path.exists(os.path.join(corners_path, "model.onnx")):
            raise FileNotFoundError(f"Corners model not found at {corners_path}")
        
        analyzer = PositionBasedTracker(pieces_path, corners_path)
    
    return analyzer


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('app.html')


@app.route('/api/init', methods=['GET'])
def init():
    """Initialize the analyzer."""
    try:
        get_analyzer()
        return jsonify({'success': True, 'message': 'Analyzer initialized'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/sample-videos', methods=['GET'])
def list_sample_videos():
    """List available sample videos from multiple possible locations."""
    videos = []
    
    # Check multiple directories
    search_paths = [
        Path("../videos"),
        Path("videos"),
        Path("../videos copy"),
    ]
    
    for videos_path in search_paths:
        if videos_path.exists():
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                for f in videos_path.glob(ext):
                    videos.append({
                        'name': f.name,
                        'path': str(f.resolve())
                    })
    
    return jsonify({'success': True, 'videos': videos})


@app.route('/api/get-frame', methods=['POST'])
def get_frame():
    """Get a frame from a video."""
    try:
        data = request.json
        video_path = data.get('videoPath')
        frame_number = data.get('frameNumber', 0)
        
        if not video_path:
            return jsonify({'success': False, 'message': 'No video path provided'}), 400
        
        # Handle relative paths for sample videos
        if not os.path.isabs(video_path):
            for base in ["../videos", "videos", "../videos copy"]:
                test_path = os.path.join(base, video_path)
                if os.path.exists(test_path):
                    video_path = test_path
                    break
        
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'message': f'Video not found: {video_path}'}), 404
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Failed to open video'}), 500
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_number, total_frames - 1))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to read frame'}), 500
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'frame': f'data:image/jpeg;base64,{frame_base64}',
            'totalFrames': total_frames,
            'fps': fps,
            'width': frame.shape[1],
            'height': frame.shape[0]
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/find-corners', methods=['POST'])
def find_corners():
    """Find board corners from a frame image."""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided'}), 400
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'message': 'Failed to decode image'}), 400
        
        # Find corners
        analyzer = get_analyzer()
        result = analyzer.find_corners_auto(frame)
        
        if not result['success']:
            return jsonify(result)
        
        # Draw visualization
        vis_frame = frame.copy()
        corners = result['corners']
        
        # Draw corner points and labels
        colors = {'a1': (0, 255, 0), 'h1': (0, 200, 255), 'h8': (255, 0, 0), 'a8': (255, 0, 255)}
        for key, point in corners.items():
            x, y = int(point[0]), int(point[1])
            color = colors.get(key, (0, 255, 0))
            cv2.circle(vis_frame, (x, y), 15, color, -1)
            cv2.circle(vis_frame, (x, y), 17, (255, 255, 255), 2)
            cv2.putText(vis_frame, key, (x + 20, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw board outline
        pts = np.array([corners['a1'], corners['h1'], corners['h8'], corners['a8']], dtype=np.int32)
        cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 3)
        
        # Draw X-corners if available
        if 'xcorners' in result and result['xcorners']:
            for xc in result['xcorners']:
                cv2.circle(vis_frame, (int(xc[0]), int(xc[1])), 4, (255, 165, 0), -1)
        
        # Encode visualization
        _, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        vis_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'message': result['message'],
            'corners': corners,
            'visualization': f'data:image/jpeg;base64,{vis_base64}'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/analyze-initial', methods=['POST'])
def analyze_initial():
    """Analyze the initial frame with smart tracking - shows expected vs detected."""
    try:
        data = request.json
        video_path = data.get('videoPath')
        corners = data.get('corners')
        
        if not corners:
            return jsonify({'success': False, 'message': 'No corners provided'}), 400
        
        # Resolve video path
        if video_path and not os.path.isabs(video_path):
            for base in ["../videos", "videos", "../videos copy"]:
                test_path = os.path.join(base, video_path)
                if os.path.exists(test_path):
                    video_path = test_path
                    break
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'success': False, 'message': f'Video not found: {video_path}'}), 404
        
        # Read first frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Failed to open video'}), 500
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'message': 'Failed to read first frame'}), 500
        
        # Set corners and analyze with smart tracker
        analyzer = get_analyzer()
        analyzer.reset()
        analyzer.set_corners(corners)
        
        result = analyzer.analyze_initial_frame(frame)
        
        if not result['success']:
            return jsonify(result), 400
        
        # Draw debug overlay with all 32 expected pieces shown
        vis_frame = frame.copy()
        
        # Draw grid and boundary
        if analyzer.boundary is not None:
            pts = analyzer.boundary.astype(np.int32)
            cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 2)
            for i, name in enumerate(['a1', 'h1', 'h8', 'a8']):
                x, y = int(analyzer.boundary[i][0]), int(analyzer.boundary[i][1])
                cv2.circle(vis_frame, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(vis_frame, name, (x + 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw all 64 square labels
        if analyzer.centers is not None:
            for i, name in enumerate(SQUARE_NAMES):
                cx, cy = int(analyzer.centers[i][0]), int(analyzer.centers[i][1])
                cv2.putText(vis_frame, name, (cx - 8, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)
        
        # Draw detection results with match/mismatch indicators
        for det_result in result['detection_results']:
            if det_result.get('box'):
                box = det_result['box']
                l, t, r, b = [int(v) for v in box]
                
                detected = det_result['detected_piece']
                expected = det_result['expected_piece']
                conf = det_result['confidence']
                sq_name = det_result['square_name']
                match = det_result['match']
                
                # Green for match, red for mismatch, yellow for unexpected detection
                if match and expected:
                    color = (0, 255, 0)  # Green - correct detection
                elif expected and detected != expected:
                    color = (0, 165, 255)  # Orange - wrong piece detected
                elif not expected and detected:
                    color = (0, 255, 255)  # Yellow - unexpected piece
                else:
                    color = (0, 0, 255)  # Red - error
                
                cv2.rectangle(vis_frame, (l, t), (r, b), color, 2)
                
                label = f"{detected}@{sq_name} {conf:.0%}"
                if expected and detected != expected:
                    label += f" (exp:{expected})"
                
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(vis_frame, (l, t - lh - 4), (l + lw + 4, t), color, -1)
                cv2.putText(vis_frame, label, (l + 2, t - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw indicators for expected but not detected pieces
        if analyzer.centers is not None:
            for det_result in result['detection_results']:
                if det_result['expected_piece'] and not det_result['detected_piece']:
                    sq_idx = det_result['square_idx']
                    cx, cy = int(analyzer.centers[sq_idx][0]), int(analyzer.centers[sq_idx][1])
                    
                    # Draw red circle for missing piece
                    cv2.circle(vis_frame, (cx, cy), 15, (0, 0, 255), 2)
                    cv2.putText(vis_frame, f"?{det_result['expected_piece']}", (cx - 8, cy + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw mini board with expected position
        h, w = vis_frame.shape[:2]
        size = min(180, w // 5)
        sq = size // 8
        ox, oy = w - size - 10, 10
        
        cv2.rectangle(vis_frame, (ox - 2, oy - 2), (ox + size + 2, oy + size + 2), (40, 40, 40), -1)
        
        for rank in range(8):
            for file in range(8):
                x = ox + file * sq
                y = oy + (7 - rank) * sq
                is_light = (file + rank) % 2 == 1
                color = (240, 217, 181) if is_light else (181, 136, 99)
                cv2.rectangle(vis_frame, (x, y), (x + sq, y + sq), color, -1)
        
        for sq_idx, piece in STARTING_POSITION.items():
            file_idx = sq_idx % 8
            rank_idx = sq_idx // 8
            x = ox + file_idx * sq + sq // 2
            y = oy + (7 - rank_idx) * sq + sq // 2
            
            is_white = piece.isupper()
            text_color = (255, 255, 255) if is_white else (0, 0, 0)
            cv2.circle(vis_frame, (x, y), sq // 3, (0, 0, 0) if is_white else (255, 255, 255), -1)
            cv2.putText(vis_frame, piece.upper(), (x - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)
        
        # Encode visualization
        _, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        vis_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'message': result['message'],
            'num_expected': result['num_expected'],
            'num_detected_correct': result['num_detected_correct'],
            'num_detected_wrong': result['num_detected_wrong'],
            'num_not_detected': result['num_not_detected'],
            'detection_results': result['detection_results'],
            'visualization': f'data:image/jpeg;base64,{vis_base64}'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/process-video-stream')
def process_video_stream():
    """Process video with Server-Sent Events for real-time updates."""
    video_path = request.args.get('videoPath')
    corners_json = request.args.get('corners')
    speed = float(request.args.get('speed', 1.0))  # Speed control: 0.25 to 2.0
    
    if not video_path or not corners_json:
        def error_gen():
            yield f"data: {json.dumps({'error': 'Missing parameters'})}\n\n"
        return Response(error_gen(), mimetype='text/event-stream')
    
    try:
        corners = json.loads(corners_json)
    except:
        def error_gen():
            yield f"data: {json.dumps({'error': 'Invalid corners JSON'})}\n\n"
        return Response(error_gen(), mimetype='text/event-stream')
    
    # Clamp speed
    speed = max(0.1, min(2.0, speed))
    
    # Resolve video path
    if not os.path.isabs(video_path):
        for base in ["../videos", "videos", "../videos copy"]:
            test_path = os.path.join(base, video_path)
            if os.path.exists(test_path):
                video_path = test_path
                break
    
    def generate():
        try:
            analyzer = get_analyzer()
            analyzer.reset()
            analyzer.set_corners(corners)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield f"data: {json.dumps({'error': 'Failed to open video'})}\n\n"
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            frame_count = 0
            # Lower speed = process more frames = more accurate
            # At speed 0.5, process 6 fps instead of 3
            # At speed 0.25, process 12 fps (every frame at 12fps video)
            process_fps = int(3 / speed)  # Base is 3 fps, adjusted by speed
            process_interval = max(1, int(fps / process_fps))
            
            # Delay between sending frames to simulate slower playback
            frame_delay = (1.0 / process_fps) / speed if speed < 1 else 0
            
            yield f"data: {json.dumps({'type': 'start', 'totalFrames': total_frames, 'fps': fps, 'speed': speed})}\n\n"
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % process_interval == 0:
                    timestamp = frame_count / fps * 1000
                    result = analyzer.process_frame(frame, timestamp)
                    
                    # Draw full debug overlay - pass the full result dict, not just detections
                    vis_frame = analyzer.draw_debug_overlay(frame, result)
                    
                    # Encode frame with overlay
                    _, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    update = {
                        'type': 'frame',
                        'frameNumber': frame_count,
                        'progress': frame_count / total_frames * 100,
                        'numPieces': result.get('num_pieces', 0),
                        'moves': result.get('moves', []),
                        'newMove': result.get('new_move'),
                        'fen': result.get('fen', ''),
                        'stateSummary': result.get('state_summary', {}),
                        'frame': f'data:image/jpeg;base64,{frame_base64}'
                    }
                    
                    yield f"data: {json.dumps(update)}\n\n"
                    
                    # Add delay for slower playback
                    if frame_delay > 0:
                        time.sleep(frame_delay)
                
                frame_count += 1
            
            cap.release()
            
            # Final result
            pgn = analyzer.generate_pgn()
            
            # Save PGN
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            pgn_filename = f"{video_name}_{datetime.now().strftime('%H%M%S')}.pgn"
            pgn_path = os.path.join(app.config['OUTPUT_FOLDER'], pgn_filename)
            
            with open(pgn_path, 'w') as f:
                f.write(pgn)
            
            final_result = {
                'type': 'complete',
                'moves': analyzer.moves,
                'pgn': pgn,
                'pgnFile': pgn_filename,
                'totalFrames': total_frames,
                'framesProcessed': frame_count // process_interval
            }
            
            yield f"data: {json.dumps(final_result)}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/process-video', methods=['POST'])
def process_video_api():
    """Process a video (non-streaming version)."""
    try:
        data = request.json
        video_path = data.get('videoPath')
        corners = data.get('corners')
        
        if not corners:
            return jsonify({'success': False, 'message': 'No corners provided'}), 400
        
        # Resolve video path
        if video_path and not os.path.isabs(video_path):
            for base in ["../videos", "videos", "../videos copy"]:
                test_path = os.path.join(base, video_path)
                if os.path.exists(test_path):
                    video_path = test_path
                    break
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'success': False, 'message': f'Video not found: {video_path}'}), 404
        
        analyzer = get_analyzer()
        result = process_video(video_path, corners, analyzer)
        
        if not result['success']:
            return jsonify(result), 500
        
        # Save PGN
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        pgn_filename = f"{video_name}_{datetime.now().strftime('%H%M%S')}.pgn"
        pgn_path = os.path.join(app.config['OUTPUT_FOLDER'], pgn_filename)
        
        with open(pgn_path, 'w') as f:
            f.write(result['pgn'])
        
        return jsonify({
            'success': True,
            'message': f'Processed {result["frames_processed"]} frames',
            'moves': result['moves'],
            'pgn': result['pgn'],
            'pgn_file': pgn_filename
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download a generated file."""
    try:
        return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 404


if __name__ == '__main__':
    print("=" * 60)
    print("  ♔ Chess Video Analyzer - Real-time Processing ♔")
    print("=" * 60)
    print()
    print("  Features:")
    print("  • Real-time video playback with detection overlay")
    print("  • Live move detection and PGN generation")
    print("  • Automatic and manual corner detection")
    print()
    print("  Workflow:")
    print("  1. Select a video from the list")
    print("  2. Find corners (auto or manual)")
    print("  3. Watch real-time processing with live moves")
    print("  4. Download the generated PGN")
    print()
    print("  Server: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
