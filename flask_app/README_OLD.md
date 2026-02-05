# Chess Video Analyzer - Flask Version

A simple Flask web application to analyze chess game videos and automatically detect moves.

## Features

- 🎥 **Video Upload**: Upload chess game videos (MP4, AVI, MOV, MKV)
- 📹 **Sample Videos**: Process pre-loaded sample videos
- ♟️ **Move Detection**: Automatically detect chess moves using YOLO-based detection
- 📄 **PGN Export**: Generate and download PGN files
- 🎨 **Clean UI**: Modern, responsive interface with Bootstrap 5

## Quick Start

### 1. Install Dependencies

```bash
cd flask_app
pip install -r requirements.txt
```

### 2. Download Models

Make sure the ONNX models are downloaded. If not, run:

```bash
cd ../redjohn
python download_models.py
cd ../flask_app
```

### 3. Run the Application

```bash
python app.py
```

### 4. Open Your Browser

Navigate to: **http://localhost:5000**

## Project Structure

```
flask_app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML page
├── static/
│   └── app.js            # Frontend JavaScript
├── utils/
│   ├── detector.py       # Chess piece detection
│   ├── video_processor.py # Video processing
│   └── board_tracker.py  # Board state tracking
├── uploads/              # Uploaded videos (created automatically)
└── output/               # Generated PGN files (created automatically)
```

## How to Use

### Upload a Video

1. Go to the "Upload Video" tab
2. Drag & drop a video or click to browse
3. Click "Process Video"
4. Wait for processing to complete
5. View detected moves and download PGN

### Use Sample Videos

1. Go to the "Sample Videos" tab
2. Click on any sample video
3. Click "Process Selected Video"
4. View results in the "Results" tab

## API Endpoints

- `GET /` - Main page
- `GET /api/init` - Initialize detector
- `POST /api/upload` - Upload video file
- `POST /api/process` - Process uploaded video
- `POST /api/process-sample/<video_name>` - Process sample video
- `GET /api/download/<filename>` - Download PGN file
- `GET /api/sample-videos` - List available sample videos

## Troubleshooting

### Models Not Found

If you see "Models not found" error:
```bash
cd ../redjohn
python download_models.py
```

### Port Already in Use

Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001
```

### Processing Too Slow

The app processes every 30th frame by default. To change:
- Edit `app.py` line with `if frames_processed % 30 == 0:`
- Increase number (e.g., 60) for faster processing (less accuracy)
- Decrease number (e.g., 15) for slower processing (more accuracy)

## Notes

- Maximum upload size: 500MB
- Supported formats: MP4, AVI, MOV, MKV
- Processing speed depends on video length and system performance
- Models use float16 precision for efficiency

## Credits

Built with:
- Flask (Backend)
- Bootstrap 5 (UI)
- OpenCV (Video Processing)
- ONNX Runtime (Model Inference)
- python-chess (Chess Logic)
