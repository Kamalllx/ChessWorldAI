# 🚀 QUICK START GUIDE - Flask Chess Video Analyzer

## ✅ Your app is ready to use!

### Server Status
- ✅ Flask server is running on **http://localhost:5000**
- ✅ All dependencies installed
- ✅ Models downloaded and ready

### How to Use

#### Option 1: Open in Browser
1. **Open your browser** and go to: `http://localhost:5000`
2. You'll see a beautiful interface with 3 tabs:
   - **Upload Video** - Upload your own chess game videos
   - **Sample Videos** - Process videos from the `videos` folder
   - **Results** - View detected moves and download PGN

#### Option 2: Try Sample Videos
1. Click on the **"Sample Videos"** tab
2. Select any video (e.g., game_1.mp4, game_2.mp4)
3. Click **"Process Selected Video"**
4. Wait for processing (may take 1-2 minutes)
5. View results in the **"Results"** tab

#### Option 3: Upload Your Own Video
1. Click on the **"Upload Video"** tab
2. Drag & drop a video or click to browse
3. Supported formats: MP4, AVI, MOV, MKV (max 500MB)
4. Click **"Process Video"**
5. Wait for processing to complete
6. Download the generated PGN file

### Features
- 🎥 **Drag & Drop Upload** - Easy video upload
- 📹 **Sample Videos** - Test with pre-loaded videos
- ♟️ **Auto Detection** - Automatic chess move detection
- 📄 **PGN Export** - Download moves in PGN format
- 🎨 **Modern UI** - Clean, responsive Bootstrap 5 interface
- ⚡ **Real-time Processing** - See status updates

### Troubleshooting

#### If the page doesn't load:
- Make sure the Flask server is running (check the terminal)
- Try refreshing the page
- Check firewall settings

#### If processing fails:
- Make sure models are downloaded in `../public/` folder
- Check that the video file is valid
- Look at terminal output for error messages

#### To stop the server:
- Press `Ctrl+C` in the terminal

#### To restart the server:
```bash
cd flask_app
python app.py
```

### Project Structure
```
flask_app/
├── app.py                    # Flask backend
├── templates/
│   └── index.html           # Frontend HTML
├── static/
│   └── app.js               # Frontend JavaScript
├── utils/
│   ├── detector.py          # Chess detection
│   ├── video_processor.py   # Video processing
│   └── board_tracker.py     # Move tracking
├── uploads/                 # Uploaded videos
└── output/                  # Generated PGN files
```

### API Endpoints
- `GET /` - Main page
- `GET /api/init` - Initialize detector
- `POST /api/upload` - Upload video
- `POST /api/process` - Process uploaded video
- `POST /api/process-sample/<name>` - Process sample video
- `GET /api/download/<file>` - Download PGN
- `GET /api/sample-videos` - List sample videos

### Notes
- Processing speed depends on video length
- The app processes every 30th frame by default
- Models use float16 precision for efficiency
- All processing happens server-side

### Next Steps
1. **Open http://localhost:5000 in your browser**
2. Try processing a sample video first
3. Then upload your own chess game video
4. Download the PGN and import it into chess software

Enjoy! 🎉
