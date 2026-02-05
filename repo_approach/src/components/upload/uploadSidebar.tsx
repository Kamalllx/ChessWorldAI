import { VideoButton, PlayButton, RestartButton, PlaybackButtons, StopButton } from "./buttons";
import { CornersButton, Sidebar, FenButton, TrackingModeButton } from "../common";
import { SetBoolean, SetStringArray } from "../../types";
import { settingsSelect } from "../../slices/settingsSlice";

const UploadSidebar = ({ videoRef, xcornersModelRef, piecesModelRef, canvasRef, 
  sidebarRef, text, setText, playing, setPlaying, cornersRef }: {
  videoRef: any, xcornersModelRef: any, piecesModelRef: any, canvasRef: any, sidebarRef: any,
  text: string[], setText: SetStringArray,
  playing: boolean, setPlaying: SetBoolean,
  cornersRef: any
}) => {
  const settings = settingsSelect();

  const inputStyle = {
    display: playing ? "none": "inline-block"
  }

  return (
    <Sidebar sidebarRef={sidebarRef} playing={playing} text={text} setText={setText} >
      <li className="my-1" style={inputStyle}>
        <div className="text-white text-start p-2 mb-2" style={{"backgroundColor": "#34495e", "borderRadius": "5px", "fontSize": "0.85rem"}}>
          <strong>📹 Step 1:</strong> Upload Video<br/>
          <strong>🎚️ Step 2:</strong> Set Speed (0.5x recommended)<br/>
          <strong>📍 Step 3:</strong> Find/Adjust Corners<br/>
          <strong>♟️ Step 4:</strong> Detect Position<br/>
          <strong>▶️ Step 5:</strong> Play Analysis
        </div>
      </li>
      <li className="my-1" style={inputStyle}>
        <div className="alert alert-info p-2" style={{"fontSize": "0.75rem", "margin": "0"}}>
          {settings.usePositionTracking ? (
            <>
              <strong>🎯 Position Tracking</strong><br/>
              Tracks all 32 pieces from starting position. More accurate!
            </>
          ) : (
            <>
              <strong>📊 Score-Based</strong><br/>
              Original algorithm using confidence scores.
            </>
          )}
        </div>
      </li>
      <li className="my-1" style={inputStyle}>
        <TrackingModeButton />
      </li>
      <li className="my-1" style={inputStyle}>
        <VideoButton videoRef={videoRef} canvasRef={canvasRef} setPlaying={setPlaying} />
      </li>
      <li className="my-1" style={inputStyle}>
        <CornersButton piecesModelRef={piecesModelRef} xcornersModelRef={xcornersModelRef} 
        videoRef={videoRef} canvasRef={canvasRef} setText={setText} />
      </li>
      <li className="my-1" style={inputStyle}>
        <FenButton piecesModelRef={piecesModelRef} videoRef={videoRef} 
        canvasRef={canvasRef} setText={setText} cornersRef={cornersRef} />
      </li>
      <li className="my-1" style={inputStyle}>
        <small className="text-white d-block mb-1">⚡ Playback Speed:</small>
        <PlaybackButtons videoRef={videoRef} />
      </li>
      <li className="my-1">
        <div className="btn-group w-100" role="group">
          <PlayButton videoRef={videoRef} playing={playing} setPlaying={setPlaying} />
          <StopButton videoRef={videoRef} setPlaying={setPlaying} setText={setText} />
          <RestartButton videoRef={videoRef} setText={setText} />
        </div>
      </li>
    </Sidebar>
  );
};

export default UploadSidebar;