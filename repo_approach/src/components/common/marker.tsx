import Draggable from 'react-draggable';
import React from "react";
import { MARKER_DIAMETER } from "../../utils/constants";
import { useDispatch } from 'react-redux';
import { cornersSet } from '../../slices/cornersSlice';
import { CornersPayload, CornersKey } from '../../types';

const Marker = ({ name, xy }: { name: CornersKey, xy: number[] }) => {
  const boxStyle: React.CSSProperties = {
    "height": MARKER_DIAMETER,
    "width": MARKER_DIAMETER,
    "backgroundColor": "#ff3333",
    "borderRadius": "50%",
    "textAlign": "center",
    "position": "absolute",
    "userSelect": "none",
    "opacity": 0.8,
    "cursor": "move",
    "border": "3px solid white",
    "boxShadow": "0 0 10px rgba(0,0,0,0.5)",
    "zIndex": 1000
  };
  const cursorStyle: React.CSSProperties = {
    "display": "flex",
    "height": "100%",
    "width": "100%",
    "textAlign": "center",
    "justifyContent": "center",
    "alignItems": "center",
    "cursor": "move",
    "fontWeight": "bold",
    "fontSize": "14px",
    "color": "white"
  }
  const nodeRef = React.useRef(null);
  const dispatch = useDispatch();

  return (
    <Draggable
    nodeRef={nodeRef}
    position={{"x": xy[0], "y": xy[1]}}
    onDrag={(_, data) => {
      const payload: CornersPayload = {
        "xy": [data.x, data.y],
        "key": name
      }
      dispatch(cornersSet(payload))
    }}
    >
      <div className="box" style={boxStyle} ref={nodeRef}>
        <div className="cursor" style={cursorStyle}>{name}</div>
      </div>
    </Draggable>
  );
};

export default Marker;