import { Chessboard } from "kokopu-react";
import { HomeButton, PgnButton } from "./index.tsx";
import { Game } from "../../types.tsx";
import { gameSelect } from "../../slices/gameSlice.tsx";

const Sidebar = (props: any) => {
  const game: Game = gameSelect();

  const boardDisplay = () => {
    return (
      <Chessboard 
        turnVisible={false} 
        squareSize={20} 
        position={game.fen}
        coordinateVisible={false}
      />
    );
  }

  const textDisplay = () => {
    return (
      <div className="text-white">
        {props.text.map(function(t: any, i: any){
            return <div key={i}>{t}</div>;
        })}
      </div>
    )
  }

  const buttons = () => {
    return (
      <div className="btn-group w-100" role="group">
        <PgnButton setText={props.setText} playing={props.playing} />
        <HomeButton />
      </div>
    );
  }

  return (
    <div ref={props.sidebarRef} className="d-flex flex-column text-center mx-1" 
      style={{"minWidth": "200px"}}>
      <div className="text-white mb-3 p-2" style={{"backgroundColor": "#2c3e50", "borderRadius": "8px"}}>
        <h4 className="mb-0" style={{"fontWeight": "bold"}}>♟️ ChessVision Pro</h4>
        <small style={{"opacity": 0.8}}>Video to PGN Converter</small>
      </div>
      <ul className="nav nav-pills flex-column">
        <li className="my-1" style={{display: props.playing ? "inline-block": "none"}}>
          {boardDisplay()}
        </li>
        {props.children}
        <li className="border-top"></li>
        <li className="my-1">
          {textDisplay()}
        </li>
        <li className="border-top"></li>
        <li className="my-1">
          {buttons()}
        </li>
      </ul>
    </div>
  );
};

export default Sidebar;