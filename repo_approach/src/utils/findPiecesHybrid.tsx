import { renderState} from "./render/renderState";
import * as tf from "@tensorflow/tfjs-core";
import { getInvTransform, transformBoundary, transformCenters } from "./warp";
import { gameUpdate, makeUpdatePayload } from "../slices/gameSlice";
import { getBoxesAndScores, getInput, getXY, invalidVideo } from "./detect";
import { Mode, MovesPair, MovesData } from "../types";
import { zeros } from "./math";
import { CORNER_KEYS } from "./constants";
import { Chess } from "chess.js";
import { PositionBasedTracker } from "./positionTracker";
import { 
  detect as originalDetect, 
  getKeypoints as originalGetKeypoints,
  getSquares,
  getUpdate
} from "./findPieces";

const calculateScore = (state: any, move: MovesData, from_thr=0.6, to_thr=0.6) => {
  let score = 0;
  move.from.forEach(square => {
    score += 1 - Math.max(...state[square]) - from_thr;
  })

  for (let i = 0; i < move.to.length; i++) {
    score += state[move.to[i]][move.targets[i]] - to_thr;
  }

  return score
}

const processState = (state: any, movesPairs: MovesPair[], possibleMoves: Set<string>): {
  bestScore1: number, bestScore2: number, bestJointScore: number, 
  bestMove: MovesData | null, bestMoves: MovesData | null
} => {
  let bestScore1 = Number.NEGATIVE_INFINITY;
  let bestScore2 = Number.NEGATIVE_INFINITY;
  let bestJointScore = Number.NEGATIVE_INFINITY;
  let bestMove: MovesData | null = null;
  let bestMoves: MovesData | null = null;
  const seen: Set<string> = new Set();

  movesPairs.forEach(movePair => {
    if (!(movePair.move1.sans[0] in seen)) {
      seen.add(movePair.move1.sans[0]);
      const score = calculateScore(state, movePair.move1);
      if (score > 0) {
        possibleMoves.add(movePair.move1.sans[0]);
      }
      if (score > bestScore1) {
        bestMove = movePair.move1;
        bestScore1 = score;
      }
    }

    if ((movePair.move2 === null) || (movePair.moves === null) || !(possibleMoves.has(movePair.move1.sans[0]))) {
      return;
    }
    
    const score2: number = calculateScore(state, movePair.move2);
    if (score2 < 0) {
      return;
    } else if (score2 > bestScore2) {
      bestScore2 = score2;
    }

    const jointScore: number = calculateScore(state, movePair.moves);
    if (jointScore > bestJointScore) {
      bestJointScore = jointScore;
      bestMoves = movePair.moves;
    }
  })

  return {bestScore1, bestScore2, bestJointScore, bestMove, bestMoves};
}

const sanToLan = (board: Chess, san: string): string => {
  board.move(san);
  const history: any = board.history({ verbose: true });
  const lan: string = history[history.length - 1].lan;
  board.undo();
  return lan;
}

export const findPiecesHybrid = (
  modelRef: any, 
  videoRef: any, 
  canvasRef: any,
  playingRef: any, 
  setText: any, 
  dispatch: any, 
  cornersRef: any, 
  boardRef: any, 
  movesPairsRef: any, 
  lastMoveRef: any, 
  moveTextRef: any, 
  mode: Mode,
  usePositionTracking: boolean = true
) => {
  let centers: number[][] | null = null;
  let boundary: number[][];
  let centers3D: tf.Tensor3D;
  let boundary3D: tf.Tensor3D;
  let state: number[][];
  let keypoints: number[][];
  let requestId: number;
  
  // Position-based tracker
  let positionTracker: PositionBasedTracker | null = null;
  let framesSinceLastMove = 0;
  let possibleMoves: Set<string>;
  let greedyMoveToTime: { [move: string] : number};

  const loop = async () => {
    if (playingRef.current === false || invalidVideo(videoRef)) {
      centers = null;
      positionTracker = null;
    } else {
      if (centers === null) {
        keypoints = originalGetKeypoints(cornersRef, canvasRef);
        const invTransform = getInvTransform(keypoints);
        [centers, centers3D] = transformCenters(invTransform);
        [boundary, boundary3D] = transformBoundary(invTransform);
        state = zeros(64, 12);
        possibleMoves = new Set<string>();
        greedyMoveToTime = {};
        
        // Initialize position tracker
        if (usePositionTracking) {
          positionTracker = new PositionBasedTracker();
          framesSinceLastMove = 0;
        }
      }
      
      const startTime: number = performance.now();
      const startTensors: number = tf.memory().numTensors;

      const {boxes, scores} = await originalDetect(modelRef, videoRef, keypoints);
      const squares: number[] = getSquares(boxes, centers3D, boundary3D);
      const update: number[][] = getUpdate(scores, squares);
      
      // Always update state for rendering
      state = updateState(state, update);

      const endTime: number = performance.now();
      const fps: string = (1000 / (endTime - startTime)).toFixed(1);
      
      let hasMove = false;
      let moveText = "";

      if (usePositionTracking && positionTracker) {
        // Use our position-based tracking approach
        const detectedMove = positionTracker.processFrame(state);
        
        if (detectedMove) {
          hasMove = true;
          boardRef.current = positionTracker.getBoard();
          const history = positionTracker.getMoveHistory();
          moveText = formatMoveText(history);
          moveTextRef.current = moveText;
          framesSinceLastMove = 0;
          lastMoveRef.current = detectedMove;
          
          const payload = makeUpdatePayload(boardRef.current, false);
          console.log("✅ Position Tracker Move:", detectedMove, "Total moves:", history.length);
          dispatch(gameUpdate(payload));
        } else {
          framesSinceLastMove++;
        }
        
        setText([`FPS: ${fps}`, moveText || moveTextRef.current || "Ready - make a move", `🎯 Position Tracking`]);
      } else {
        // Use original score-based approach
        const {bestScore1, bestScore2, bestJointScore, bestMove, bestMoves} = processState(state, movesPairsRef.current, possibleMoves);

        if ((bestMoves !== null) && (mode !== "play")) {
          const move: string = bestMoves.sans[0];
          hasMove = (bestScore2 > 0) && (bestJointScore > 0) && (possibleMoves.has(move));
          if (hasMove) {
            boardRef.current.move(move);
            possibleMoves.clear();
            greedyMoveToTime = {};
          }
        }

        let hasGreedyMove: boolean = false;
        if (bestMove !== null && !(hasMove) && (bestScore1 > 0)) {
          const move: string = bestMove.sans[0];
          if (!(move in greedyMoveToTime)) { 
            greedyMoveToTime[move] = endTime;
          }

          const secondElapsed = (endTime - greedyMoveToTime[move]) > 1000;
          const newMove = sanToLan(boardRef.current, move) !== lastMoveRef.current;
          hasGreedyMove = secondElapsed && newMove;
          if (hasGreedyMove) {
            boardRef.current.move(move);
            greedyMoveToTime = {greedyMove: greedyMoveToTime[move]};
          }
        }
        
        if (hasMove || hasGreedyMove) {
          const greedy = (mode === "play") ? false : hasGreedyMove;
          const payload = makeUpdatePayload(boardRef.current, greedy);
          console.log("📊 Score-based move:", payload);
          dispatch(gameUpdate(payload));
        }
        setText([`FPS: ${fps}`, moveTextRef.current, "📊 Score-Based"]);
      }
      
      renderState(canvasRef.current, centers, boundary, state);

      tf.dispose([boxes, scores]);

      const endTensors: number = tf.memory().numTensors;
      if (startTensors < endTensors) {
        console.error(`Memory Leak! (${endTensors} > ${startTensors})`)
      }
    }
    requestId = requestAnimationFrame(loop);
  }
  
  requestId = requestAnimationFrame(loop);

  return () => {
    tf.disposeVariables();
    if (requestId) {
      window.cancelAnimationFrame(requestId);
    }
  };
};

const updateState = (state: number[][], update: number[][], decay: number=0.5) => {
  for (let i = 0; i < 64; i++) {
    for (let j = 0; j < 12; j++) {
      state[i][j] = decay * state[i][j] + (1 - decay) * update[i][j]
    }
  }
  return state
}

const formatMoveText = (history: string[]): string => {
  if (history.length === 0) return "";
  
  if (history.length === 1) {
    return `1. ${history[0]}`;
  }
  
  const moves: string[] = [];
  for (let i = 0; i < history.length; i += 2) {
    const moveNum = Math.floor(i / 2) + 1;
    if (i + 1 < history.length) {
      moves.push(`${moveNum}.${history[i]} ${history[i+1]}`);
    } else {
      moves.push(`${moveNum}.${history[i]}`);
    }
  }
  
  return moves.slice(-3).join(" ");
}

export default findPiecesHybrid;
