import { Chess, Square } from "chess.js";
import { SQUARE_NAMES } from "./constants";

// Configuration constants
const DETECTION_THRESHOLD = 0.15;
const VACATED_THRESHOLD = 0.25;
const ARRIVED_THRESHOLD = 0.55;
const CONFIRM_FRAMES = 3;
const COOLDOWN_FRAMES = 20;
const WINDOW_SIZE = 20;

interface SquareHistory {
  square: Square;
  history: number[];
  lastChangeFrame: number;
  confirmedEmpty: boolean;
  confirmedOccupied: boolean;
}

interface CandidateMove {
  from: Square;
  to: Square;
  score: number;
  frames: number;
}

class SquareTracker {
  private squares: Map<Square, SquareHistory>;
  private frameCount: number;

  constructor() {
    this.squares = new Map();
    this.frameCount = 0;
    
    // Initialize all 64 squares
    SQUARE_NAMES.forEach(square => {
      this.squares.set(square, {
        square,
        history: [],
        lastChangeFrame: 0,
        confirmedEmpty: false,
        confirmedOccupied: false
      });
    });
  }

  updateSquare(square: Square, detectionScore: number): void {
    const tracker = this.squares.get(square)!;
    
    // Add to sliding window
    tracker.history.push(detectionScore);
    if (tracker.history.length > WINDOW_SIZE) {
      tracker.history.shift();
    }
  }

  getAveragePresence(square: Square): number {
    const tracker = this.squares.get(square)!;
    if (tracker.history.length === 0) return 0;
    
    const sum = tracker.history.reduce((a, b) => a + b, 0);
    return sum / tracker.history.length;
  }

  isVacated(square: Square): boolean {
    const tracker = this.squares.get(square)!;
    if (tracker.history.length < CONFIRM_FRAMES) return false;
    
    const recentFrames = tracker.history.slice(-CONFIRM_FRAMES);
    return recentFrames.every(score => score < VACATED_THRESHOLD);
  }

  isArrived(square: Square): boolean {
    const tracker = this.squares.get(square)!;
    if (tracker.history.length < CONFIRM_FRAMES) return false;
    
    const recentFrames = tracker.history.slice(-CONFIRM_FRAMES);
    return recentFrames.every(score => score > ARRIVED_THRESHOLD);
  }

  isInCooldown(square: Square): boolean {
    const tracker = this.squares.get(square)!;
    return (this.frameCount - tracker.lastChangeFrame) < COOLDOWN_FRAMES;
  }

  markChanged(square: Square): void {
    const tracker = this.squares.get(square)!;
    tracker.lastChangeFrame = this.frameCount;
  }

  incrementFrame(): void {
    this.frameCount++;
  }

  reset(): void {
    this.frameCount = 0;
    this.squares.forEach(tracker => {
      tracker.history = [];
      tracker.lastChangeFrame = 0;
      tracker.confirmedEmpty = false;
      tracker.confirmedOccupied = false;
    });
  }
}

export class PositionBasedTracker {
  private board: Chess;
  private tracker: SquareTracker;
  private expectedPosition: Map<Square, boolean>; // true = occupied, false = empty
  private pendingMoves: CandidateMove[];
  private lastMove: string | null;
  private moveHistory: string[];

  constructor() {
    this.board = new Chess();
    this.tracker = new SquareTracker();
    this.expectedPosition = new Map();
    this.pendingMoves = [];
    this.lastMove = null;
    this.moveHistory = [];
    
    this.initializePosition();
  }

  private initializePosition(): void {
    // Starting position: 32 pieces on their starting squares
    const startingPieces = [
      // White pieces
      'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
      'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
      // Black pieces
      'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
      'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8'
    ];

    SQUARE_NAMES.forEach(square => {
      this.expectedPosition.set(square, startingPieces.includes(square));
    });
  }

  processFrame(detections: number[][]): string | null {
    this.tracker.incrementFrame();
    
    // Update all squares with their detection scores
    SQUARE_NAMES.forEach((square, idx) => {
      const maxScore = Math.max(...detections[idx]);
      this.tracker.updateSquare(square, maxScore);
    });

    // Find vacated and arrived squares
    const vacatedSquares: Square[] = [];
    const arrivedSquares: Square[] = [];

    SQUARE_NAMES.forEach(square => {
      const expectedOccupied = this.expectedPosition.get(square)!;
      
      if (expectedOccupied && this.tracker.isVacated(square) && !this.tracker.isInCooldown(square)) {
        vacatedSquares.push(square);
      }
      
      if (!expectedOccupied && this.tracker.isArrived(square) && !this.tracker.isInCooldown(square)) {
        arrivedSquares.push(square);
      }
    });

    // Debug logging
    if (vacatedSquares.length > 0 || arrivedSquares.length > 0) {
      console.log("🔍 Changes detected - Vacated:", vacatedSquares, "Arrived:", arrivedSquares);
    }

    // If we have both vacated and arrived, try to find matching move
    if (vacatedSquares.length > 0 && arrivedSquares.length > 0) {
      const move = this.findBestMove(vacatedSquares, arrivedSquares);
      if (move) {
        return this.executeMove(move);
      }
    }

    return null;
  }

  private findBestMove(vacatedSquares: Square[], arrivedSquares: Square[]): CandidateMove | null {
    const legalMoves = this.board.moves({ verbose: true });
    let bestMove: CandidateMove | null = null;
    let bestScore = -Infinity;

    for (const from of vacatedSquares) {
      for (const to of arrivedSquares) {
        // Check if this is a legal move
        const legalMove = legalMoves.find(m => m.from === from && m.to === to);
        if (!legalMove) continue;

        // Score the move
        let score = 0;
        
        // +3 for from square being vacated
        const fromAvg = this.tracker.getAveragePresence(from);
        score += (1 - fromAvg) * 3;
        
        // +3 for to square being arrived
        const toAvg = this.tracker.getAveragePresence(to);
        score += toAvg * 3;
        
        // +2 for to square having strong recent detection
        if (this.tracker.isArrived(to)) {
          score += 2;
        }

        if (score > bestScore) {
          bestScore = score;
          bestMove = {
            from,
            to,
            score,
            frames: CONFIRM_FRAMES
          };
        }
      }
    }

    // Handle castling
    if (!bestMove) {
      bestMove = this.checkCastling(vacatedSquares, arrivedSquares);
    }

    return bestMove && bestMove.score > 3 ? bestMove : null;
  }

  private checkCastling(vacatedSquares: Square[], arrivedSquares: Square[]): CandidateMove | null {
    const legalMoves = this.board.moves({ verbose: true });
    
    // White kingside castling
    if (vacatedSquares.includes('e1' as Square) && arrivedSquares.includes('g1' as Square)) {
      const castleMove = legalMoves.find(m => m.from === 'e1' && m.to === 'g1' && m.flags.includes('k'));
      if (castleMove) {
        return { from: 'e1' as Square, to: 'g1' as Square, score: 8, frames: CONFIRM_FRAMES };
      }
    }
    
    // White queenside castling
    if (vacatedSquares.includes('e1' as Square) && arrivedSquares.includes('c1' as Square)) {
      const castleMove = legalMoves.find(m => m.from === 'e1' && m.to === 'c1' && m.flags.includes('q'));
      if (castleMove) {
        return { from: 'e1' as Square, to: 'c1' as Square, score: 8, frames: CONFIRM_FRAMES };
      }
    }
    
    // Black kingside castling
    if (vacatedSquares.includes('e8' as Square) && arrivedSquares.includes('g8' as Square)) {
      const castleMove = legalMoves.find(m => m.from === 'e8' && m.to === 'g8' && m.flags.includes('k'));
      if (castleMove) {
        return { from: 'e8' as Square, to: 'g8' as Square, score: 8, frames: CONFIRM_FRAMES };
      }
    }
    
    // Black queenside castling
    if (vacatedSquares.includes('e8' as Square) && arrivedSquares.includes('c8' as Square)) {
      const castleMove = legalMoves.find(m => m.from === 'e8' && m.to === 'c8' && m.flags.includes('q'));
      if (castleMove) {
        return { from: 'e8' as Square, to: 'c8' as Square, score: 8, frames: CONFIRM_FRAMES };
      }
    }

    return null;
  }

  private executeMove(move: CandidateMove): string | null {
    try {
      const result = this.board.move({ from: move.from, to: move.to });
      if (!result) return null;

      // Update expected position
      this.expectedPosition.set(move.from, false);
      this.expectedPosition.set(move.to, true);

      // Mark squares as changed
      this.tracker.markChanged(move.from);
      this.tracker.markChanged(move.to);

      // Handle castling - update rook position
      if (result.flags.includes('k') || result.flags.includes('q')) {
        if (move.to === 'g1') {
          this.expectedPosition.set('h1' as Square, false);
          this.expectedPosition.set('f1' as Square, true);
          this.tracker.markChanged('h1' as Square);
          this.tracker.markChanged('f1' as Square);
        } else if (move.to === 'c1') {
          this.expectedPosition.set('a1' as Square, false);
          this.expectedPosition.set('d1' as Square, true);
          this.tracker.markChanged('a1' as Square);
          this.tracker.markChanged('d1' as Square);
        } else if (move.to === 'g8') {
          this.expectedPosition.set('h8' as Square, false);
          this.expectedPosition.set('f8' as Square, true);
          this.tracker.markChanged('h8' as Square);
          this.tracker.markChanged('f8' as Square);
        } else if (move.to === 'c8') {
          this.expectedPosition.set('a8' as Square, false);
          this.expectedPosition.set('d8' as Square, true);
          this.tracker.markChanged('a8' as Square);
          this.tracker.markChanged('d8' as Square);
        }
      }

      // Handle en passant - remove captured pawn
      if (result.flags.includes('e')) {
        const captureSquare = (move.to.charAt(0) + move.from.charAt(1)) as Square;
        this.expectedPosition.set(captureSquare, false);
        this.tracker.markChanged(captureSquare);
      }

      // Handle capture
      if (result.captured) {
        // Already handled by setting move.to to true
      }

      const san = result.san;
      this.lastMove = san;
      this.moveHistory.push(san);

      return san;
    } catch (error) {
      console.error('Move execution failed:', error);
      return null;
    }
  }

  reset(): void {
    this.board.reset();
    this.tracker.reset();
    this.initializePosition();
    this.pendingMoves = [];
    this.lastMove = null;
    this.moveHistory = [];
  }

  getBoard(): Chess {
    return this.board;
  }

  getFen(): string {
    return this.board.fen();
  }

  getMoveHistory(): string[] {
    return this.moveHistory;
  }
}

export default PositionBasedTracker;
