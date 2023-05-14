import random
import chess
import time

from learning import ChessMLP
from search import AlphaBeta, ChessState

start_state = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1'
white_mate_in_1_1 = "8/8/8/k1K5/8/1R6/8/8 w - - 0 1"
white_mate_in_1_2 = "3r2k1/4Qp2/5P1p/2p3p1/3q4/8/P1B1R1PP/4K3 b - - 6 28"
white_mate_in_2_1 = "k7/8/2K5/8/8/1R6/8/8 w - - 0 1"
white_mate_in_2_2 = "4rk1r/3R1ppp/p2p4/1p1p2B1/3q1P2/3B4/PPP3PP/2K1R3 w - - 2 18"
white_mate_in_3_rook = "8/k7/8/2K5/8/1R6/8/8 w - - 0 1"
white_mate_in_3_queen = "8/k7/8/2K5/8/1Q6/8/8 w - - 0 1"
black_mate_in_2 = "K7/8/2k5/8/8/1r6/8/8 b - - 0 1"

class Experiment:
    def __init__(self, max_depth, train_steps, lambda_value):
        self.max_depth = max_depth
        self.train_steps = train_steps
        self.lambda_value = lambda_value

    def print_state(self, state):
        pass

    def run(self, model, start_states):
        for start_state in start_states:
            # Search
            print(f"Start state:\n{start_state.get_string()}")
            start_time = time.time()
            alpha_beta = AlphaBeta()
            principal_variation = alpha_beta.run(
                state=start_state,
                model=model,
                depth=0,
                max_depth=self.max_depth,
                alpha=-100,
                beta=100
            )
            end_time = time.time() - start_time
            print("Alphabeta - Time: %.4s" % end_time)
            print("Principal Variation - Evaluation: %s" % principal_variation.reward)
            for state_idx, state in enumerate(principal_variation.states):
                print(f"State {state_idx}:\n{state.get_string()}")

            # Learning
            model.train(principal_variation,
                        steps=self.train_steps,
                        lambda_value=self.lambda_value)

class ChessExperiment(Experiment):
    def __init__(self, max_depth, train_steps, lambda_value):
        super().__init__(max_depth, train_steps, lambda_value)

    def create_random_chess_board(self, non_king_pieces, turn):
        kings = [chess.Piece(chess.KING, chess.WHITE),
                 chess.Piece(chess.KING, chess.BLACK)]
        pieces = kings + non_king_pieces
        board = None
        while True:
            board = chess.Board().empty()
            populated_squares = []
            for piece in pieces:
                while True:
                    square = random.choice(chess.SQUARES)
                    if square not in populated_squares:
                        board.set_piece_at(square, piece)
                        populated_squares.append(square)
                        break
            board.turn = turn
            if board.is_valid():
                break
        return board


class RookEndgamesExperiment(ChessExperiment):
    def __init__(self, max_depth, train_steps, lambda_value, iterations):
        super().__init__(max_depth, train_steps, lambda_value)
        self.iterations = iterations

    def run(self, model):
        start_states = [ChessState(chess.Board(white_mate_in_2_2)), ChessState(chess.Board(black_mate_in_2))]
        # start_states = []
        for i in range(self.iterations):
            start_states.append(ChessState(self.create_random_king_rook_endgame()))
        super().run(model, start_states)

    def create_random_king_rook_endgame(self):
        rooks = [chess.Piece(chess.ROOK, chess.WHITE),
                 chess.Piece(chess.ROOK, chess.BLACK)]
        random_rook = random.choice(rooks)
        return super().create_random_chess_board([random_rook], random_rook.color)


class ExperimentRunner:
    def __init__(self):
        self.model = None

    def set_up(self):
        self.model = ChessMLP()
        self.model.build()
        self.model.load_weights()

    def wrap_up(self):
        self.model.save_weights()

    def run_experiment(self, experiment):
        experiment.run(self.model)
