import random
import chess
import time

from learning import ChessMLP
from search import alphabeta

start_position = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1'
white_mate_in_1_1 = "8/8/8/k1K5/8/1R6/8/8 w - - 0 1"
white_mate_in_1_2 = "3r2k1/4Qp2/5P1p/2p3p1/3q4/8/P1B1R1PP/4K3 b - - 6 28"
white_mate_in_2_1 = "k7/8/2K5/8/8/1R6/8/8 w - - 0 1"
white_mate_in_2_2 = "4rk1r/3R1ppp/p2p4/1p1p2B1/3q1P2/3B4/PPP3PP/2K1R3 w - - 2 18"
white_mate_in_3_rook = "8/k7/8/2K5/8/1R6/8/8 w - - 0 1"
white_mate_in_3_queen = "8/k7/8/2K5/8/1Q6/8/8 w - - 0 1"
black_mate_in_2 = "K7/8/2k5/8/8/1r6/8/8 b - - 0 1"


class ChessExperiment:
    def __init__(self, max_depth, train_steps, lambda_value):
        self.max_depth = max_depth
        self.train_steps = train_steps
        self.lambda_value = lambda_value

    def run(self, positions, model):
        for position in positions:
            # Search
            print(f"Start Position: {position}")
            board = chess.Board(position)
            # print("Board:\n" + str(board))

            # for i in range(6):
            #     print(i*64, (i+1)*64)
            #     print(bitboard[i*64:(i+1)*64])
            start_time = time.time()
            principal_variation = alphabeta(
                board=board,
                model=model,
                depth=0,
                max_depth=self.max_depth,
                alpha=-100,
                beta=100
            )
            end_time = time.time() - start_time
            print("Alphabeta - Time: %.4s" % end_time)
            print("Principal Variation - Evaluation: %s" % principal_variation.reward)
            # print("Principal Variation - Moves: %s" % principal_variation.moves)
            # for move in principal_variation.moves:
            #     print(chess.Board(move))

            # Learning
            model.train(principal_variation,
                        steps=self.train_steps,
                        lambda_value=self.lambda_value)

    @staticmethod
    def create_random_chess_position(non_king_pieces, turn):
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
        return board.fen()


class RookEndgamesExperiment(ChessExperiment):
    def __init__(self, max_depth, train_steps, lambda_value, iterations):
        super().__init__(max_depth, train_steps, lambda_value)
        self.iterations = iterations

    def run(self, model):
        positions = []
        for i in range(self.iterations):
            positions.append(self.create_random_king_rook_endgame())
        super().run(positions, model)

    def create_random_king_rook_endgame(self):
        rooks = [chess.Piece(chess.ROOK, chess.WHITE),
                 chess.Piece(chess.ROOK, chess.BLACK)]
        random_rook = random.choice(rooks)
        return super().create_random_chess_position([random_rook], random_rook.color)


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
