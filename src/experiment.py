import random
import chess
from pathlib import Path
import csv
import time
import os

from search import alpha_beta, ChessState
from logger import logger

start_state = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1'
white_mate_in_1_1 = "8/8/8/k1K5/8/1R6/8/8 w - - 0 1"
white_mate_in_1_2 = "3r2k1/4Qp2/5P1p/2p3p1/3q4/8/P1B1R1PP/4K3 b - - 6 28"
white_mate_in_2_1 = "k7/8/2K5/8/8/1R6/8/8 w - - 0 1"
white_mate_in_2_2 = "4rk1r/3R1ppp/p2p4/1p1p2B1/3q1P2/3B4/PPP3PP/2K1R3 w - - 2 18"
white_mate_in_3_rook = "8/k7/8/2K5/8/1R6/8/8 w - - 0 1"
white_mate_in_3_queen = "8/k7/8/2K5/8/1Q6/8/8 w - - 0 1"
white_mate_in_4_rook = "8/8/k7/2K5/2R5/8/8/8 w - - 0 1"
black_mate_in_2 = "K7/8/2k5/8/8/1r6/8/8 b - - 0 1"


class ChessSquare():
    def __init__(self, file, rank):
        self.file = file
        self.rank = rank

    def __repr__(self):
        return f"({self.file}, {self.rank})"
    
    def __eq__(self, other): 
        if not isinstance(other, ChessSquare):
            return NotImplemented
        return self.file == other.file and self.rank == other.rank

    
    def create_square(self):
        return chess.square(self.file, self.rank)
    
    def legal_neighbors(self):
        neighbor_squares = []
        # Check bottom-left neighbor
        if self.rank > 0 and self.file > 0:
            neighbor_squares.append(ChessSquare(self.file-1, self.rank-1))
        # Check bottom neighbor
        if self.rank > 0:
            neighbor_squares.append(ChessSquare(self.file, self.rank-1))
        # Check bottom-right neighbor
        if self.rank > 0 and self.file < ChessBoard.num_files - 1:
            neighbor_squares.append(ChessSquare(self.file+1, self.rank-1))
        # Check left neighbor
        if self.file > 0:
            neighbor_squares.append(ChessSquare(self.file-1, self.rank))
        # Check right neighbor
        if self.file < ChessBoard.num_files - 1:
            neighbor_squares.append(ChessSquare(self.file+1, self.rank))
        # Check top-left neighbor
        if self.rank < ChessBoard.num_ranks - 1 and self.file > 0:
            neighbor_squares.append(ChessSquare(self.file-1, self.rank+1))
        # Check top neighbor
        if self.rank < ChessBoard.num_ranks - 1:
            neighbor_squares.append(ChessSquare(self.file, self.rank+1))
        # Check top-right neighbor
        if self.rank < ChessBoard.num_ranks - 1 and self.file < ChessBoard.num_files - 1:
            neighbor_squares.append(ChessSquare(self.file+1, self.rank+1))
        return neighbor_squares


class ChessBoard():
    num_files = 8
    num_ranks = 8

    def __init__(self, squares):
        self.squares = squares

    def random_square(self):
        square = random.choice(self.squares)
        return square
    
    def remove_illegal_squares(self, board, piece):
        for square_idx, square in enumerate(self.squares):
            if board.piece_at(square.create_square()) is not None:
                self.squares[square_idx] = None
            else:
                board.set_piece_at(square.create_square(), piece)
                if not board.is_valid():
                    self.squares[square_idx] = None
                board.remove_piece_at(square.create_square())
        # for i in range(8):
        #     print(self.squares[i*8 : i*8+8])
        # print('\n')


    def flood_fill(self, square, region):
        if square is None or square not in self.squares:
            return
        else:
            region.append(square)
            square_idx = self.squares.index(square)
            self.squares[square_idx] = None
            neighbors = square.legal_neighbors()
            for neighbor in neighbors:
                self.flood_fill(neighbor, region)

    def find_smallest_region(self):
        regions = []
        for square in self.squares:
            if square is not None:
                region = []
                self.flood_fill(square, region)
                if region:
                    regions.append(region)
                    # for i in range(8):
                    #     print(self.squares[i*8 : i*8+8])
                    # print('\n')
        # for region in regions:
        #     print(region)
        return min(regions, key=len)


class Game:
    def __init__(self, name, max_depth, train_steps, lambda_value, results_path='/src/results/'):
        self.name = name
        self.max_depth = max_depth
        self.train_steps = train_steps
        self.lambda_value = lambda_value
        self.results_path = str(Path().absolute()) + results_path

    def play(self, model, start_state, train):
        # Search
        logger.info("Searching")
        self.write_results("real_reward", [start_state.real_reward])

        pv = alpha_beta(
            state=start_state,
            model=model,
            depth=0,
            max_depth=self.max_depth,
            alpha=-100,
            beta=100
        )
        self.write_results("pv_reward", [pv.reward])

        reward_loss = abs(start_state.real_reward - pv.reward)
        self.write_results("reward_loss", [reward_loss])

        # Learning
        logger.info("Training")
        if train:
            evaluations = model.train(pv,
                                      steps=self.train_steps,
                                      lambda_value=self.lambda_value)
            old_evaluations = evaluations.get("old_evaluations")
            old_evaluation_losses = [abs(eval - pv.reward) for eval in old_evaluations]
            self.write_results("old_evaluation_losses", old_evaluation_losses)
            
            new_evaluations = evaluations.get("new_evaluations")
            new_evaluation_losses = [abs(eval - pv.reward) for eval in new_evaluations]
            self.write_results("new_evaluation_losses", new_evaluation_losses)

    def write_results(self, results_name, results):
        results_file = open(f"{self.results_path}{self.name}_{results_name}.csv", 'a')
        writer = csv.writer(results_file)
        writer.writerow(results)
        results_file.close()

class ChessGame(Game):
    # def create_random_chess_board(self, non_king_pieces, turn):
    #     kings = [chess.Piece(chess.KING, chess.WHITE),
    #              chess.Piece(chess.KING, chess.BLACK)]
    #     pieces = kings + non_king_pieces
    #     board = None
    #     while True:
    #         board = chess.Board().empty()
    #         populated_squares = []
    #         for piece in pieces:
    #             while True:
    #                 square = random.choice(chess.SQUARES)
    #                 if square not in populated_squares:
    #                     board.set_piece_at(square, piece)
    #                     populated_squares.append(square)
    #                     break
    #         board.turn = turn
    #         if board.is_valid():
    #             break
    #     return board
    
    def create_region_chess_board(self, pieces, turn):
        chess_squares = ChessBoard([ChessSquare(file, rank) for rank in range(ChessBoard.num_ranks-1, -1, -1) for file in range(0, ChessBoard.num_files, 1)])
        board = chess.Board().empty()
        board.turn = turn

        # King
        king_square = chess_squares.random_square()
        board.set_piece_at(king_square.create_square(), pieces[0])

        # King's piece
        neighbor_squares = ChessBoard(king_square.legal_neighbors())
        piece_square = neighbor_squares.random_square()
        board.set_piece_at(piece_square.create_square(), pieces[1])

        # Other King
        chess_squares.remove_illegal_squares(board, pieces[2])
        other_king_squares = ChessBoard(chess_squares.find_smallest_region())
        other_king_square = other_king_squares.random_square()
        board.set_piece_at(other_king_square.create_square(), pieces[2])
        
        assert(board.is_valid())

        return board
    
    def create_region_chess_state(self, piece):
        winning_color = random.choice([chess.WHITE, chess.BLACK])
        first_king = chess.Piece(chess.KING, winning_color)
        first_king_piece = chess.Piece(piece, winning_color)
        other_king = chess.Piece(chess.KING, not winning_color)
        pieces = (first_king, first_king_piece, other_king)
        board = self.create_region_chess_board(pieces, winning_color)
        board = chess.Board(black_mate_in_2)
        logger.info(f"\n{board}")
        return ChessState(board)


class RookEndgameGame(ChessGame):
    def __init__(self, game_name, max_depth, train_steps, lambda_value):
        super().__init__(game_name, max_depth, train_steps, lambda_value)

    def play(self, model, train=True):
        start_state = self.create_region_chess_state(chess.ROOK)
        return super().play(model, start_state, train)

    # def create_random_rook_endgame(self):
    #     rooks = [chess.Piece(chess.ROOK, chess.WHITE),
    #              chess.Piece(chess.ROOK, chess.BLACK)]
    #     random_rook = random.choice(rooks)
    #     return super().create_random_chess_board([random_rook], random_rook.color)


class QueenEndgameGame(ChessGame):
    def __init__(self, game_name, max_depth, train_steps, lambda_value):
        super().__init__(game_name, max_depth, train_steps, lambda_value)

    def play(self, model, train=True):
        start_state = self.create_region_chess_state(chess.QUEEN)
        return super().play(model, start_state, train)
    
    # def create_random_queen_endgame(self):
    #     queens = [chess.Piece(chess.QUEEN, chess.WHITE),
    #              chess.Piece(chess.QUEEN, chess.BLACK)]
    #     queen = random.choice(queens)
    #     return super().create_random_chess_board([queen], queen.color)


class GamePlayer:
    def __init__(self, model, figures_path='/src/figures/', results_path='/src/results/'):
        self.model = model
        self.figures_path = str(Path().absolute()) + figures_path
        self.results_path = str(Path().absolute()) + results_path

    def set_up(self, load_weights=True, clear_results=True):
        self.model.build()
        if load_weights:
            self.model.load_weights()
        if clear_results:
            self.clear_results()

    def clear_results(self):
        for f in os.listdir(self.results_path):
            if not f.endswith(".csv"):
                continue
            os.remove(os.path.join(self.results_path, f))


    def wrap_up(self, save_weights=True):
        if save_weights:
            self.model.save_weights()

    def play_games(self, game, iterations):
        for i in range(iterations):
            logger.info(f"{game.name} {i}")
            start_time = time.time()
            game.play(self.model)
            end_time = time.time() - start_time 
            logger.info(f"time: {end_time:.4f}")
        # self.plot_losses(game.name, abs_losses)

    # def plot_losses(self, name, losses, ylim=(-0.05, 1.05)):
    #     fig, ax = plt.subplots()
    #     ax.plot(range(len(losses)), losses)
    #     plt.xticks(range(len(losses)))
    #     plt.xlabel('Game')
    #     plt.ylabel('Loss')
    #     plt.ylim(ylim[0], ylim[1])
    #     # plt.show()
    #     fig.savefig(self.figures_path + name + '.png')
