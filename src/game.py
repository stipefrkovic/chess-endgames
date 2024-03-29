import random
import chess
from pathlib import Path
import csv
import time
import os
from matplotlib import pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.ticker import FormatStrFormatter
from itertools import chain
import numpy as np
import pandas as pd

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
    
    def eight_neighbors(self, num_files, num_ranks):
        neighbor_squares = self.four_neighbors(num_files, num_ranks)
        # Check bottom-left neighbor
        if self.rank > 0 and self.file > 0:
            neighbor_squares.append(ChessSquare(self.file-1, self.rank-1))
        # Check bottom-right neighbor
        if self.rank > 0 and self.file < num_files - 1:
            neighbor_squares.append(ChessSquare(self.file+1, self.rank-1))
        # Check top-left neighbor
        if self.rank < num_ranks - 1 and self.file > 0:
            neighbor_squares.append(ChessSquare(self.file-1, self.rank+1))
        # Check top-right neighbor
        if self.rank < num_ranks - 1 and self.file < num_files - 1:
            neighbor_squares.append(ChessSquare(self.file+1, self.rank+1))
        return neighbor_squares
    
    def four_neighbors(self, num_files, num_ranks):
        neighbor_squares = []
        # Check bottom neighbor
        if self.rank > 0:
            neighbor_squares.append(ChessSquare(self.file, self.rank-1))
        # Check left neighbor
        if self.file > 0:
            neighbor_squares.append(ChessSquare(self.file-1, self.rank))
        # Check right neighbor
        if self.file < num_files - 1:
            neighbor_squares.append(ChessSquare(self.file+1, self.rank))
        # Check top neighbor
        if self.rank < num_ranks - 1:
            neighbor_squares.append(ChessSquare(self.file, self.rank+1))
        return neighbor_squares


class ChessBoard():
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

    def flood_fill(self, square, region, num_files, num_ranks):
        if square is None or square not in self.squares:
            return
        else:
            region.append(square)
            square_idx = self.squares.index(square)
            self.squares[square_idx] = None
            neighbors = square.four_neighbors(num_files, num_ranks)
            for neighbor in neighbors:
                self.flood_fill(neighbor, region, num_files, num_ranks)

    def find_smallest_region(self, num_files, num_ranks):
        regions = []
        for square in self.squares:
            if square is not None:
                region = []
                self.flood_fill(square, region, num_files, num_ranks)
                if region:
                    regions.append(region)
                    # for i in range(8):
                    #     print(self.squares[i*8:i*8+8])
        return min(regions, key=len)


class Game:
    def __init__(self, max_depth, train, train_steps, lambda_value, name, figures_path='/src/figures/', results_path='/src/results/'):
        self.max_depth = max_depth
        self.train = train
        self.train_steps = train_steps
        self.lambda_value = lambda_value
        self.name = name
        self.figures_path = str(Path().absolute()) + figures_path
        self.results_path = str(Path().absolute()) + results_path

    def play(self, model, start_state, real_reward):
        # Search
        logger.info("Searching")
        logger.info(f"start_state: {start_state.string()}\n{start_state}")
        pv = alpha_beta(
            state=start_state,
            model=model,
            depth=0,
            max_depth=self.max_depth,
            alpha=-100,
            beta=100
        )
        logger.info(f"end_state:\n{pv.get_states()[-1]}")
        logger.info(f"real_reward: {real_reward}")
        logger.info(f"pv_reward: {pv.reward}")

        reward_loss = pv.reward - real_reward
        self.write_results("reward_loss", [reward_loss])

        # Learning
        logger.info("Training")
        if self.train:
            results = model.train(pv,
                                      steps=self.train_steps,
                                      lambda_value=self.lambda_value)
            old_predictions = results.get("old_predictions")
            old_prediction_loss = [pv.reward - pred for pred in old_predictions]
            self.write_results("old_prediction_loss", old_prediction_loss)
            
            new_predictions = results.get("new_predictions")
            new_prediction_loss = [pv.reward - pred for pred in new_predictions]
            self.write_results("new_prediction_loss", new_prediction_loss)

            old_lambda_tds = results.get("old_lambda_tds")
            self.write_results("old_lambda_td_loss", old_lambda_tds)

            new_lambda_tds = results.get("new_lambda_tds")
            self.write_results("new_lambda_td_loss", new_lambda_tds)

# TODO results and plot class

    def write_results(self, results_name, results):
        results_file = open(f"{self.results_path}{self.name}_{results_name}.csv", 'a')
        writer = csv.writer(results_file)
        writer.writerow(results)
        results_file.close()

    def plot_reward_losses(self):
        df = pd.read_table(f"{self.results_path}{self.name}_reward_loss.csv", sep=",", header=None)
        reward_loss_2d = np.array(df)
        reward_losses_1d = list(chain(*reward_loss_2d))

        logger.info(f"reward_loss_mean: {np.mean(reward_losses_1d)}")

        logger.info("Plotting reward losses")
        fig, ax = plt.subplots()
        plt.xlabel('game')
        plt.ylabel('reward_loss')
        plt.axhline(0, color='black', linestyle='dashed', linewidth=1)
        # plt.xticks(range(1, len(reward_losses_1d)+1, 1))
        # plt.ylim(min(reward_losses_1d) - 0.1, max(reward_losses_1d) + 0.1)
        ax.plot(range(1, len(reward_losses_1d)+1, 1), reward_losses_1d)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        fig.savefig(f"{self.figures_path}{self.name}_reward_loss.png")

    def plot_old_new_state_losses(self, loss_name):
        df1 = pd.read_table(f"{self.results_path}{self.name}_old_{loss_name}_loss.csv", sep=",", names=list(range(self.max_depth+1)))
        df2 = pd.read_table(f"{self.results_path}{self.name}_new_{loss_name}_loss.csv", sep=",", names=list(range(self.max_depth+1)))
        
        old_loss = np.array(df1)
        old_loss_mean = np.nanmean(old_loss, axis=0)
        old_loss_std = np.nanstd(old_loss, axis=0)
        new_loss = np.array(df2)
        new_loss_mean = np.nanmean(new_loss, axis=0)
        new_loss_std = np.nanstd(new_loss, axis=0)
        
        logger.info(f"old_{loss_name}_loss_mean: {old_loss_mean}")
        logger.info(f"old_{loss_name}_loss_std: {old_loss_std}")
        logger.info(f"new_{loss_name}_loss_mean: {new_loss_mean}")
        logger.info(f"new_{loss_name}_loss_std: {new_loss_std}")

        logger.info(f"Plotting {loss_name} losses")
        fig, ax = plt.subplots()
        trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
        trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
        plt.xlabel('state')
        plt.ylabel(f'{loss_name}_loss')
        plt.xticks(range(len(old_loss_mean)))
        plt.axhline(0, color='black', linestyle='dashed', linewidth=1)
        ax.errorbar(range(len(old_loss_mean)),
                    old_loss_mean,
                    old_loss_std,
                    transform=trans1,
                    marker='o',
                    linestyle='none',
                    label='Before training')
        ax.errorbar(range(len(new_loss_mean)),
                    new_loss_mean,
                    new_loss_std,
                    transform=trans2,
                    marker='o',
                    linestyle='none',
                    label='After training')
        ax.legend()
        fig.savefig(f"{self.figures_path}{self.name}_{loss_name}_loss.png")


class ChessGame(Game):
    def create_random_chess_board(self, pieces):
        board = chess.Board().empty()
        board.turn = random.choice(chess.COLORS)

        king_square = random.choice(chess.SQUARES)
        board.set_piece_at(king_square, pieces[0])

        while True:
            piece_square = random.choice(chess.SQUARES)
            board.set_piece_at(piece_square, pieces[1])
            other_king_square = random.choice(chess.SQUARES)
            board.set_piece_at(other_king_square, pieces[2])
            if board.is_valid() and board.outcome() == None and all(not board.is_capture(legal_move) for legal_move in board.legal_moves):
                real_reward = 1 if pieces[1].color else -1
                # for legal_move in board.legal_moves:
                #     if board.is_capture(legal_move):
                #         real_reward = 0
                return board, real_reward
            else:
                board.remove_piece_at(piece_square)
                board.remove_piece_at(other_king_square)
                board.set_piece_at(king_square, pieces[0])
            
    def create_region_chess_board(self, pieces, turn):
        board = chess.Board().empty()
        board.turn = turn

        # King
        king_squares = ChessBoard([ChessSquare(file, rank) for rank in range(7, -1, -1) for file in range(0, 8, 1)])
        king_square = king_squares.random_square()
        board.set_piece_at(king_square.create_square(), pieces[0])

        # King's piece
        neighbor_squares = ChessBoard(king_square.eight_neighbors(8, 8))
        piece_square = neighbor_squares.random_square()
        board.set_piece_at(piece_square.create_square(), pieces[1])

        # Other King
        chess_squares = ChessBoard([ChessSquare(file, rank) for rank in range(7, -1, -1) for file in range(0, 8, 1)])
        chess_squares.remove_illegal_squares(board, pieces[2])
        other_king_squares = ChessBoard(chess_squares.find_smallest_region(8, 8))
        other_king_square = other_king_squares.random_square()
        board.set_piece_at(other_king_square.create_square(), pieces[2])
        
        assert(board.is_valid())

        return board
    
    def create_pieces(self, piece):
        winning_color = random.choice([chess.WHITE, chess.BLACK])
        first_king = chess.Piece(chess.KING, winning_color)
        first_king_piece = chess.Piece(piece, winning_color)
        other_king = chess.Piece(chess.KING, not winning_color)
        pieces = (first_king, first_king_piece, other_king)
        return pieces

class RookEndgameGame(ChessGame):
    def __init__(self, max_depth, train, train_steps, lambda_value, name):
        super().__init__(max_depth, train, train_steps, lambda_value, name)

    # TODO refractor into setup and play
    def play(self, model):
        pieces = self.create_pieces(chess.ROOK)
        # board = self.create_region_chess_board(pieces, winning_color)
        board, real_reward = self.create_random_chess_board(pieces)
        return super().play(model, ChessState(board), real_reward)


class QueenEndgameGame(ChessGame):
    def __init__(self, max_depth, train, train_steps, lambda_value, name):
        super().__init__(max_depth, train, train_steps, lambda_value, name)

    # TODO refractor into setup and play
    def play(self, model):
        pieces = self.create_pieces(chess.QUEEN)
        # board = self.create_region_chess_board(pieces, winning_color)
        board, real_reward = self.create_random_chess_board(pieces)
        return super().play(model, ChessState(board), real_reward)


class GamePlayer:
    def __init__(self, model, figures_path='/src/figures/', results_path='/src/results/'):
        self.model = model
        self.figures_path = str(Path().absolute()) + figures_path
        self.results_path = str(Path().absolute()) + results_path

    def set_up(self, load_weights=True, clear_dirs=True):
        self.model.build()
        if load_weights:
            self.model.load_weights()
        if clear_dirs:
            self.clear_dirs()

    def clear_dirs(self):
        self.clear_dir(self.results_path, ".csv")
        self.clear_dir(self.figures_path, ".png")

    def clear_dir(self, path, ending):
        for f in os.listdir(path):
            if not f.endswith(ending):
                continue
            os.remove(os.path.join(path, f))

    def wrap_up(self, save_weights=True):
        if save_weights:
            self.model.save_weights()

    def play_game(self, game, iterations, save_every_n_iter):
        logger.info(f"game.name: {game.name}")
        logger.info(f"game.max_depth: {game.max_depth}")
        logger.info(f"game.train_steps: {game.train_steps}")
        logger.info(f"game.lambda_value: {game.lambda_value}")
        for i in range(1, iterations+1, 1):
            logger.info(f"{game.name} {i} of {iterations}")
            start_time = time.time()
            game.play(self.model)
            end_time = time.time() - start_time
            logger.info(f"time: {end_time:.4f} sec")
            if i % save_every_n_iter == 0:
                self.model.save_weights()
        game.plot_reward_losses()
        game.plot_old_new_state_losses('prediction')
        game.plot_old_new_state_losses('lambda_td')
