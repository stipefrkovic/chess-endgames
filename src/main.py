import time

import chess

from src.learning import compute_td_lambda_update, compute_tds
from src.models import Model
from src.search import alphabeta
from src.util import print_bitboard


def main():
    mate_in_1 = "8/8/8/k1K5/8/1R6/8/8 w - - 0 1"
    mate_in_2 = "k7/8/2K5/8/8/1R6/8/8 w - - 0 1"
    mate_in_3_rook = "8/k7/8/2K5/8/1R6/8/8 w - - 0 1"
    mate_in_3_queen = "8/k7/8/2K5/8/1Q6/8/8 w - - 0 1"
    start_position = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    board = chess.Board(mate_in_1)
    model = Model()

    # Search
    start_time = time.time()
    principal_variation = alphabeta(board, model, 0, 3, -100, 100)
    end_time = time.time() - start_time
    print("Alphabeta - Time: %.4s, Evaluation: %s" % (end_time, principal_variation.evaluation))
    # for move in principal_variation.moves:
    #     print(chess.Board(move))

    # Learning
    tds = compute_tds(model, principal_variation)
    print("Temporal differences: " + str(tds))
    fit_data = compute_td_lambda_update(principal_variation, tds, 0.95)
    for i in range(len(fit_data[0])):
        print_bitboard(fit_data[0][i])
        print(fit_data[1][i])


if __name__ == "__main__":
    main()
