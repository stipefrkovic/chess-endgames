import time

import chess

from src.learning import compute_lambda_tds, compute_tds, compute_evaluations
from src.models import MLP
from src.search import alphabeta
from src.util import fen_to_bitboard, fens_to_bitboards


def main():
    mate_in_1 = "8/8/8/k1K5/8/1R6/8/8 w - - 0 1"
    mate_in_2 = "k7/8/2K5/8/8/1R6/8/8 w - - 0 1"
    mate_in_3_rook = "8/k7/8/2K5/8/1R6/8/8 w - - 0 1"
    mate_in_3_queen = "8/k7/8/2K5/8/1Q6/8/8 w - - 0 1"
    start_position = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1'

    board = chess.Board(mate_in_2)
    print("Board:\n" + str(board))

    model = MLP()
    model.build()

    # Util
    # bitboard = fen_to_bitboard(board.fen())
    # print("Bitboard - Shape %s:" % str(bitboard.shape))
    # for i in range(6):
    #     print(i*64, (i+1)*64)
    #     print(bitboard[i*64:(i+1)*64])

    # Search
    start_time = time.time()
    principal_variation = alphabeta(board, model, 0, 3, -100, 100)
    end_time = time.time() - start_time
    print("Alphabeta - Time: %.4s" % end_time)
    print("PV - Evaluation: %s" % principal_variation.reward)
    print("PV - Moves: %s" % principal_variation.moves)
    # for move in principal_variation.moves:
    #     print(chess.Board(move))

    # Learning
    moves = fens_to_bitboards(principal_variation.moves)
    print("Moves: " + str(len(moves)))
    evaluations = compute_evaluations(moves, model)
    print("Evaluations: " + str(evaluations))
    tds = compute_tds(principal_variation.reward, evaluations)
    print("TDs: " + str(tds))
    lambda_value = 0.95
    lambda_tds = compute_lambda_tds(moves, tds, lambda_value)
    print("Lambda TDs: " + str(lambda_tds))
    targets = [principal_variation.reward] * len(moves)
    print("Targets: " + str(targets))

    model.train(moves, targets, lambda_tds)

    updated_evaluations = compute_evaluations(moves, model)
    print("Updated evaluations: " + str(updated_evaluations))


if __name__ == "__main__":
    main()
