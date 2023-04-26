import time

import chess

from src.td_learning import compute_lambda_tds, compute_tds
from src.models import MLP
from src.search import alphabeta


def main():
    mate_in_1 = "8/8/8/k1K5/8/1R6/8/8 w - - 0 1"
    other_mate_in_1 = "3r2k1/4Qp2/5P1p/2p3p1/3q4/8/P1B1R1PP/4K3 b - - 6 28"
    mate_in_2 = "k7/8/2K5/8/8/1R6/8/8 w - - 0 1"
    other_mate_in_2 = "4rk1r/3R1ppp/p2p4/1p1p2B1/3q1P2/3B4/PPP3PP/2K1R3 w - - 2 18"
    mate_in_3_rook = "8/k7/8/2K5/8/1R6/8/8 w - - 0 1"
    mate_in_3_queen = "8/k7/8/2K5/8/1Q6/8/8 w - - 0 1"
    start_position = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1'

    positions = [other_mate_in_2, other_mate_in_1]

    model = MLP()
    model.build()

    for position in positions:
        board = chess.Board(position)
        print("Board:\n" + str(board))
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
        print("Principal Variation - Evaluation: %s" % principal_variation.reward)
        print("Principal Variation - Moves: %s" % principal_variation.moves)
        for move in principal_variation.moves:
            print(chess.Board(move))

        # Learning
        model.train(principal_variation)


if __name__ == "__main__":
    main()
