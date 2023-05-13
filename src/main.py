import time
import chess

from models import MLP
from search import alphabeta
from position import create_random_position

def main():
    model = MLP()
    model.build()
    
    Krk_pieces = [
        chess.Piece(chess.KING, chess.WHITE),
        chess.Piece(chess.ROOK, chess.BLACK),
        chess.Piece(chess.KING, chess.BLACK),
    ]
    Krk_turn = False
    Krk_endgames = [create_random_position(Krk_pieces, Krk_turn) for i in range(1)]

    for endgame in Krk_endgames:
        board = chess.Board(endgame)
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
