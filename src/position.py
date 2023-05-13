import random
import chess

mate_in_1 = "8/8/8/k1K5/8/1R6/8/8 w - - 0 1"
other_mate_in_1 = "3r2k1/4Qp2/5P1p/2p3p1/3q4/8/P1B1R1PP/4K3 b - - 6 28"
mate_in_2 = "k7/8/2K5/8/8/1R6/8/8 w - - 0 1"
other_mate_in_2 = "4rk1r/3R1ppp/p2p4/1p1p2B1/3q1P2/3B4/PPP3PP/2K1R3 w - - 2 18"
mate_in_3_rook = "8/k7/8/2K5/8/1R6/8/8 w - - 0 1"
mate_in_3_queen = "8/k7/8/2K5/8/1Q6/8/8 w - - 0 1"
start_position = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1'

def create_random_position(pieces, turn):
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
        if board.is_valid():
            print(f"turn: {board.turn}")
            break

    return board.fen()