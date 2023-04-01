import numpy as np


def fens_to_bitboards(fens):
    bitboards = []
    for fen in fens:
        bitboards.append(fen_to_bitboard(fen))
    return bitboards


def fen_to_bitboard(fen):
    boards = [[] for i in range(6)]
    check_turn = False
    end_of_pieces = ' '
    next_row = '/'
    empty_squares = '12345678'
    pieces = 'kqrbnp'
    for char in fen:
        if check_turn:
            bitboard = np.array(boards).reshape((1, 384))
            if char == 'b':
                bitboard = bitboard * -1
            return bitboard
        elif char == end_of_pieces:
            check_turn = True
        elif char == next_row:
            continue
        elif char in empty_squares:
            for board in boards:
                board.extend([0] * int(char))
        else:
            piece_idx = pieces.lower().find(char)
            piece_color = -1
            if piece_idx == -1:
                piece_idx = pieces.upper().find(char)
                piece_color = 1
            for board_idx, board in enumerate(boards):
                if board_idx == piece_idx:
                    board.append(piece_color)
                else:
                    board.append(0)
