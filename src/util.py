# TODO one np array, adjust print_bitboard accordingly
import numpy as np


def fen_to_bitboard_deprecated(fen):
    new_row = '/'
    end_of_board = ' '
    empty_squares = '12345678'
    pieces = 'kqrbnp'
    boards = [[] for i in range(len(pieces))]
    for c in fen:
        if c == new_row:
            continue
        elif c == end_of_board:
            break
        elif c in empty_squares:
            for board in boards:
                board.extend([0] * int(c))
        else:
            piece_idx = pieces.lower().find(c)
            piece_color = -1
            if piece_idx == -1:
                piece_idx = pieces.upper().find(c)
                piece_color = 1
            for board_idx, board in enumerate(boards):
                if board_idx == piece_idx:
                    board.append(piece_color)
                else:
                    board.append(0)
    return boards


def fen_to_bitboard(fen):
    boards = [[] for i in range(6)]
    check_turn = False
    end_of_pieces = ' '
    next_row = '/'
    empty_squares = '12345678'
    pieces = 'kqrbnp'
    for char in fen:
        if check_turn:
            bitboard = np.array(boards).flatten()
            print(bitboard.shape)
            if char == 'w':
                bitboard = np.append(bitboard, 1)
            else:
                bitboard = np.append(bitboard, 0)
            print(bitboard.shape)
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
