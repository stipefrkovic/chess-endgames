# TODO castling rights, side to play
def fen_to_bitboard(fen):
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


def print_bitboard(bitboard):
    res_list = [0 for i in range(64)]
    import operator
    for b in bitboard:
        for i in range(64):
            res_list[i] = operator.add(res_list[i], b[i])
    for i in range(8):
        print(res_list[i*8:(i+1)*8])
