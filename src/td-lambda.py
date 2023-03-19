# import chess
#
# import minimax
# from src.mlp import Model


# TODO castling rights, side to play
def fen_to_bitboard(fen):
    target_pieces = ['k', 'q', 'r', 'b', 'n', 'p']
    board = []
    for target_piece in target_pieces:
        target_piece_board = []
        for fen_row in fen.split('/'):
            for piece in fen_row:
                if piece == ' ':
                    break
                elif piece in '12345678':
                    target_piece_board.extend([0] * int(piece))
                elif piece == target_piece.lower():
                    target_piece_board.append(-1)
                elif piece == target_piece.upper():
                    target_piece_board.append(1)
                else:
                    target_piece_board.append(0)
        board.append(target_piece_board)
    return board


def compute_tds(model, variation):
    states = variation.moves.copy()
    states.reverse()
    evaluations = [model.predict(state) for state in states]
    evaluations.append(variation.evaluation)
    tds = [evaluations[i+1] - evaluations[i] for i in range(len(evaluations)-1)]
    print(len(states), len(tds))
    return tds


def compute_fit_data(variation, tds, lambda_value):
    states = variation.moves.copy()
    states.reverse()
    input_data = []
    output_data = []
    for state_idx, state in enumerate(states):
        input_data.append(fen_to_bitboard(state))
        target_value = 0.0
        for td_idx, td in enumerate(tds[state_idx:]):
            target_value += pow(lambda_value, td_idx) * td
            print("state: " + state + " value: " + str(target_value))
        output_data.append(target_value)
    return input_data, output_data


def main():
    mate_in_3_queen = "8/k7/8/2K5/8/1Q6/8/8 w - - 0 1"
    mate_in_1 = "8/8/8/k1K5/8/1R6/8/8 w - - 0 1"
    start_position = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

    board = fen_to_bitboard(start_position)

    # board = chess.Board(mate_in_3_queen)
    # principal_variation = minimax.alphabeta(board, 0, 5, board.turn, -100, 100)
    # model = Model()
    # tds = compute_tds(model, principal_variation)
    # print(tds)
    # fit_data = compute_fit_data(principal_variation, tds, 0.95)
    # print(fit_data[1])


if __name__ == "__main__":
    main()
