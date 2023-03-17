import time
import chess


class Variation:
    def __init__(self, evaluation, move):
        self.evaluation = evaluation
        self.moves = [move]

    def add_move(self, move):
        self.moves.append(move)


# def minimax(board, depth, max_depth, turn):
#     outcome = board.outcome()
#     if depth is max_depth or outcome is not None:
#         if depth is max_depth:
#             evaluation = 0
#         elif outcome.winner is None:
#             evaluation = 0
#         elif outcome.winner is chess.WHITE:
#             evaluation = 1
#         elif outcome.winner is chess.BLACK:
#             evaluation = -1
#         else:
#             raise Exception("Error with outcome.")
#         variation = Variation(evaluation, board.fen())
#         return variation
#
#     variations = []
#     for move in board.legal_moves:
#         board.push(move)
#         variation = minimax(board, depth + 1, max_depth, board.turn)
#         variations.append(variation)
#         board.pop()
#     if turn is chess.WHITE:
#         principal_variation = max(variations, key=lambda x: x.evaluation)
#     elif turn is chess.BLACK:
#         principal_variation = min(variations, key=lambda x: x.evaluation)
#     else:
#         raise Exception("Error with turn.")
#     principal_variation.add_move(board.fen())
#     return principal_variation


def alphabeta(board, depth, max_depth, turn, alpha, beta):
    outcome = board.outcome()
    if depth is max_depth or outcome is not None:
        if outcome is None:
            evaluation = 0
            # TODO add real evaluation
        elif outcome.winner is None:
            evaluation = 0
        elif outcome.winner is chess.WHITE:
            evaluation = 1
        elif outcome.winner is chess.BLACK:
            evaluation = -1
        else:
            raise Exception("Error with outcome.")
        variation = Variation(evaluation, board.fen())
        return variation

    if turn is chess.WHITE:
        principal_variation = Variation(-100, None)
        for move in board.legal_moves:
            board.push(move)
            variation = alphabeta(board, depth + 1, max_depth, board.turn, alpha, beta)
            principal_variation = max([principal_variation, variation], key=lambda x: x.evaluation)
            board.pop()
            if principal_variation.evaluation > beta:
                break
            alpha = max(alpha, principal_variation.evaluation)
    elif turn is chess.BLACK:
        principal_variation = Variation(100, None)
        for move in board.legal_moves:
            board.push(move)
            variation = alphabeta(board, depth + 1, max_depth, board.turn, alpha, beta)
            principal_variation = min([principal_variation, variation], key=lambda x: x.evaluation)
            board.pop()
            if principal_variation.evaluation < alpha:
                break
            beta = min(beta, principal_variation.evaluation)
    else:
        raise Exception("Error with turn.")
    principal_variation.add_move(board.fen())
    return principal_variation


def main():
    mate_in_1 = "8/8/8/k1K5/8/1R6/8/8 w - - 0 1"
    mate_in_2 = "k7/8/2K5/8/8/1R6/8/8 w - - 0 1"
    mate_in_3_rook = "8/k7/8/2K5/8/1R6/8/8 w - - 0 1"
    mate_in_3_queen = "8/k7/8/2K5/8/1Q6/8/8 w - - 0 1"

    # board = chess.Board(mate_in_3_queen)
    # start_time = time.time()
    # principal_variation = minimax(board, 0, 5, board.turn)
    # print("Minimax - Time: %.4s, Evaluation: %s" % (time.time() - start_time, principal_variation.evaluation))

    board = chess.Board(mate_in_3_queen)
    start_time = time.time()
    principal_variation = alphabeta(board, 0, 5, board.turn, -100, 100)
    print("Alphabeta - Time: %.4s, Evaluation: %s" % (time.time() - start_time, principal_variation.evaluation))

    # print(principal_variation.evaluation)
    # for move in principal_variation.moves:
    #     print(chess.Board(move))


if __name__ == "__main__":
    main()
