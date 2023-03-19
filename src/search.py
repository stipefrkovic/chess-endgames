import chess


class Variation:
    def __init__(self, evaluation, move):
        self.evaluation = evaluation
        self.moves = [move]

    def add_move(self, move):
        self.moves.insert(0, move)


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


def alphabeta(board, model, depth, max_depth, alpha, beta):
    outcome = board.outcome()
    if depth is max_depth or outcome is not None:
        if outcome is None:
            # TODO fen to bitboard
            evaluation = model.predict(board.fen)
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

    if board.turn is chess.WHITE:
        principal_variation = Variation(-100, None)
        for move in board.legal_moves:
            board.push(move)
            variation = alphabeta(board, model, depth + 1, max_depth, alpha, beta)
            principal_variation = max([principal_variation, variation], key=lambda x: x.evaluation)
            board.pop()
            if principal_variation.evaluation > beta:
                break
            alpha = max(alpha, principal_variation.evaluation)
    else:
        principal_variation = Variation(100, None)
        for move in board.legal_moves:
            board.push(move)
            variation = alphabeta(board, model, depth + 1, max_depth, alpha, beta)
            principal_variation = min([principal_variation, variation], key=lambda x: x.evaluation)
            board.pop()
            if principal_variation.evaluation < alpha:
                break
            beta = min(beta, principal_variation.evaluation)
    principal_variation.add_move(board.fen())
    return principal_variation
