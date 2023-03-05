import chess


class Variation:
    def __init__(self, evaluation, move):
        self.evaluation = evaluation
        self.moves = [move]

    def add_move(self, move):
        self.moves.append(move)


def td_minimax(board, turn, depth, max_depth):
    outcome = board.outcome()
    if depth is max_depth or outcome is not None:
        if depth is max_depth:
            evaluation = 0
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

    variations = []
    for move in board.legal_moves():
        board.push(move)
        variation = td_minimax(board, board.turn, depth+1, max_depth)
        variations.append(variation)
        board.pop()
    if turn is chess.WHITE:
        principal_variation = max(variations, key=lambda x: x.evaluation)
    elif turn is chess.BLACK:
        principal_variation = min(variations, key=lambda x: x.evaluation)
    else:
        raise Exception("Error with turn.")
    principal_variation.add_move(board.fen())
    return principal_variation
