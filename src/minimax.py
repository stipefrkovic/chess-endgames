import chess


class Variation:
    def __init__(self, evaluation, move):
        self.evaluation = evaluation
        self.moves = [move]

    def add_move(self, move):
        self.moves.append(move)


# TODO implement alpha beta pruning
# TODO prefer shorter variations
def td_minimax(board, turn, depth, max_depth):
    outcome = board.outcome()
    if depth is max_depth or outcome is not None:
        if depth is max_depth:
            # TODO add real evaluation
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
    for move in board.legal_moves:
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


def main():
    mate_in_1 = "8/8/8/k1K5/8/1R6/8/8 w - - 0 1"
    mate_in_2 = "k7/8/2K5/8/8/1R6/8/8 w - - 0 1"
    mate_in_3 = "8/k7/8/2K5/8/1R6/8/8 w - - 0 1"
    board = chess.Board(mate_in_2)
    principal_variation = td_minimax(board, chess.WHITE, 0, 4)
    print(principal_variation.evaluation)
    for move in principal_variation.moves:
        print(chess.Board(move))


if __name__ == "__main__":
    main()
