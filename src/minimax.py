class Node:
    def __init__(self, terminate, evaluation, root_nodes):
        self.terminate = terminate
        self.evaluation = evaluation
        self.root_nodes = root_nodes


def minimax(node, max_player):
    if node.terminate is True:
        principal_variation = [node]
        return [node.evaluation, principal_variation]
    variations = []
    for root_node in node.root_nodes:
        variation = minimax(root_node, not max_player)
        variations.append(variation)
    if max_player:
        principal_variation = max(variations, key=lambda x: x[0])
    else:
        principal_variation = min(variations, key=lambda x: x[0])
    principal_variation[1].append(node)
    return principal_variation


# def minimax(board, turn):
#     outcome = board.outcome()
#     if outcome is not None:
#         principal_variation = [board]
#         if outcome.winner is None:
#             return 0
#         elif outcome.winner is chess.WHITE:
#             return 1
#         elif outcome.winner is chess.BLACK:
#             return -1
#         else:
#             raise Exception("Winner not recognised.")
#     evals = []
#     for move in board.legal_moves():
#         board_copy = board.copy()
#         board_copy.push(move)
#         eval = minimax(board_copy, board.turn)
#         evals.append(eval)
#     if turn is chess.WHITE:
#         return np.argmax(evals)
#     elif turn is chess.BLACK:
#         return min(evals)
#     else:
#         raise Exception("Turn not recognised.")


def main():
    node30 = Node(True, 5, [])
    node31 = Node(True, 3, [])
    node32 = Node(True, 2, [])
    node33 = Node(True, 1, [])
    node34 = Node(True, 7, [])
    node35 = Node(True, 3, [])
    node36 = Node(True, 2, [])
    node37 = Node(True, 4, [])
    node20 = Node(False, 0, [node30, node31])
    node21 = Node(False, 0, [node32, node33])
    node22 = Node(False, 0, [node34, node35])
    node23 = Node(False, 0, [node36, node37])
    node10 = Node(False, 0, [node20, node21])
    node11 = Node(False, 0, [node22, node23])
    node00 = Node(False, 0, [node10, node11])
    principal_variation = minimax(node00, True)
    print(principal_variation)


if __name__ == "__main__":
    main()
