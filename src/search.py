import chess

class Variation:
    def __init__(self, reward, state):
        self.reward = reward
        self.states = [state]

    def add_state(self, state):
        self.states.insert(0, state)


class ChessState():
    def __init__(self, board):
        self.board = board
    
    def outcome(self):
        return self.board.outcome()

class AlphaBeta:
    def outcome_reward(self, outcome):
        if outcome is None:
            board_fen = board.fen()
            board_input = model.fen_to_model_input(board_fen)
            reward = model.predict(board_input)
        elif outcome.winner is None:
            reward = 0
        elif outcome.winner is chess.WHITE:
            reward = 1
        elif outcome.winner is chess.BLACK:
            reward = -1
        else:
            raise Exception("Error with outcome.")

    def run(self, state, model, depth, max_depth, alpha, beta):
        outcome = state.outcome()
        if depth is max_depth or outcome is not None:
            
            variation = Variation(reward, board.fen())
            return variation

        if board.turn is chess.WHITE:
            principal_variation = Variation(-100, None)
            for move in board.legal_moves:
                board.push(move)
                variation = self.run(board, model, depth + 1, max_depth, alpha, beta)
                principal_variation = max([principal_variation, variation], key=lambda x: x.reward)
                board.pop()
                if principal_variation.reward > beta:
                    break
                alpha = max(alpha, principal_variation.reward)
        else:
            principal_variation = Variation(100, None)
            for move in board.legal_moves:
                board.push(move)
                variation = self.run(board, model, depth + 1, max_depth, alpha, beta)
                principal_variation = min([principal_variation, variation], key=lambda x: x.reward)
                board.pop()
                if principal_variation.reward < alpha:
                    break
                beta = min(beta, principal_variation.reward)
        principal_variation.add_state(board.fen())
        return principal_variation


class ChessAlphaBeta(Search):
    def __init__(self):
        super().__init__()

    def run(self, board, model, depth, max_depth, alpha, beta):
        outcome = board.outcome()
        if depth is max_depth or outcome is not None:
            if outcome is None:
                board_fen = board.fen()
                board_input = model.fen_to_model_input(board_fen)
                reward = model.predict(board_input)
            elif outcome.winner is None:
                reward = 0
            elif outcome.winner is chess.WHITE:
                reward = 1
            elif outcome.winner is chess.BLACK:
                reward = -1
            else:
                raise Exception("Error with outcome.")
            variation = Variation(reward, board.fen())
            return variation

        if board.turn is chess.WHITE:
            principal_variation = Variation(-100, None)
            for move in board.legal_moves:
                board.push(move)
                variation = self.run(board, model, depth + 1, max_depth, alpha, beta)
                principal_variation = max([principal_variation, variation], key=lambda x: x.reward)
                board.pop()
                if principal_variation.reward > beta:
                    break
                alpha = max(alpha, principal_variation.reward)
        else:
            principal_variation = Variation(100, None)
            for move in board.legal_moves:
                board.push(move)
                variation = self.run(board, model, depth + 1, max_depth, alpha, beta)
                principal_variation = min([principal_variation, variation], key=lambda x: x.reward)
                board.pop()
                if principal_variation.reward < alpha:
                    break
                beta = min(beta, principal_variation.reward)
        principal_variation.add_state(board.fen())
        return principal_variation
