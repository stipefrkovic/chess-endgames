import chess

class Variation:
    def __init__(self, reward, state):
        self.reward = reward
        self.states = [state]

    def add_state(self, state):
        self.states.insert(0, state)

    def get_states(self):
        return self.states
    
    def get_reward(self):
        return self.reward

# TODO create class State

class ChessState():
    def __init__(self, board):
        self.board = board
        self.real_reward = 1 if self.max_turn() else -1

    def __str__(self):
        return self.board.__str__()

    def string(self):
        return self.board.fen()

    def copy(self):
        return ChessState(self.board.copy())
    
    def max_turn(self):
        return self.board.turn is chess.WHITE
    
    def get_actions(self):
        return self.board.legal_moves
    
    def do_action(self, action):
        self.board.push(action)

    def undo_action(self):
        return self.board.pop()
    
    def get_outcome(self):
        return self.board.outcome()

    def get_reward(self, outcome, model):
        if outcome is None:
            model_input = model.fen_to_model_input(self.string())
            reward = model.predict(model_input)
        elif outcome.winner is None:
            reward = 0
        elif outcome.winner is chess.WHITE:
            reward = 1
        elif outcome.winner is chess.BLACK:
            reward = -1
        else:
            raise Exception("Error with outcome.")
        return reward


def alpha_beta(state, model, depth, max_depth, alpha, beta):
    outcome = state.get_outcome()
    if depth is max_depth or outcome is not None:
        reward = state.get_reward(outcome, model)
        variation = Variation(reward, state.copy())
        return variation
    if state.max_turn():
        principal_variation = Variation(-100, None)
        for action in state.get_actions():
            state.do_action(action)
            variation = alpha_beta(state, model, depth + 1, max_depth, alpha, beta)
            principal_variation = max([principal_variation, variation], key=lambda x: x.reward)
            state.undo_action()
            if principal_variation.reward > beta:
                break
            alpha = max(alpha, principal_variation.reward)
    else:
        principal_variation = Variation(100, None)
        for action in state.get_actions():
            state.do_action(action)
            variation = alpha_beta(state, model, depth + 1, max_depth, alpha, beta)
            principal_variation = min([principal_variation, variation], key=lambda x: x.reward)
            state.undo_action()
            if principal_variation.reward < alpha:
                break
            beta = min(beta, principal_variation.reward)
    principal_variation.add_state(state.copy())
    return principal_variation
