import os
import numpy as np
import tensorflow as tf
from pathlib import Path

from my_logger import logger

def compute_tds(reward, evaluations):
    evaluations = evaluations
    evaluations.append(reward)
    tds = [evaluations[i+1] - evaluations[i] for i in range(len(evaluations)-1)]
    return tds


def compute_lambda_tds(tds, lambda_value):
    lambda_tds = []
    for state_idx in range(len(tds)):
        lambda_tds.append(compute_lambda_td(tds[state_idx:], lambda_value))
    return lambda_tds


def compute_lambda_td(tds, lambda_value):
    lambda_td = 0.0
    for td_idx, td in enumerate(tds):
        lambda_td += pow(lambda_value, td_idx) * td
    return lambda_td


class Model:
    def __init__(self, model_weights_path):
        self.model_weights_path = str(Path().absolute()) + model_weights_path
        self.model = None

    def build(self):
        pass

    def load_weights(self):
        logger.info("Attempting to load model")
        if os.path.isfile(self.model_weights_path):
            self.model.load_weights(self.model_weights_path)
            logger.info("Loaded model")
        else:
            logger.info("Model not found")

    def save_weights(self):
        logger.info("Attempting to save weights")
        self.model.save_weights(self.model_weights_path)
        logger.info("Saved weights")

    def train(self, variation, steps, lambda_value):
        pass

    def variation_to_inputs(self, variation):
        pass

    def evaluate_inputs(self, inputs):
        evaluations = []
        for inp in inputs:
            evaluation = self.predict(inp)
            evaluations.append(evaluation)
        return evaluations

    def predict(self, x):
        prediction = self.model(x)
        prediction = prediction.numpy().item(0)
        return prediction


class MLP(Model):
    def __init__(self, model_weights_path):
        super().__init__(model_weights_path)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def train(self, variation, steps, lambda_value):
        inputs = self.variation_to_inputs(variation)
        evaluations = self.evaluate_inputs(inputs)
        # logger.info(f"Model Old Evals: {evaluations}")
        reward = variation.get_reward()
        for step in range(steps):
            input_states = inputs.copy()
            while len(input_states) > 0:
                evaluations = self.evaluate_inputs(input_states)
                # logger.debug(f"evaluations: {evaluations}")
                tds = compute_tds(reward, evaluations)
                # logger.debug(f"tds: {tds}")
                lambda_td = compute_lambda_td(tds, lambda_value)
                # logger.debug(f"lambda_td: {lambda_td}")
                input_state = input_states.pop(0)
                with tf.GradientTape() as tape:
                    predicted_value = self.model(input_state)
                    predicted_value *= -lambda_td
                gradients = tape.gradient(predicted_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        # updated_evaluations = self.evaluate_inputs(inputs)
        # logger.info(f"Model New Evals: {updated_evaluations}")


class ChessMLP(MLP):
    def __init__(self, model_weights_path='/src/weights/mlp_weights.h5'):
        super().__init__(model_weights_path)

    def build(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, input_shape=(384,)),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh),
        ])
        logger.debug(f"Model Summary:\n{str(self.model.summary())}")

    def variation_to_inputs(self, variation):
        return self.chess_states_to_model_inputs(variation.get_states())

    def chess_states_to_model_inputs(self, chess_states):
        inputs = [self.fen_to_model_input(chess_state.get_string()) for chess_state in chess_states]
        return inputs

    def fen_to_model_input(self, fen):
        boards = [[] for i in range(6)]
        check_turn = False
        end_of_pieces = ' '
        next_row = '/'
        empty_squares = '12345678'
        pieces = 'kqrbnp'
        for char in fen:
            if check_turn:
                inp = np.array(boards).reshape((1, 384))
                if char == 'b':
                    inp = inp * -1
                return inp
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
