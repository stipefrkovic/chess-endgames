import os
import numpy as np
import tensorflow as tf
from pathlib import Path

from logger import logger

def compute_tds(reward, predictions):
    predictions = predictions.copy()
    predictions.append(reward)
    tds = [predictions[i+1] - predictions[i] for i in range(len(predictions)-1)]
    return tds

def compute_lambda_td(tds, lambda_value):
    lambda_td = 0.0
    for td_idx, td in enumerate(tds):
        lambda_td += pow(lambda_value, td_idx) * td
    return lambda_td

def compute_lambda_tds(tds, lambda_value):
    lambda_tds = []
    for state_idx in range(len(tds)):
        lambda_tds.append(compute_lambda_td(tds[state_idx:], lambda_value))
    return lambda_tds


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
            logger.warning("Model not found")

    def save_weights(self):
        logger.info("Attempting to save weights")
        self.model.save_weights(self.model_weights_path)
        logger.info("Saved weights")

    def train(self, variation, steps, lambda_value, game):
        pass

    def variation_to_inputs(self, variation):
        pass

    def predict_inputs(self, inputs):
        predictions = []
        for inp in inputs:
            prediction = self.predict(inp)
            predictions.append(prediction)
        return predictions

    def predict(self, x):
        prediction = self.model(x)
        prediction = prediction.numpy().item(0)
        return prediction


class MLP(Model):
    def __init__(self, model_weights_path):
        super().__init__(model_weights_path)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def train(self, variation, steps, lambda_value):
        reward = variation.get_reward()
        inputs = self.variation_to_inputs(variation)
        
        old_predictions = self.predict_inputs(inputs)
        logger.info(f"old_predictions: {old_predictions}")
        
        old_tds = compute_tds(reward, old_predictions)
        old_lambda_tds = compute_lambda_tds(old_tds, lambda_value)
        
        for step in range(steps):
            input_states = inputs.copy()
            while len(input_states) > 0:
                predictions = self.predict_inputs(input_states)
                # logger.debug(f"predictions: {predictions}")
                tds = compute_tds(reward, predictions)
                # logger.debug(f"tds: {tds}")
                lambda_td = compute_lambda_td(tds, lambda_value)
                # logger.debug(f"lambda_td: {lambda_td}")
                input_state = input_states.pop(0)
                with tf.GradientTape() as tape:
                    predicted_value = self.model(input_state)
                    predicted_value *= -lambda_td
                gradients = tape.gradient(predicted_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        
        new_predictions = self.predict_inputs(inputs)
        logger.info(f"new_predictions: {new_predictions}")
        
        new_tds = compute_tds(reward, new_predictions)
        new_lambda_tds = compute_lambda_tds(new_tds, lambda_value)
        
        return {
            "old_predictions": old_predictions,
            "new_predictions": new_predictions,
            "old_lambda_tds": old_lambda_tds,
            "new_lambda_tds": new_lambda_tds
        }


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
        inputs = [self.fen_to_model_input(chess_state.string()) for chess_state in chess_states]
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
