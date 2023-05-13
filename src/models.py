import os

import numpy as np
import tensorflow as tf

import td_learning


class Model:
    def __init__(self):
        self.model = None

    def compute_evaluations(self, moves):
        evaluations = []
        for move in moves:
            evaluation = self.predict(move)
            evaluations.append(evaluation)
        return evaluations

    def predict(self, x):
        prediction = self.model(x)
        prediction = prediction.numpy().item(0)
        return prediction


class MLP(Model):
    def __init__(self, model_weights_path='weights/weights.h5', steps=15):
        super().__init__()

        self.model_weights_path = model_weights_path
        self.steps = steps

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.MeanAbsoluteError()

    def build(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, input_shape=(384,)),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh),
        ])
        print(self.model.summary())

    def load_weights(self):
        if os.path.isfile(self.model_weights_path):
            self.model.load_weights(self.model_weights_path)

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           )

    def train_old(self, bitboards, td_values):
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            mode='min',
            verbose=1,
            patience=5)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_weights_path,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)
        callbacks_list = [early_stopping_callback]
        self.model.fit(bitboards, td_values,
                       epochs=50,
                       callbacks=callbacks_list,
                       )

    def train(self, variation):
        inputs = MLP.fens_to_model_inputs(variation.moves)
        evaluations = self.compute_evaluations(inputs)
        print("Model's old evaluations: " + str(evaluations))
        reward = variation.reward
        for step in range(self.steps):
            input_states = inputs.copy()
            while len(input_states) > 0:
                print(f"inputs: {len(input_states)}")
                evaluations = self.compute_evaluations(input_states)
                tds = td_learning.compute_tds(reward, evaluations)
                print(f"tds: {tds}")
                lambda_value = 0.95
                lambda_td = td_learning.compute_lambda_td(tds, lambda_value)
                print(f"lambda_td: {lambda_td}")
                input_state = input_states.pop(0)
                with tf.GradientTape() as tape:
                    predicted_value = self.model(input_state)
                    predicted_value *= -lambda_td
                gradients = tape.gradient(predicted_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        updated_evaluations = self.compute_evaluations(inputs)
        print("Model's new evaluations: " + str(updated_evaluations))

    def save_weights(self):
        self.model.save_weights(self.model_weights_path)

    @staticmethod
    def fens_to_model_inputs(fens):
        inputs = [MLP.fen_to_model_input(fen) for fen in fens]
        return inputs

    @staticmethod
    def fen_to_model_input(fen):
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
