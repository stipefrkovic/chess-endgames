import os
import tensorflow as tf


class Model:
    # TODO implement
    def predict(self, state):
        return 0.5


class MLP:
    def __init__(self, model_weights_path='weights/weights.h5'):
        self.model_weights_path = model_weights_path
        self.model = None

    def build(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, input_shape=(768,)),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
        ])

    def load_weights(self):
        if os.path.isfile(self.model_weights_path):
            self.model.load_weights(self.model_weights_path)

    def compile(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.MeanAbsoluteError,
                           )

    # TODO figure out target value vs td-lambda thing
    def train(self, boards, td_values):
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
        self.model.fit(boards, td_values,
                       epochs=20,
                       callbacks=callbacks_list,
                       )

    def save_weights(self):
        self.model.save_weights(self.model_weights_path)

    def predict(self, x):
        return self.model.predict(x)


