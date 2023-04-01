import os
import tensorflow as tf

from src.learning import compute_evaluations


class MLP:
    def __init__(self, model_weights_path='weights/weights.h5', epochs=20):
        super().__init__()

        self.model_weights_path = model_weights_path
        self.epochs = epochs

        self.model = None

    def build(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, input_shape=(384,)),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
        ])
        print(self.model.summary())

    def load_weights(self):
        if os.path.isfile(self.model_weights_path):
            self.model.load_weights(self.model_weights_path)

    def compile(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.MeanSquaredError,
                           )

    def train(self, bitboards, td_values):
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

    def custom_train(self, input_states, target_values, lambda_td_values):
        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.MeanAbsoluteError()
        for epoch in range(self.epochs):
            for input_state, target_value, td_value in zip(input_states, target_values, lambda_td_values):
                with tf.GradientTape() as tape:
                    predicted_value = self.model(input_state)
                    loss_value = loss(target_value, predicted_value)
                    # print("Predicted, Target: %s, %s" % (predicted_value, target_value))
                gradients = tape.gradient(loss_value, self.model.trainable_weights)
                # gradients *= td_value
                optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

    def save_weights(self):
        self.model.save_weights(self.model_weights_path)

    def predict(self, x):
        prediction = self.model(x)
        prediction = prediction.numpy().item(0)
        return prediction

