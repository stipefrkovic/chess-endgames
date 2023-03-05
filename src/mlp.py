import tensorflow as tf


class MLP:
    def __init__(self):
        self.model = None
        self.model_checkpoint_dir = '/model/checkpoint'

    def build(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu, input_shape=(768,)),
            tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss=tf.keras.losses.MeanAbsoluteError,
                           )
        try:
            self.model.load_weights(self.model_checkpoint_dir)
        except(Exception,) as e:
            print(e)

    def train(self, boards, td_values):
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            mode='min',
            verbose=1,
            patience=4)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_checkpoint_dir,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)
        callbacks_list = [early_stopping_callback, model_checkpoint_callback]
        self.model.fit(boards, td_values,
                       epochs=12,
                       callbacks=callbacks_list,
                       )
