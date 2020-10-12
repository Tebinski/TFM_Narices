import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import os


class SeqModel:
    """ Sequential Neutal Net."""
    FOLDER = 'models/sequential'

    def __init__(self):
        pass

    def save_model(self, name):
        self.model.save(os.path.join(self.FOLDER, name))

    def load_model(self, name):
        self.model = keras.models.load_model(os.path.join(self.FOLDER, name))

    def _gen_model_seq(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(self.x_size, 1)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.y_size)
            ])
        return model

    def _gen_and_complile_model(self):
        model = self._gen_model_seq()
        print(model.summary())
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def model_train(self, X_train, y_train):
        # TensorFlow and tf.keras
        self.x_size = X_train.shape[1]
        self.y_size = y_train.shape[1]
        model = self._gen_and_complile_model()
        tb = TensorBoard(log_dir="logs/{}".format(time()))
        model.fit(X_train, y_train, epochs=30, callbacks=[tb])
        self.model = model

    def model_evaluate(self, X_test, y_test):
        model = self.model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print('\nTest accuracy:', test_acc)
        return test_loss, test_acc

    def get_model(self):
        return self.model


