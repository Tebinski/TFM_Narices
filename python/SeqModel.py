import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import os
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


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
        model.fit(X_train, y_train, epochs=50, callbacks=[tb])
        self.model = model

    def model_evaluate(self, X_test, y_test):
        model = self.model
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print('\nTest accuracy:', test_acc)
        return test_loss, test_acc

    def get_model(self):
        return self.model


class AbstractSequentialModel(ABC):

    @abstractmethod
    def split_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_and_save_model(self, modelname):
        seq = SeqModel()
        seq.model_train(self.X_train, self.y_train)
        seq.save_model(modelname)


class SeqModelSimple(AbstractSequentialModel):

    def __init__(self, df):
        self.df = df

    def split_data(self):
        ## First, we will NOT use concentration data
        df_gas = self.df
        gas_X = df_gas.drop(columns=['Batch ID', 'GAS', 'CONCENTRATION']).to_numpy()
        gas_y = pd.get_dummies(df_gas['GAS'], drop_first=False)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(gas_X, gas_y, test_size=0.33, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test


class SeqModelWithConcentration(AbstractSequentialModel):

    def __init__(self, df):
        self.df = df

    def split_data(self):
        ## Use concentration data
        df_gas = self.df
        gas_X = df_gas.drop(columns=['Batch ID', 'GAS']).to_numpy()
        gas_y = pd.get_dummies(df_gas['GAS'], drop_first=False)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(gas_X, gas_y, test_size=0.33, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test


if __name__ == '__main__':
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(128, 1)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(6)
    ])

    model.summary()