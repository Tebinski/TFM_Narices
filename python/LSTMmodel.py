import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, \
    Dropout, TimeDistributed, Activation, Conv1D, MaxPooling1D

def reshape_data_uci(x):
    num_steps = 30
    num_features = 128
    x_shaped = np.reshape(x, newshape=(-1, num_steps, num_features))
    return x_shaped


class LSTMmodel2:
    """
    https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
    """
    def __init__(self):
        pass

    def _structure(self):
        # create the model
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(6))
        return model

    def _gen_and_complile_model(self):
        model = self._structure()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def model_train(self, X_train, y_train):
        model = self._gen_and_complile_model()
        model.fit(X_train, y_train, epochs=3, batch_size=64)

    def model_evaluate(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=2)
        print('\nTest accuracy:', test_acc)
        return test_loss, test_acc


class LSTMmodel1:
    def __init__(self):
        pass

    def _structure(self):
        # create the model
        model = Sequential()
        model.add(LSTM(100))
        model.add(Dense(6))
        return model

    def _gen_and_complile_model(self):
        model = self._structure()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def model_train(self, X_train, y_train):
        model = self._gen_and_complile_model()
        model.fit(X_train, y_train, epochs=3, batch_size=64)

    def model_evaluate(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=2)
        print('\nTest accuracy:', test_acc)
        return test_loss, test_acc
