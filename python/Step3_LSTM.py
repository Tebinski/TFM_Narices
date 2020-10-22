"""
Comprobarmos el rendimiento de las redes neuronales para realizar la tarea de clasificacion utilizando la informacion
de la concentracion dentro del dataset.
"""

from python.LSTMmodel import LSTMmodel1
from python.LoadUciData import load_data
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, \
    Dropout, TimeDistributed, Activation, Conv1D, MaxPooling1D

import numpy as np
import stat

def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

if __name__ == '__main__':

    df_gas = load_data()

    ## GAS CLASIFICATION
    ## First, we will NOT use concentration data
    gas_X = df_gas.drop(columns=['Batch ID', 'GAS', 'CONCENTRATION']).to_numpy()
    gas_y = df_gas['GAS'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(gas_X, gas_y,
                                                        test_size=0.33,
                                                        random_state=42)

    x_size = X_train.shape[0]
    y_size = y_train.shape[0]

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(x_size, 1)),
        keras.layers.LSTM(100, input_shape=[X_train.shape[1], X_train.shape[2]]),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(y_size)
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, epochs=3, batch_size=64)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
