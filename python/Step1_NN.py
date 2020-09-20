import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

from python.LoadUciData import load_data
from python.StandardFigure import save_figure


def gen_model_seq(x_size):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(x_size, 1)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10)
        ])
    return model

def gen_and_complile_model(x_size):
    model = gen_model_seq(x_size)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def model_train(X_train, y_train):
    # TensorFlow and tf.keras
    x_size = X_train.shape[1]
    model = gen_model_seq(x_size)
    model.summary()

    model = gen_and_complile_model(x_size)
    tb = TensorBoard(log_dir="logs/{}".format(time()))
    model.fit(X_train, y_train, epochs=30, callbacks=[tb])
    return model

def model_evaluate(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    return test_loss, test_acc

def parte1():
    # Load data
    df_gas = load_data()

    ## GAS CLASIFICATION
    ## First, we will NOT use concentration data
    gas_X = df_gas.drop(columns=['Batch ID', 'GAS', 'CONCENTRATION']).to_numpy()
    gas_y = df_gas['GAS'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(gas_X, gas_y,
                                                        test_size=0.33,
                                                        random_state=42)

    model = model_train(X_train, y_train)
    test_loss0, test_acc0 = model_evaluate(model, X_test, y_test)

    # Check the changes if we include concentration info
    gas_X = df_gas.drop(columns=['Batch ID', 'GAS']).to_numpy()
    gas_y = df_gas['GAS'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(gas_X, gas_y,
                                                        test_size=0.33,
                                                        random_state=42)
    model = model_train(X_train, y_train)
    test_loss1, test_acc1 = model_evaluate(model, X_test, y_test)

def evolution_drift(N_BATCH_TRANING):
    ## Check the drift importance. Use the first 3 batch to train, and check the
    ## clasifications results with the others.
    df_gas = load_data()

    df_gas_train = df_gas[df_gas['Batch ID'].isin(range(1, N_BATCH_TRANING+1))]
    gas_train_X = df_gas_train.drop(columns=['Batch ID', 'GAS']).to_numpy()
    gas_train_y = df_gas_train['GAS'].to_numpy()
    model = model_train(gas_train_X, gas_train_y)

    model_dict = {}
    for batch in range(N_BATCH_TRANING+1, 11):
        df_gas_test  = df_gas[df_gas['Batch ID'] == batch]
        gas_test_X  = df_gas_test.drop(columns=['Batch ID', 'GAS']).to_numpy()
        gas_test_y  = df_gas_test['GAS'].to_numpy()

        print(f'Batch {batch}')

        loss, acc  = model_evaluate(model, gas_test_X, gas_test_y)
        model_dict[batch] = {'acc': acc, 'loss': loss}

    # figuras
    df_results = pd.DataFrame.from_dict(model_dict).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
    fig.suptitle(f'Training with first {N_BATCH_TRANING} batches')
    ax1 = df_results.plot(kind='bar', y='acc', ax=ax1)
    ax1.set_ylim([0,1])
    ax2 = df_results.plot(kind='bar', y='loss', ax=ax2)
    save_figure(fig, f'Step1_NBATCH_{N_BATCH_TRANING}_acc_loss')


if __name__ == '__main__':
    for i in range(1, 10):
        evolution_drift(i)