"""
Vamos a demotrar que el drift efectivamente ocurre en las muestras de los sensores.

Se demostrará de la siguiente forma:
- Entrenar con los datos de un batch, testear con los datos de los batchs anteriores y posteriores.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from python.LoadUciData import load_data
from python.StandardFigure import save_figure
from python.SeqModel import SeqModel


class Evolution_drift:

    def __init__(self, df, n_batch_training, batch_test):
        self.df = df
        self.n_batch_training = n_batch_training
        self.batch_test = batch_test

    def split_data(self):
        df_gas = self.df

        df_gas_train = df_gas[df_gas['Batch ID'].isin(range(1, self.n_batch_training + 1))]
        self.X_train = df_gas_train.drop(columns=['Batch ID', 'GAS']).to_numpy()
        self.y_train = pd.get_dummies(df_gas_train['GAS'], prefix='GAS')

        df_gas_test = df_gas[df_gas['Batch ID'] == self.batch_test]
        self.X_test = df_gas_test.drop(columns=['Batch ID', 'GAS']).to_numpy()
        self.y_test = pd.get_dummies(df_gas_test['GAS'], prefix='GAS')
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_and_save_model(self, modelname):
        seq = SeqModel()
        seq.model_train(self.X_train, self.y_train)
        self.test_loss0, self.test_acc0 = seq.model_evaluate(self.X_test, self.y_test)
        seq.save_model(modelname)


def evo_train(df_gas, n_batch_training):
    for batch in range(n_batch_training + 1, 11):
        print(f'\n\n----------Batch -----------{batch} \n\n')
        ev = Evolution_drift(df_gas, n_batch_training, batch)
        X_train, X_test, y_train, y_test = ev.split_data()
        model_name = f'temp_training_{n_batch_training}_test{batch}'
        ev.train_and_save_model(model_name)


def evolution_drift_with_seq(df_gas, n_batch_training):
    ## Check the drift importance. Use the first N batch to train, and check the
    ## clasifications results with the others.

    #Train
    evo_train(df_gas, n_batch_training)

    # Test and save results in dict
    model_dict = {}
    for batch in range(n_batch_training + 1, 11):
        print(f'\n\n----------Batch 2-----------{batch} \n\n')
        seq = SeqModel()
        model_name = f'temp_training_{n_batch_training}_test{batch}'
        model = seq.load_model(model_name)

        ev = Evolution_drift(df_gas, n_batch_training, batch)
        _, X_test, _, y_test = ev.split_data()
        loss, acc = seq.model_evaluate(X_test, y_test)
        model_dict[batch] = {'acc': acc, 'loss': loss}

    #Plot results
    df_results = pd.DataFrame.from_dict(model_dict).T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Training with first {n_batch_training} batches')
    ax1 = df_results.plot(kind='bar', y='acc', ax=ax1)
    ax1.set_ylim([0, 1])
    ax2 = df_results.plot(kind='bar', y='loss', ax=ax2)
    save_figure(fig, f'Step1_NBATCH_{n_batch_training}_acc_loss')


if __name__ == '__main__':
    # Check drift relevance  with different training sets.
    df_gas = load_data()

    for col in ['GAS', 'Batch ID']:
        df_gas[col] = df_gas[col].astype('category')

    for n in range(1, 10+1):
        evo_train(df_gas, n_batch_training=n)

    for n in range(1, 10+1):
        evolution_drift_with_seq(df_gas, n_batch_training=n)

