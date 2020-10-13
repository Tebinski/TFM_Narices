"""
Comprobarmos el rendimiento de las redes neuronales para realizar la tarea de clasificacion
"""


import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from python.LoadUciData import load_data
from python.StandardFigure import save_figure
from python.SeqModel import SeqModel, SeqModelSimple, SeqModelWithConcentration


def plot_conf(model, X_test, y_test):

    # Plot the confusion matrix
    f = plt.figure(figsize=(8, 6))
    # First add a Softmax layer to get probabilities.
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(X_test)
    y_test_values = np.argmax(y_test.values, axis=1)
    gas_predictions = np.argmax(predictions, axis=1)
    confusion = confusion_matrix(y_test_values, gas_predictions)

    label_gas = ['Gas1', 'Gas2', 'Gas3', 'Gas4', 'Gas5', 'Gas6']
    df_conf = pd.DataFrame(data=confusion, columns=label_gas, index=label_gas)
    sns.heatmap(df_conf, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.yticks(rotation=0)
    plt.show()
    return f


if __name__ == '__main__':

    # Check the results for the sequential Neural Net
    # Load data
    df_gas = load_data()

    mod1 = SeqModelSimple(df_gas)
    X_train, X_test, y_train, y_test = mod1.split_data()
    #mod1.train_and_save_model('ModelSimple')

    mod2 = SeqModelWithConcentration(df_gas)
    X_train2, X_test2, y_train2, y_test2 = mod2.split_data()
    #mod2.train_and_save_model('ModelWithConcentration')


    seq = SeqModel()
    seq.load_model('ModelSimple')
    f = plot_conf(seq.get_model(), X_test, y_test)
    save_figure(f, 'ConfMatrix_ModelSimple')

    seq = SeqModel()
    seq.load_model('ModelWithConcentration')
    f = plot_conf(seq.get_model(), X_test2, y_test2)
    save_figure(f, 'ConfMatrix_ModelWithConcentration')