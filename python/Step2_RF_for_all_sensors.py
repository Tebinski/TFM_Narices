import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from python.LoadUciData import load_data
from python.LoadSensorData import get_sensors_list, get_sensor
from python.StandardFigure import save_figure
from python.RandomForestClasification import RandomForestMod


if __name__ == '__main__':

    #Load data
    df = load_data()

    df = get_sensors_list([0, 1, 8, 9])
    df = get_sensors_list([2, 3, 10, 11])
    df = get_sensors_list([4, 5, 12, 13])
    df = get_sensors_list([6, 7, 14, 15])

    for col in ['GAS']:
        df[col] = df[col].astype('category')

    df['is_train'] = np.where(np.logical_and(df['Batch ID'] >= 1, df['Batch ID'] <= 8), True, False)
    df['is_test']  = np.where(np.logical_and(df['Batch ID'] >= 9, df['Batch ID'] <= 9), True, False)

    df_train = df[df['is_train'] == True]
    df_test = df[df['is_test'] == True]

    X_train = df_train.drop(columns=['GAS', 'Batch ID', 'CONCENTRATION'])
    y_train = df_train['GAS']

    X_test = df_test.drop(columns=['GAS', 'Batch ID', 'CONCENTRATION'])
    y_test = df_test['GAS']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


    rfm = RandomForestMod()
    rfm.model_train(X_train.values, y_train.values)
    rfm.save_model('RF1')

    rf = rfm.get_model()
    rf.score(X_test, y_test)
    y_pred = rf.predict(X_test)
    conf = confusion_matrix(y_test, y_pred)
    conf_percent = conf.T/y_test.value_counts().sort_index().values


    fig = plt.figure(figsize=(8, 8))
    ax = sns.heatmap(conf_percent, annot=True, fmt=".02f", cmap='Blues')
    plt.yticks(rotation=0)
    plt.show()

    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))