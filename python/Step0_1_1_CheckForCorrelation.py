"""
Comprobamos si existe correlacion entre las señales de los diferentes sensores

Efectivamente, al haber usado 4 tipos de sensores, los sensores del mismo tipo han obtenido
señales muy similares. Esto podria dar problemas de redundancia a la hora de entrenar un modelo,
por lo que tendremos que separa sensores correlacionados.

"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from python.LoadUciData import load_data
from python.StandardFigure import save_figure
from python.LoadSensorData import get_sensors_list

if __name__ == '__main__':

    # Cargo datos
    df = load_data()

    corr = df.iloc[:, :128].corr()
    fig = plt.figure(figsize=(20,20))
    sns.heatmap(corr, vmin=-1, vmax=1,
                cmap='coolwarm',
                square=True)
    plt.title('Correlation Between Features')
    plt.show()
    save_figure(fig, 'Step0_1_1_CorrelationBetweenFeatures')

    # Obtenemos un dataframe con solo un sensor de Tipo I, II, III y IV
    df_sens = get_sensors_list([0, 2, 4, 6])
    sensors_features = df_sens.drop(['Batch ID', 'GAS','CONCENTRATION'],axis=1)
    fig = plt.figure(figsize=(20, 20));
    ax = sns.heatmap(sensors_features.corr(),
                vmin=-1, vmax=1, annot=True, cmap='coolwarm');
    ax.set_title('Correlation between SensorPack1')
    plt.show()
    save_figure(fig, 'Step0_1_1_CorrelationBetweenFeatures_Data1')

    # Otra combinacion de 4 sensores
    df_sens = get_sensors_list([1, 3, 5, 7])
    sensors_features = df_sens.drop(['Batch ID', 'GAS', 'CONCENTRATION'], axis=1)
    fig = plt.figure(figsize=(20, 20));
    ax = sns.heatmap(sensors_features.corr(),
                vmin=-1, vmax=1, annot=True, cmap='coolwarm')
    ax.set_title('Correlation between SensorPack2')
    plt.show()
    save_figure(fig, 'Step0_1_1_CorrelationBetweenFeatures_Data2')

    # Y otra
    df_sens = get_sensors_list([8, 10, 12, 14])
    sensors_features = df_sens.drop(['Batch ID', 'GAS', 'CONCENTRATION'], axis=1)
    fig = plt.figure(figsize=(20, 20));
    ax = sns.heatmap(sensors_features.corr(),
                     vmin=-1, vmax=1, annot=True, cmap='coolwarm')
    ax.set_title('Correlation between SensorPack3')
    plt.show()
    save_figure(fig, 'Step0_1_1_CorrelationBetweenFeatures_Data3')

    # Y otra
    df_sens = get_sensors_list([9, 11, 13, 15])
    sensors_features = df_sens.drop(['Batch ID', 'GAS', 'CONCENTRATION'], axis=1)
    fig = plt.figure(figsize=(20, 20));
    ax = sns.heatmap(sensors_features.corr(),
                     vmin=-1, vmax=1, annot=True, cmap='coolwarm')
    ax.set_title('Correlation between SensorPack4')
    plt.show()
    save_figure(fig, 'Step0_1_1_CorrelationBetweenFeatures_Data4')

    # Aun existe correlacion entre las mediciones de los sensores, pese a ser de diferentes tipos,
    # pero hemos reducido en gran medida las variables relacionadas.
