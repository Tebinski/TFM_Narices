"""This file will explore the diferences between gas data with several plots."""


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from python.LoadUciData import load_data
from python.StandardFigure import save_figure

from sklearn.preprocessing import StandardScaler

def plot_what(df_gas):

    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()
    for gas, ax in zip(range(1, 7, 1), axes):

        # Selecciono el gas
        df_gas_1 = df_gas[df_gas['GAS'] == gas]
        df_signal = df_gas_1.drop(columns=['GAS', 'Batch ID'])

        # Agrupo las señales por concentracion y calculo su media
        df_in = df_signal.groupby(by='CONCENTRATION').mean()

        # Estandarizo cada concentracion
        for col_name, col_data in df_in.iteritems():
            df_in[col_name] = (col_data - col_data.mean()) / col_data.std()

        # Represento el grafico
        ax = df_in.plot(figsize=(20, 10), style='.-', ax=ax)
        ax.legend().remove()
        ax.title.set_text('GAS ' + str(gas))
    fig.tight_layout()
    plt.show()
    save_figure(fig, 'Comparacion entre gases')



def get_sensors_array(df, gas, nrows):
    N_SENSORS = 16
    N_COMPONENTS = 8

    # Selecciono el gas
    df_gas = df[df['GAS'] == gas]
    # Me quedo con las señales y la concentracion
    df_signal = df_gas.drop(columns=['GAS', 'Batch ID'])

    df_measures = df_signal.iloc[nrows, 0:128]
    # Pasamos a forma tabla, donde veremos las 8 componentes para cada sensor
    arr_sensors = np.reshape(np.ravel(df_measures), (N_SENSORS, N_COMPONENTS))
    return arr_sensors

def plt_sensors(df, gas, nrows, norm=True):
    arr_sensors = get_sensors_array(df, gas, nrows)
    if norm:
        sca = StandardScaler()
        arr_sensors = sca.fit_transform(arr_sensors)
    df_sensors = pd.DataFrame(arr_sensors)
    ax = df_sensors.T.plot()
    return ax

def plt_sensors_fig(df, gas, nrows):
    arr_sensors = get_sensors_array(df, gas, nrows)
    sca = StandardScaler()
    arr_sensors_sca = sca.fit_transform(arr_sensors)
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(arr_sensors.T)
    axes[1].plot(arr_sensors_sca.T)
    plt.show()


def plt_sensors_fig_2(df, gas, nrows):

    dict_sensor_types = {'type1': [0, 1, 8, 9],
                         'type2': [2, 3, 6, 7, 10, 11, 14, 15],
                         'type3': [4, 5, 12, 13]}

    dict_sensor_colors = {'type1': 'r',
                          'type2': 'b',
                          'type3': 'k'}

    dict_label = {0 : ['b', 'type1'],
                  1 : ['b', 'type1'],
                  2 : ['r', 'type2'],
                  3 : ['r', 'type2'],
                  4 : ['k', 'type3'],
                  5 : ['k', 'type3'],
                  6 : ['r', 'type2'],
                  7 : ['r', 'type2'],
                  8 : ['b', 'type1'],
                  9 : ['b', 'type1'],
                  10: ['r', 'type2'],
                  11: ['r', 'type2'],
                  12: ['k', 'type3'],
                  13: ['k', 'type3'],
                  14: ['r', 'type2'],
                  15: ['r', 'type2']}

    arr_sensors = get_sensors_array(df, gas, nrows)
    fig = plt.figure()
    for sensor in range(0, 16):
        arrplot = arr_sensors[sensor, :].T
        color, label = dict_label[sensor]
        plt.plot(arrplot, color=color, label=label)
    plt.legend()
    plt.show()

    sca = StandardScaler()
    arr_sensors_sca = sca.fit_transform(arr_sensors)
    fig = plt.figure()
    for sensor in range(0, 16):
        arrplot = arr_sensors_sca[sensor, :].T
        color, label = dict_label[sensor]
        plt.plot(arrplot, color=color, label=label)
    plt.legend()
    plt.show()



if __name__ == '__main__':

    # Cargo datos
    df = load_data()

    # grafico para visualizar señales
    # plot_what(df)

    # There are 16 sensors, which signals have been refactorized in 8 components = 128 components in total
    # Let's check if there is any differences between sensors.
    #plt_sensors_fig(df, 1, 0)
    plt_sensors_fig_2(df, 1, 0)