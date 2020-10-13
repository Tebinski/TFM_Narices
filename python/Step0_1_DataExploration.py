"""
This file will explore the differences between gas data with several plots.
- PLot A.  REpresenta esto
- Plot B. Representa aquello

Un vistazo al dataframe normalizado reflaja claramente cómo las mediciones de los 16 sensores han sido organizadas
de la siguiente forma.
1 1 2 2 3 3 4 4 - Y se repite una vez.

"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from python.LoadUciData import load_data
from python.StandardFigure import save_figure



def get_sensors_array(df, gas, selected_row):
    """
    Divido la fila 'selected_row' del dataframe
    (131 columns = 128 columns de features + Batch ID + Concentration + Gas)
    en una tabla 16*8 donde cada fila es un sensor y cada columna una componente.
    """
    N_SENSORS = 16
    N_COMPONENTS = 8

    # Selecciono el gas
    df_gas = df[df['GAS'] == gas]
    # Me quedo con las señales y la concentracion
    df_signal = df_gas.drop(columns=['GAS', 'Batch ID'])

    df_measures = df_signal.iloc[selected_row, 0:128]
    # Pasamos a forma tabla, donde veremos las 8 componentes para cada sensor
    arr_sensors = np.reshape(np.ravel(df_measures), (N_SENSORS, N_COMPONENTS))
    return arr_sensors


def plot_gas_feature_comparison(df_gas):
    """
    - Genero un plot para cada gas (& graficas)
        - Para las muestras para mismo gas y concentracion, calculo la media de los valores de las features
        - Estandarizo cada columna
        - Represento
    """
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


def plt_sensors(df, gas, nrows, norm=True):
    """
    - Dado el array definido en la func get_sensors_array,
        - Represento las mediciones de cada sensor para ese gas a esa concentracion.
    """

    arr_sensors = get_sensors_array(df, gas, nrows)
    if norm:
        sca = StandardScaler()
        arr_sensors = sca.fit_transform(arr_sensors)
    df_sensors = pd.DataFrame(arr_sensors)
    ax = df_sensors.T.plot()
    return ax


def plt_sensors_fig(df, gas, nrows):
    """
    Dado el array definido en la func get_sensors_array,
        - Represento las mediciones de cada sensor para ese gas a esa concentracion.
    """
    arr_sensors = get_sensors_array(df, gas, nrows)
    sca = StandardScaler()
    arr_sensors_sca = sca.fit_transform(arr_sensors)
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(arr_sensors.T)
    axes[1].plot(arr_sensors_sca.T)
    plt.show()
    return fig


def plt_sensors_fig_2(df, gas, nrows):

    dict_sensor_types = {'type1': [0, 1, 8, 9],
                         'type2': [2, 3, 10, 11],
                         'type3': [4, 5, 12, 13],
                         'type4': [6, 7, 14, 15]}

    dict_sensor_colors = {'type1': 'r',
                          'type2': 'b',
                          'type3': 'k',
                          'type4': 'g'}

    dict_label = {0:  ['b', 'type1'],
                  1:  ['b', 'type1'],
                  2:  ['r', 'type2'],
                  3:  ['r', 'type2'],
                  4:  ['k', 'type3'],
                  5:  ['k', 'type3'],
                  6:  ['g', 'type4'],
                  7:  ['g', 'type4'],
                  8:  ['b', 'type1'],
                  9:  ['b', 'type1'],
                  10: ['r', 'type2'],
                  11: ['r', 'type2'],
                  12: ['k', 'type3'],
                  13: ['k', 'type3'],
                  14: ['g', 'type4'],
                  15: ['g', 'type4']}

    arr_sensors = get_sensors_array(df, gas, nrows)
    fig, axes = plt.subplots(1, 2, figsize=(16,8))
    for sensor in range(0, 16):
        arrplot = arr_sensors[sensor, :].T
        color, label = dict_label[sensor]
        axes[0].plot(arrplot, color=color, label=label)
    axes[0].set_title('Sensor features')

    sca = StandardScaler()
    arr_sensors_sca = sca.fit_transform(arr_sensors)
    for sensor in range(0, 16):
        arrplot = arr_sensors_sca[sensor, :].T
        color, label = dict_label[sensor]
        axes[1].plot(arrplot, color=color, label=label)
    axes[1].set_title('Sensor features normalized')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    return fig


if __name__ == '__main__':

    # Cargo datos
    df = load_data()

    # grafico para visualizar señales
    plot_gas_feature_comparison(df)

    # There are 16 sensors, which signals have been refactorized in 8 components = 128 components in total
    # Let's check if there is any differences between sensors.
    fig = plt_sensors_fig(df, 1, 0)
    save_figure(fig, 'Step0_1_SensorFeaturesNoColor')
    fig = plt_sensors_fig_2(df, 1, 0)
    save_figure(fig, 'Step0_1_SensorFeaturesColor')