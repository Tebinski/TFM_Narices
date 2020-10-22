import pandas as pd
import numpy as np
from python.LoadUciData import load_data

sensor_type_dict = {'A': [0, 1, 8, 9],
                    'B': [2, 3, 10, 11],
                    'C': [4, 5, 12, 13],
                    'D': [6, 7, 14, 1]}

def get_sensor_by_type(letter):
    """
    Select the sensors as type, A,B,C or D
    and returns the  sensors features (N*8), Batch ID, GAS and concentration
    :arg list of integers
    :return pandas dataframe
    """
    return get_sensors_list(sensor_type_dict[f'{letter}'])


def get_sensors_list(list_of_n):
    """
     Get the N sensor features (N*8), Batch ID, GAS and concentration,
    by index
    :arg list of integers
    :return pandas dataframe
    """

    df = load_data()
    sensors_columns = []
    for n in list_of_n:
        ix = list(sensor_features_column(n))
        sensors_columns.extend(ix)
    #sensors_columns = np.r_[1:10, 15, 17, 50:100]
    df_features_n = df.iloc[:, sensors_columns]

    df_gas = df[['Batch ID', 'GAS', 'CONCENTRATION']]
    return pd.concat([df_features_n, df_gas], axis=1)


def get_sensor(n):
    """ Get the sensor 8 features, Batch ID, GAS and concentration,
    by index
    """
    df = load_data()
    df_features = get_sensor_features(df, n)
    df_gas = df[['Batch ID', 'GAS', 'CONCENTRATION']]
    return pd.concat([df_features, df_gas], axis=1)


def get_sensor_features(df, n):
    return df.iloc[:, sensor_features_column(n)]


def sensor_features_column(n):
    if 0 <= n <= 15:
        return range(8*n, 8*n + 8)
    else:
        return None


def get_sensor_by_col_name(n):
    """
    Get the sensor 8 features, Batch ID, GAS and concentration,
    by column names
    """
    df = load_data()
    filter_col = [col for col in df if col.startswith(f'S{n}')]
    df_sensor = df[filter_col]
    df_gas = df[['Batch ID', 'GAS', 'CONCENTRATION']]
    return pd.concat([df_sensor, df_gas], axis=1)


if __name__ == '__main__':

    df_sensor_7 = get_sensor(7)
    df = get_sensors_list([0, 2, 4, 6])

    #Comparamos que ambos metodos son equivalentes
    df_index = get_sensor(2)
    df_name = get_sensor_by_col_name(2)
    print(df_index.equals(df_name))
