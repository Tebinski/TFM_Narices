import pandas as pd
import numpy as np
from python.LoadUciData import load_data

def get_sensors(list_of_n):
    """
    Obtengo un dataframe con un sensor de Tipo I, II, III y IV + las columnas de Batch ID, GAS y concentration
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
    """Returns UCI gas data from sensor N. """
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


if __name__ == '__main__':

    df_sensor_7 = get_sensor(7)
    df = get_sensors([0, 2, 4, 6])

