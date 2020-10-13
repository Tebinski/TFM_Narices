"""
Este codigo no lo voy a presentar, la idea era detectar los 4 tipos de sensores

Si obtienes la matriz de Sensoresx features para cada medicion, se ve claramete cómo la primera
componente ΔR marca la diferencia entre los sensores, pero su division no es tan clara y no la he encontrado
en la bibliografia.
"""


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from python.LoadUciData import load_data
from python.StandardFigure import save_figure
from python.Step0_1_DataExploration import get_sensors_array


if __name__ == '__main__':

    # Cargo datos
    df = load_data()
    arr_sensor = get_sensors_array(df, 1, 0)

    pca = PCA(n_components=2)
    pca.fit(arr_sensor)
    arr_transf = pca.transform(arr_sensor)

    number_of_clusters = 4
    km = KMeans(n_clusters=number_of_clusters)
    # Normally people fit the matrix
    y_pred = km.fit_predict(arr_transf)

    plt.scatter(arr_transf[:, 0], arr_transf[:, 1], c=y_pred)
    plt.title("Cluster of sensors")
    plt.show()


