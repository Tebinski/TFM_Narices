"""
Este codigo no lo voy a presentar, la idea era utilizar PCA y clustering para dividir
las diferentes señales, pero no es muy eficaz

podemos ver en la descomposicion PCA que las muesrtas para diferentes gases acaban solapadas.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from python.LoadUciData import load_data
from python.StandardFigure import save_figure
from python.Step0_1_DataExploration import get_sensors_array

if __name__ == '__main__':
    # Cargo datos
    df = load_data()
    # Me quedo con los datos de un unico sensor, y los valores estacionarios
    X = df.iloc[:, 2:8]
    y = df['GAS']

    pca = PCA(n_components=3)
    pca.fit(X, y)
    xp = pca.transform(X)

    number_of_clusters = 6
    km = KMeans(n_clusters=number_of_clusters)
    # Normally people fit the matrix
    y_pred = km.fit_predict(X)
    #igualamos las categorias a los indices 1 al 6 del dataframe
    y_pred = y_pred + 1

    fig = plt.figure(1, figsize=(5, 4))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(xp[:, 0], xp[:, 1], xp[:, 2], c=y_pred)
    plt.show()
    save_figure(fig, 'Step0_3_Color for each cluster')

    fig = plt.figure(2, figsize=(5, 4))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(xp[:, 0], xp[:, 1], xp[:, 2], c=y)
    plt.show()
    save_figure(fig, 'Step0_3_Color for each gas')

    fig = plt.figure(3, figsize=(8, 6))
    conf = confusion_matrix(y, y_pred)
    sns.heatmap(conf, annot=True)
    plt.title("Confusion matrix")
    plt.show()

    from sklearn.manifold import TSNE
    X_embedded = TSNE(n_components=3).fit_transform(X)
    fig = plt.figure(3, figsize=(8, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y)
    plt.show()
    save_figure(fig, 'Step0_3_TSNE_Color for each gas')

