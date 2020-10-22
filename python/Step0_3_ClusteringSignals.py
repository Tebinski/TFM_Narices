"""
Use PCA, Kmeans and TSNE to reduce the dimensionality and classify gases.
"""

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from python.LoadUciData import load_data, calculate_bins_concentration
from python.StandardFigure import save_figure
from python.Step0_1_DataExploration import get_sensors_array


def apply_KMeans_2d(X,y, name):
    pca = PCA(n_components=2)
    pca.fit(X, y)
    xp = pca.transform(X)

    number_of_clusters = 6
    km = KMeans(n_clusters=number_of_clusters)
    # Normally people fit the matrix
    y_pred = km.fit_predict(X)
    #igualamos las categorias a los indices 1 al 6 del dataframe
    y_pred = y_pred + 1

    #Plots
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Color for each cluster')
    scatter = ax.scatter(xp[:, 0], xp[:, 1], c=y_pred)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Clusters")
    plt.show()
    save_figure(fig, f'Step0_3_Color for each cluster_2d_{name}')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Color for each gas')
    scatter = ax.scatter(xp[:, 0], xp[:, 1], c=y)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Gas")
    plt.show()
    save_figure(fig, f'Step0_3_Color for each gas_{name}')

    fig = plt.figure(3, figsize=(8, 6))
    conf = confusion_matrix(y, y_pred)
    sns.heatmap(conf, annot=True, fmt='d')
    plt.title(f"Confusion matrix_{name}")
    plt.show()


def apply_KMeans_3d(X,y, name):
    pca = PCA(n_components=3)
    pca.fit(X, y)
    xp = pca.transform(X)

    number_of_clusters = 6
    km = KMeans(n_clusters=number_of_clusters)
    # Normally people fit the matrix
    y_pred = km.fit_predict(X)
    #igualamos las categorias a los indices 1 al 6 del dataframe
    y_pred = y_pred + 1

    fig = plt.figure(figsize=(8,6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    scatter = ax.scatter(xp[:, 0], xp[:, 1], xp[:, 2], c=y_pred)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    plt.show()
    save_figure(fig, f'Step0_3_Color for each cluster_3d_{name}')

    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    scatter = ax.scatter(xp[:, 0], xp[:, 1], xp[:, 2], c=y)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Gas")
    plt.show()
    save_figure(fig, f'Step0_3_Color for each gas_3d_{name}')

    fig = plt.figure(3, figsize=(8, 6))
    conf = confusion_matrix(y, y_pred)
    sns.heatmap(conf, annot=True, fmt='d')
    plt.title(f"Confusion matrix_{name}")
    plt.show()


def apply_tsne(X, y, name):
    print('tsne2d')
    X_embedded = TSNE(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('TSNE 2d Batch1, Sensor1, Concentration less 100ppmv')
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, label=y.unique())
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Gas")
    plt.show()
    save_figure(fig, f'Step0_3_TSNE_2d_{name}')

    print('tsne3d')
    X_embedded = TSNE(n_components=3).fit_transform(X)
    fig = plt.figure(figsize=(8, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y)
    ax.set_title('TSNE 3d Batch1, Sensor1, Concentration less 100ppmv')
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Gas")
    plt.show()
    save_figure(fig, f'Step0_3_TSNE_3d_{name}')


if __name__ == '__main__':
    # Load data
    df = load_data()

    #keep only values of concentration less or equal than 100
    df = df[df['CONCENTRATION'] <= 100]

    #Keep only fist batch
    df = df[df['Batch ID'] == 1]

    # keep only data from sensor1
    X = df.iloc[:, :8]
    y = df['GAS']

    apply_KMeans_2d(X, y, 'Batch1_Sensor1_Conc less 100ppmv')
    apply_KMeans_3d(X, y, 'Batch1_Sensor1_Conc less 100ppmv')
    apply_tsne(X, y, 'Batch1_Sensor1_Conc less 100ppmv')


    ## Now lets keep batch 1 and 10
    # Load data
    df = load_data()
    df = df[df['CONCENTRATION'] <= 100]
    df = df[df['Batch ID'].isin([1,10])]

    # keep only data from sensor1
    X = df.iloc[:, :8]
    y = df['GAS']

    apply_KMeans_2d(X, y, 'Batch1and10_Sensor1_Conc less 100ppmv')
    apply_KMeans_3d(X, y, 'Batch1and10_Sensor1_Conc less 100ppmv')
    apply_tsne(X, y, 'Batch1_Sensor1_Conc less 100ppmv')

    ### Now all the data available
    # Load data
    df = load_data()

    # keep only data from sensor1
    X = df.iloc[:, :128]
    y = df['GAS']

    apply_KMeans_2d(X, y, 'All data')
    apply_KMeans_3d(X, y, 'All data')
    apply_tsne(X, y, 'All data')
