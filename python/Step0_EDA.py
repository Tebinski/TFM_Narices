"""
Exploratory Data Analysis.
Para explorar el dataset de gases, vamos a represntar
    - Cuantos gases hay en cada batch
    - calcular tabla conteo de:
        - batch vs gases
        - Cuantas muestras hay para cada gas.
"""

import pandas as pd
import matplotlib.pyplot as plt

from python.LoadUciData import LoadDatFolder
from python.StandardFigure import save_figure
pd.set_option('display.max_columns', 10)


def plot_count_per_batch_and_gas(df_gas):
    props = df_gas.groupby("Batch ID")['GAS'].value_counts(normalize=False).unstack()
    ax = props.plot(kind='bar', stacked='True', figsize=(24, 16))
    ax.set_ylabel('Number of samples')
    save_figure(plt.gcf(), 'Step0_Count_Batch_Gas')


def plot_sample_count_per_gas(df_gas):
    props = df_gas.groupby('GAS')['GAS'].value_counts(normalize=False).unstack()
    ax = props.plot(kind='bar', stacked='True', figsize=(24, 16))
    ax.set_ylabel('Number of samples of each gas')
    save_figure(plt.gcf(), 'Step0_Count_Gas')


if __name__ == '__main__':

    # Load .dat files
    folder = r'data_uci/driftdataset'
    df_gas = LoadDatFolder(folder).df

    ## Tables
    # Show samples per Batch and Gas
    gas_batch_group = pd.crosstab(df_gas['Batch ID'], df_gas['GAS']).sort_index()
    print('\n', gas_batch_group.to_markdown())

    # Show concentration range statistics per GAS
    pivot = pd.pivot_table(df_gas,
                           index=['GAS'],
                           values='CONCENTRATION',
                           aggfunc=['min', 'max', 'mean', 'std', 'count'])
    pivot.round(2)
    print('\n', pivot.round(2).to_markdown())

    ## Plots
    # Show samples count per GAS
    plot_count_per_batch_and_gas(df_gas)
    plot_sample_count_per_gas(df_gas)

