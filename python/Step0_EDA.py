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

from python.LoadUciData import load_data, load_data_scaled, calculate_bins_concentration
from python.StandardFigure import save_figure
pd.set_option('display.max_columns', 10)


def plot_count_per_batch_and_gas(df_gas):
    props = df_gas.groupby("Batch ID")['GAS'].value_counts(normalize=False).unstack()
    ax = props.plot(kind='bar', stacked='True', figsize=(8, 6))
    ax.set_ylabel('Number of samples')
    return plt.gcf()


def plot_sample_count_per_gas(df_gas):
    props = df_gas.groupby('GAS')['GAS'].value_counts(normalize=False).unstack()
    ax = props.plot(kind='bar', stacked='True', figsize=(8, 6))
    ax.set_ylabel('Number of samples of each gas')
    return plt.gcf()


def concentration_plot_count(ax, df_gas, gas=1):
    df = df_gas.copy()
    df_signal = df[df['GAS'] == gas]
    df_signal = df_signal.drop(columns=['GAS', 'Batch ID'])
    df_in = df_signal.groupby(by='CONCENTRATION')

    #ax = df_in.count().plot(style='.-', ax=ax, color='b')
    ax = df_in.count().plot.bar(ax=ax, color='blue')
    ax.get_legend().remove()
    ax.title.set_text('GAS ' + str(gas))


if __name__ == '__main__':

    # Load .dat files
    df_gas = load_data()

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

    # Calculate which concentration if more common for each gas
    df_c = df_gas[['GAS', 'CONCENTRATION']].value_counts()
    df_c1 = df_c.reset_index()
    df_c1 = df_c1.rename(columns={0: 'count'})
    idx = df_c1.groupby(['GAS'])['count'].transform(max) == df_c1['count']
    result = df_c1[idx].sort_values(by='GAS')
    print(result)

    ## Plots

    # Show samples count per GAS
    fig = plot_count_per_batch_and_gas(df_gas)
    save_figure(fig, 'Step0_Count_Batch_Gas')
    plt.show()

    fig = plot_sample_count_per_gas(df_gas)
    save_figure(fig, 'Step0_Count_Gas')
    plt.show()

    ## Concentration plot (takes time to plot)
    fig, axes = plt.subplots(3, 2, figsize=(25, 20))
    fig.suptitle('Count of measurements of each Gas and concentration')
    for i, ax in enumerate(axes.flatten(), start=1):
        print(i)
        concentration_plot_count(ax, df_gas, gas=i)
    plt.tight_layout()
    plt.show()
    save_figure(fig, 'Step0_Concentration Distribution per gas')

    # Group concentration
    df_gas['ConcentrationCat'] = pd.cut(df_gas['CONCENTRATION'], bins=range(0, 1000, 100))

    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    fig.suptitle('Count of measurements of each Gas and concentration')
    for gas, ax in enumerate(axes.flatten(), start=1):
        print(gas)
        df = df_gas.copy()
        df_signal = df[df['GAS'] == gas]
        df_signal = df_signal.drop(columns=['GAS', 'Batch ID', 'CONCENTRATION'])
        df_in = df_signal.groupby(by='ConcentrationCat')

        #ax = df_in.count().plot(style='.-', ax=ax, color='b')
        ax = df_in.count().plot.bar(ax=ax, color='blue')
        ax.get_legend().remove()
        ax.title.set_text('GAS ' + str(gas))
    plt.show()
    save_figure(fig, 'Step0_Concentration Distribution per gas_binned')

    ## Para el gas 2, concentracion 50, plotear la señal de un sensor en los
    # diferentes batch
    # Escalamos las muestras
    df_sca_gas = load_data_scaled()
    df_sca_gas = calculate_bins_concentration(df_sca_gas)

    #Representamos las medias
    fig, axes = plt.subplots(3, 4, figsize=(20, 20))
    fig.subplots_adjust(top=0.8)
    #fig.suptitle('Evolution of Gas2-Concentration50 in every batch')

    for batch, ax in enumerate(axes.flatten(), start=1):
        if batch > 10:
            break
        print(batch)
        df = df_sca_gas.copy()
        df_signal = df[(df['GAS'] == 2) & (df['Batch ID'] == batch)]

        df_signal = df_signal.reset_index().drop(columns=['ConcentrationCat'])
        df_signal = df_signal.iloc[:,:8]

        # ax = df_in.count().plot(style='.-', ax=ax, color='b')
        ax = df_signal.mean().plot(ax=ax, color='blue')
        #ax.get_legend().remove()
        ax.title.set_text('Batch ' + str(batch))
        ax.set_ylim([-1, 1])
    fig.subplots_adjust(top=2)
    plt.tight_layout()
    plt.show()
    save_figure(fig, 'Step0_Evolution_of_signal_sensor1_mean')

    # Representamos las muestras
    fig, axes = plt.subplots(3, 4, figsize=(20, 20))
    #fig.suptitle('Evolution of Gas2-Concentration50 in every batch')
    fig.subplots_adjust(top=1.2)
    for batch, ax in enumerate(axes.flatten(), start=1):

        print(batch)
        df = df_sca_gas.copy()
        df_signal = df[(df['GAS'] == 2) & (df['Batch ID'] == batch)]
        df_signal = df_signal.set_index('ConcentrationCat').loc[50]
        df_signal = df_signal.reset_index().drop(columns=['ConcentrationCat'])
        df_signal = df_signal.iloc[:,:8]

        # ax = df_in.count().plot(style='.-', ax=ax, color='b')
        ax = df_signal.plot(ax=ax)
        ax.get_legend().remove()
        ax.title.set_text('Batch ' + str(batch))
        ax.set_ylim([-1,1])
        if batch==10:
            break
    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles,labels, bbox_to_anchor=(2, 0), loc='lower right')
    lgd.FontSize = 18;
    plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.35)

    plt.tight_layout()
    plt.show()
    save_figure(fig, 'Step0_Evolution_of_signal_sensor1')


