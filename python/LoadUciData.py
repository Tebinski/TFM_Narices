import os
import pandas as pd
import re
import numpy as np
from python.FileUtils import get_list_of_files_with_extension
from sklearn.preprocessing import StandardScaler

class LoadDatFile:
    """
    This class aims to load a .dat files from UCI
    https://archive.ics.uci.edu/ml//datasets/Gas+Sensor+Array+Drift+Dataset
    , and returns a pandas.dataframe object

    :arg .dat file
    :return df
    """

    def __init__(self, file):
        self.file = file

    @property
    def batch_number(self):
        base = os.path.basename(self.file)
        name, ext = os.path.splitext(base)
        num = re.findall(r'\d+', name)[0]
        # num = num.zfill(2)
        return int(num)

    @property
    def df(self):
        df = pd.read_table(self.file, engine='python', sep='\s+\d+:', header=None)
        df['Batch ID'] = self.batch_number
        return df


class GasDataFrame:
    """ Process the .dat file to get all the information contained:
    - Gas, concentration and measures."""

    def __init__(self, file):
        self.file = file

    @property
    def df(self):
        df_raw = LoadDatFile(self.file).df
        return self._add_gas_info(df_raw)

    @staticmethod
    def _add_gas_info(df):
        df[['GAS', 'CONCENTRATION']] = df.iloc[:, 0].str.split(";", expand=True, )
        df.drop(df.columns[0], axis=1, inplace=True)
        df['GAS'] = df['GAS'].astype('int')
        df['CONCENTRATION'] = df['CONCENTRATION'].astype('float')
        return df


class LoadDatFolder:
    """
    This class aims to load all .dat files contained in a folder,
    gives each file a GasDataframe format and concats all in a pandas.dataframe object with

    :inputs: folder with many .dat files
    :return df
    """
    def __init__(self, folder):
        self.folder = folder

    @property
    def df(self):
        files = get_list_of_files_with_extension(self.folder, 'dat')
        df_full = pd.DataFrame()
        for f in files:
            dftemp = GasDataFrame(f).df
            df_full = df_full.append(dftemp)
        return df_full


def load_data():
    folder = r'data_uci/driftdataset'
    df_gas = LoadDatFolder(folder).df

    #Rename sensor columns
    col_names_dict = {}
    i = 1
    for sensor in range(0, 15 + 1):
        for feature in range(0, 7 + 1):
            col_names_dict[i] = f'S{sensor}_{feature}'
            i = i + 1

    df_gas = df_gas.rename(columns=col_names_dict)
    return df_gas


def load_data_scaled():
    # Load data
    df = load_data()

    # init scaler
    sc = StandardScaler()

    # Scale only sensor data
    sensor_features = df.iloc[:, :128]
    sc.fit(sensor_features)
    data_sc = sc.transform(sensor_features)

    # Get the unscaled info
    info = df[['Batch ID', 'GAS', 'CONCENTRATION']].values

    # Merge scaled data and the info into a pandas dataframe.
    data = np.concatenate([data_sc, info], axis=1)
    df_sca_gas = pd.DataFrame(data, columns=df.columns)
    for col in ['GAS', 'Batch ID']:
        df_sca_gas[col] = df_sca_gas[col].astype('int').astype('category')
    return df_sca_gas

def calculate_bins_concentration(df):
    # Create ConcentrationCat column.
    df['ConcentrationCat'] = pd.cut(df['CONCENTRATION'], bins=range(0, 1000, 100))
    return df



if __name__ == '__main__':
    file_data = r'data_uci/driftdataset/batch1.dat'
    lf = LoadDatFile(file_data)
    my_dataframe = lf.df

    gdf = GasDataFrame(file_data)
    my_dataframe_gas = gdf.df

    folder = r'data_uci/driftdataset/'
    ldf = LoadDatFolder(folder)
    my_dataframe_full = ldf.df

    my_df_scaled = load_data_scaled()



