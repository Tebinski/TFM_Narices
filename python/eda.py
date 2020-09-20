import os
import pandas as pd

df = pd.read_table("./data_uci/driftdataset/batch1.dat",
                   engine='python',
                   sep ='\s+\d+:',
                   header=None)

df[['Gas','Seconds']] = df.loc[:,0].str.split(';', expand=True)

gasgroup = df.groupby('Gas')