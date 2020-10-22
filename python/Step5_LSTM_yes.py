import seaborn as sns
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np

from python.LoadUciData import load_data
from python.LoadSensorData import sensor_features_column, sensor_type_dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

if __name__ == '__main__':


    model = Sequential()
    features = 128
    time_step=1
    samples = 50
    model.add(LSTM(32, input_shape=(samples, features)))
    model.add(Dense(6))

    # prepare data for LSTM
    # LSTM needs 3d data
    df = load_data()
    X = df.iloc[:, :128]
    y = df['GAS']
    