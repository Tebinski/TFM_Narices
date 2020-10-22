import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np

from python.LoadUciData import load_data
from python.LoadSensorData import sensor_features_column, sensor_type_dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':

    df = load_data()

    X = df.iloc[:, :128]
    y = df['GAS']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=0)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)

    ## Batch9

    df_train = df[df['Batch ID'].isin(range(0, 9))]
    df_test  = df[df['Batch ID'] == 9]

    X_train = df_train.iloc[:, :128]
    y_train = df_train['GAS']

    X_test = df_test.iloc[:, :128]
    y_test = df_test['GAS']

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)

