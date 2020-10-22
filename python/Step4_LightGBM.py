import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np

from python.LoadUciData import load_data
from python.LoadSensorData import sensor_features_column, sensor_type_dict
from python.StandardFigure import save_figure

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def plot_conf(conf):
    fig = plt.figure(figsize=(8, 8))
    ax = sns.heatmap(conf, annot=True, fmt=".02f", cmap='Blues')
    plt.yticks(rotation=0)
    plt.show()
    return fig


def train_lgbm(X_train, y_train):
    lgbm = lgb.LGBMClassifier(objective='multiclass', random_state=5)
    lgbm.fit(X_train, y_train)
    return lgbm


def lgbm_conf_shap(X_train, X_test, y_train, y_test):
    lgbm = train_lgbm(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    conf = confusion_matrix(y_test, y_pred)
    conf_percent = conf.T/y_test.value_counts().sort_index().values
    plot_conf(conf_percent)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    print('Start shapely')
    model = lgbm
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    fig = plt.figure()
    plt.title("SHAP VALUES")
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    return fig


if __name__ == '__main__':

    df = load_data()

    # Keep only one pair of Gas and Concentration
    """
    | GAS   | CONCENTRATION   | count |
    |------:|----------------:|------:|
    | 1     | 50              | 488   |
    | 2     | 50              | 588   |
    | 3     | 200             | 177   |
    | 4     | 100             | 357   |
    | 5     | 50              | 485   |
    | 6     | 50              | 345   |
    """
    # Get count of each pair value
    df_c = df[['GAS', 'CONCENTRATION']].value_counts()
    df_c1 = df_c.reset_index()
    df_c1 = df_c1.rename(columns={0: 'count'})
    idx = df_c1.groupby(['GAS'])['count'].transform(max) == df_c1['count']
    result = df_c1[idx].sort_values(by='GAS')
    print(result)

    # Total of measurements we are using is 2440.

    dict_gas_concentration = {1: 50,
                              2: 50,
                              3: 200,
                              4: 100,
                              5: 50,
                              6: 50}

    df_temp = pd.DataFrame()
    for g, c in dict_gas_concentration.items():
        df_select = df[(df['GAS'] == g) & (df['CONCENTRATION'] == c)]
        df_temp = df_temp.append(df_select)

    # Select only one sensor
    X = df_temp.iloc[:, :8]
    y = df_temp['GAS']


    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
    fig = lgbm_conf_shap(X_train, X_test, y_train, y_test)
    save_figure(fig, 'Step4_LGBM_one_sensor')

    ### Now let's use all sensor data
    # Select all sensors
    X = df_temp.iloc[:, :128]
    y = df_temp['GAS']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
    fig = lgbm_conf_shap(X_train, X_test, y_train, y_test)

    ### Okay, now let's train with batch 1 to 3, and predict batch 4
    df_temp_train = df_temp[df_temp['Batch ID'].isin([1,2,3,4,5,6,7,8])]
    df_temp_test = df_temp[df_temp['Batch ID'].isin([9])]

    X_train = df_temp_train.iloc[:, :128]
    y_train = df_temp_train['GAS']

    X_test = df_temp_test.iloc[:, :128]
    y_test = df_temp_test['GAS']

    lgbm_conf_shap(X_train, X_test, y_train, y_test)
    save_figure(fig, 'Step4_LGBM_all data')

    # Now we get that the drift has changed the measurement of gas 3 and 6,
    # and the model get totally confuse. It still precise, but not accurate.

    # maybe is because, as shown the shapely plot, it decision is based only in a few features
    X = df_temp.iloc[:, :128]
    y = df_temp['GAS']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)
    lgbm_conf_shap(X_train, X_test, y_train, y_test)

    ### Okay, now let's train with batch 1 to 5, and predict batch 6
    df_temp_train = df_temp[df_temp['Batch ID'].isin([1, 2, 3, 4, 5])]
    df_temp_test = df_temp[df_temp['Batch ID'].isin([6])]

    X_train = df_temp_train.iloc[:, :128]
    y_train = df_temp_train['GAS']

    X_test = df_temp_test.iloc[:, :128]
    y_test = df_temp_test['GAS']

    lgbm_conf_shap(X_train, X_test, y_train, y_test)

    # maybe if we train 4 models, one for each type of sensor, and get a votation
    range_cols_typeA = [sensor_features_column(n) for n in sensor_type_dict['A']]
    idx_cols = np.r_[range_cols_typeA].ravel()

    df_sA = df_temp.iloc[:, idx_cols]
    df_feat = df_temp[['GAS', 'CONCENTRATION', 'Batch ID']]
    dfA_full = pd.concat([df_sA, df_feat], axis=1)

    df_A_train = dfA_full[dfA_full['Batch ID'].isin([1, 2, 3, 4, 5,6,7,8])]
    df_A_test =  dfA_full[dfA_full['Batch ID'].isin([9])]

    X_train = df_A_train.drop(columns= ['GAS', 'CONCENTRATION', 'Batch ID'])
    X_test  = df_A_test.drop(columns= ['GAS', 'CONCENTRATION', 'Batch ID'])

    y_train = df_A_train['GAS']
    y_test  = df_A_test['GAS']

    lgbmA = train_lgbm(X_train, y_train)
    y_pred = lgbmA.predict(X_test)
    lgbmA.score(X_test, y_test)
    confA = confusion_matrix(y_test, y_pred)

    # tendria que elegir las concetracion por rangos.
