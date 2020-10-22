""" UsarShapely for random forest"""

import shap
import pandas as pd

from python.LoadUciData import load_data
from python.StandardFigure import save_figure
from python.LoadSensorData import get_sensors_list

from sklearn.model_selection import train_test_split

from python.SeqModel import SeqModel, SeqModelSimple
from python.RandomForestClasification import RandomForestMod


if __name__ == '__main__':

    # load your data here, e.g. X and y
    # create and fit your model here
    df = load_data()

    X = df.iloc[:, :16]
    y = df['GAS']
    y = pd.get_dummies(y, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=0.33,
                                                       random_state=42)

    rfm = RandomForestMod()
    #rfm.model_train(X_train, y_train)
    #rfm.save_model('RF_feature')
    rfm.load_model('RF_feature')

    rf = rfm.get_model()
    y_pred = rf.predict(X_test)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    print('Start shapely')
    model = rf
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
#
    ## visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    #shap.force_plot(explainer.expected_value[0], shap_values[0])
    shap.summary_plot(shap_values, X, plot_type="bar")

