import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib

class RandomForestMod:
    """ RandomForestClasificator."""

    FOLDER = 'models/randomforest'

    def __init__(self):
        pass

    def save_model(self, name):
        joblib.dump(self.model, os.path.join(self.FOLDER, name))

    def load_model(self, name):
        self.model = joblib.load(os.path.join(self.FOLDER, name))

    def _gen_model_base(self):
        model = RandomForestClassifier(max_depth=2, random_state=0)
        return model

    def _gen_model(self):
        model = RandomForestClassifier(max_depth=80,
                                       max_features=3,
                                       min_samples_leaf=3,
                                       min_samples_split=12,
                                       n_estimators=200,
                                       random_state=0)
        return model

    def model_search(self, X_train, y_train):
        model = self._gen_model_base()

        param_grid = {
            'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]}
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        grid_search.best_params_
        self.best_grid = grid_search.best_estimator_
        self.model = model

    def model_train(self, X_train, y_train):
        model = self._gen_model()
        model.fit(X_train, y_train)
        self.model = model
        return model

    def evaluate(self, test_features, test_labels):
        predictions = self.model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))

        return accuracy

    def model_evaluate(self, X_test, y_test):
        grid_accuracy = self.evaluate(self.best_grid,  X_test, y_test)
        return grid_accuracy

    def get_model(self):
        return self.model


if __name__ == '__main__':
    rf = RandomForestMod()