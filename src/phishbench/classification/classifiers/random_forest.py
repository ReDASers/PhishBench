from os import path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from ..core import BaseClassifier


class RandomForest(BaseClassifier):
    def __init__(self, io_dir):
        super().__init__(io_dir)
        self.clf = None
        self.model_path: str = path.join(self.io_dir, "model_rf.pkl")

    def load_model(self):
        self.clf = joblib.load(self.model_path)

    def save_model(self):
        if self.clf is not None:
            joblib.dump(self.clf, self.model_path)

    def fit(self, x, y):
        self.clf = RandomForestClassifier()
        self.clf.fit(x, y)

    def fit_weighted(self, x, y):
        self.clf = RandomForestClassifier(class_weight='balanced_subsample')
        self.clf.fit(x, y)

    def predict(self, x):
        assert self.clf is not None, 'Classifier must be trained first'
        return self.clf.predict(x)

    def predict_proba(self, x):
        assert self.clf is not None, 'Classifier must be trained first'
        return self.clf.predict_proba(x)[:, 1]

    def param_search(self, x, y):
        param_distributions = {
                'n_estimators': list(range(10, 100, 10)),
                'max_depth': list(range(10, 100, 10)),
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ['auto', 'sqrt'],
                "bootstrap": [True, False]
            }
        base_clf = RandomForestClassifier()
        clf = RandomizedSearchCV(base_clf, param_distributions, n_iter=100, n_jobs=-1, pre_dispatch='2*n_jobs')

        self.clf = clf.fit(x, y).best_estimator_
        return self.clf.get_params()
