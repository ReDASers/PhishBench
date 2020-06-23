import os.path as path

import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from ..core import BaseClassifier


class SVM(BaseClassifier):

    def __init__(self, io_dir):
        super().__init__(io_dir)
        self.clf = None
        self.model_path: str = path.join(self.io_dir, "model_svm.pkl")

    def load_model(self):
        self.clf = joblib.load(self.model_path)

    def save_model(self):
        if self.clf is not None:
            joblib.dump(self.clf, self.model_path)

    def fit(self, x, y):
        self.clf = SVC(probability=True)
        self.clf.fit(x, y)

    def fit_weighted(self, x, y):
        self.clf = SVC(probability=True, class_weight='balanced')
        self.clf.fit(x, y)

    def predict(self, x):
        assert self.clf is not None, "Classifier must be trained first"
        return self.clf.predict(x)

    def predict_proba(self, x):
        assert self.clf is not None, "Classifier must be trained first"
        return self.clf.predict_proba(x)[:, 1]

    def param_search(self, x, y):
        param_distributions = [
            {
                'kernel': ['linear'],
                'C': [0.1, 1, 10, 100]
            },
            {
                'kernel': ['poly'],
                'C': [0.1, 1, 10, 100],
                'gamma': [1, 0.1, 0.01, 0.001],
                'degree': [2, 3, 4, 5, 6]
            },
            {
                'kernel': ['rbf', 'sigmoid'],
                'C': [0.1, 1, 10, 100],
                'gamma': [1, 0.1, 0.01, 0.001]
            }
        ]
        base = SVC(probability=True)
        clf = RandomizedSearchCV(base, param_distributions, n_iter=20, n_jobs=-1, pre_dispatch='n_jobs')
        self.clf = clf.fit(x, y).best_estimator_
        return self.clf.get_params()
