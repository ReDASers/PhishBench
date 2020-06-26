from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from ..core import BaseClassifier


class Bagging(BaseClassifier):
    def __init__(self, io_dir):
        super().__init__(io_dir, "model_bag.pkl")

    def fit(self, x, y):
        self.clf = BaggingClassifier()
        self.clf.fit(x, y)

    def fit_weighted(self, x, y):
        base_classifier = DecisionTreeClassifier(class_weight='balanced')
        self.clf = BaggingClassifier(base_estimator=base_classifier)
        self.clf.fit(x, y)

    def param_search(self, x, y):
        param_grid = {
            "n_estimators": range(0, 110, 10),
            "max_features": [1, 10, 50, 100],
            "max_samples": [0.25, 0.5, 0.75, 1.0],
            "bootstrap": [True, False],
            "bootstrap_features": [True, False]
        }
        base = BaggingClassifier()
        clf = RandomizedSearchCV(base, param_grid, n_iter=100, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = clf.fit(x, y).best_estimator_
        return self.clf.get_params()
