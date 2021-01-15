"""
This module contains the RandomForest classifier
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from .._base_classifier import BaseClassifier


class RandomForest(BaseClassifier):
    """
    The Random Forest Classifier. This class wraps the `sklearn.ensemble.RandomForestClassifier`
    """
    def __init__(self, io_dir):
        super().__init__(io_dir, "model_rf.pkl")

    def fit(self, x, y):
        self.clf = RandomForestClassifier(n_jobs=-1)
        self.clf.fit(x, y)

    def fit_weighted(self, x, y):
        self.clf = RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1)
        self.clf.fit(x, y)

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
