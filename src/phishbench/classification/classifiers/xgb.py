from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ..base_classifier import BaseClassifier


class XGBoost(BaseClassifier):

    def __init__(self, io_dir):
        super().__init__(io_dir, 'model_xgb.pkl')

    def fit(self, x, y):
        self.clf = XGBClassifier(use_label_encoder=False)
        self.clf.fit(x, y)

    def fit_weighted(self, x, y):
        self.clf = XGBClassifier(use_label_encoder=False)

        weights = compute_sample_weight('balanced', y=y)
        self.clf.fit(x, y, sample_weight=weights)

    def param_search(self, x, y):
        param_grid = {
            "learning_rate": [0.001, 0.0025, 0.01, 0.025, 0.1, 0.25],
            "booster": ['gbtree', 'gblinear', 'dart']
        }
        base = XGBClassifier(use_label_encoder=False)
        clf = GridSearchCV(base, param_grid, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = clf.fit(x, y).best_estimator_
        return self.clf.get_params()
