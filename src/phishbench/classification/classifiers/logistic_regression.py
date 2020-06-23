from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression as LR

from ..core import BaseClassifier


class LogisticRegression(BaseClassifier):
    def __init__(self, io_dir):
        super().__init__(io_dir, "model_lr.pkl")

    def fit(self, x, y):
        self.clf = LR()
        self.clf.fit(x, y)

    def fit_weighted(self, x, y):
        self.clf = LR(class_weight='balanced_subsample')
        self.clf.fit(x, y)

    def param_search(self, x, y):
        param_grid = {
            "penalty": ['l1', 'l2', 'elasticnet', 'none'],
            "C": [1, 2, 3, 4],
            "solver": ['warn', 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        }
        base = LR()
        clf = RandomizedSearchCV(base, param_grid, n_iter=100, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = clf.fit(x, y).best_estimator_
        return self.clf.get_params()
