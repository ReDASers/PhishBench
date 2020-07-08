from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from ..core import BaseClassifier


class Boosting(BaseClassifier):
    def __init__(self, io_dir):
        super().__init__(io_dir, "model_boo.pkl")

    def fit(self, x, y):
        self.clf = AdaBoostClassifier()
        self.clf.fit(x, y)

    def fit_weighted(self, x, y):
        base_classifier = DecisionTreeClassifier(class_weight='balanced')
        self.clf = AdaBoostClassifier(base_estimator=base_classifier)
        self.clf.fit(x, y)

    def param_search(self, x, y):
        param_grid = {
            "n_estimators": range(10, 110, 10),
            "learning_rate": [0.01, 0.1, 0.25, 1.0, 1.25, 1.5],
            "algorithm": ['SAMME', 'SAMME.R']
        }
        base = AdaBoostClassifier()
        clf = RandomizedSearchCV(base, param_grid, n_iter=100, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = clf.fit(x, y).best_estimator_
        return self.clf.get_params()
