from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from ..core import BaseClassifier


class DecisionTree(BaseClassifier):
    def __init__(self, io_dir):
        super().__init__(io_dir, "model_dt.pkl")

    def fit(self, x, y):
        self.clf = DecisionTreeClassifier()
        self.clf.fit(x, y)

    def param_search(self, x, y):
        param_grid = {
            "criterion": ['gini', 'entropy'],
            "max_depth": range(10, 110, 10),
            "min_samples_split": [0.1, 0.2, 0.3, 0.4, 0.5, 2, 3],
            "min_samples_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5],
            'max_features': ["sqrt", "log2", None]
        }
        base = DecisionTreeClassifier()
        clf = RandomizedSearchCV(base, param_grid, n_iter=100, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = clf.fit(x, y).best_estimator_
        return self.clf.get_params()
