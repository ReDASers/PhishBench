from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LR

from .._base_classifier import BaseClassifier


class LogisticRegression(BaseClassifier):
    def __init__(self, io_dir):
        super().__init__(io_dir, "model_lr.pkl")

    def fit(self, x, y):
        self.clf = LR(solver='liblinear')
        self.clf.fit(x, y)

    def fit_weighted(self, x, y):
        self.clf = LR(class_weight='balanced', solver='liblinear')
        self.clf.fit(x, y)

    def param_search(self, x, y):
        # param_grid = {
        #     "penalty": ['l1', 'l2', 'elasticnet', 'none'],
        #     "C": [1, 2, 3, 4],
        #     "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        # }
        param_grid = [
            {
                "solver": ['lbfgs', 'newton-cg', 'sag'],
                "penalty": ['l2'],
                "C": [1.0, 0.5, .025]
            },
            {
                "solver": ['liblinear'],
                "penalty": ['l1', 'l2'],
                "C": [1.0, 0.5, .025]
            },
            {
                "solver": ['newton-cg', 'lbfgs', 'sag', 'saga'],
                "penalty": ['none']
            },
            {
                "solver": ['saga'],
                "penalty": ['elasticnet'],
                "l1_ratio": [1.0, 0.5, .025]
            }
        ]
        base = LR()
        clf = GridSearchCV(base, param_grid, n_jobs=-1, pre_dispatch='2*n_jobs', error_score=0)
        self.clf = clf.fit(x, y).best_estimator_
        return self.clf.get_params()
