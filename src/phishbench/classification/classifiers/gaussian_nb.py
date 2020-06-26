from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

from ..core import BaseClassifier


class GaussianNaiveBayes(BaseClassifier):

    def __init__(self, io_dir):
        super().__init__(io_dir, "model_gnb.pkl")

    def fit(self, x, y):
        self.clf = GaussianNB()
        self.clf.fit(x, y)

    def param_search(self, x, y):
        param_distributions = {"var_smoothing": [1e-09, 1e-08, 1e-07, 1e-06]}
        clf = GaussianNB()
        cv_clf = GridSearchCV(clf, param_distributions, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = cv_clf.fit(x, y).best_estimator_
        return self.clf.get_params()
