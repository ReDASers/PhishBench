from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from ..base_classifier import BaseClassifier


class MultinomialNaiveBayes(BaseClassifier):

    def __init__(self, io_dir):
        super().__init__(io_dir, "model_mnb.pkl")

    def fit(self, x, y):
        self.clf = MultinomialNB()
        self.clf.fit(x, y)

    def param_search(self, x, y):
        param_distributions = {"alpha": [0.1, 0.5, 1]}
        clf = MultinomialNB()
        cv_clf = GridSearchCV(clf, param_distributions, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = cv_clf.fit(x, y).best_estimator_
        return self.clf.get_params()
