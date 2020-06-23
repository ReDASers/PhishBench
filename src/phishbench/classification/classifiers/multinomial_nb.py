from os import path

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from ..core import BaseClassifier


class MultinomialNaiveBayes(BaseClassifier):

    def __init__(self, io_dir):
        super().__init__(io_dir)
        self.clf = None
        self.model_path: str = path.join(self.io_dir, "model_svm.pkl")

    def fit(self, x, y):
        self.clf = MultinomialNB()
        self.clf.fit(x, y)

    def param_search(self, x, y):
        param_distributions = {"alpha": [0.1, 0.5, 1]}
        clf = MultinomialNB()
        cv_clf = GridSearchCV(clf, param_distributions, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = cv_clf.fit(x, y).best_estimator_
        return self.clf.get_params()

    def predict(self, x):
        assert self.clf is not None, "Classifier must be trained first"
        return self.clf.predict(x)

    def predict_proba(self, x):
        assert self.clf is not None, "Classifier must be trained first"
        return self.clf.predict_proba(x)[:, 1]

    def load_model(self):
        self.clf = joblib.load(self.model_path)

    def save_model(self):
        joblib.dump(self.clf, self.model_path)
