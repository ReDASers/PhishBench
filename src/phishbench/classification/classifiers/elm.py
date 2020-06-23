from os import path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn_extensions.extreme_learning_machines import ELMRegressor
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.exceptions import NotFittedError

from ..core import BaseClassifier


def relu(x):
    return x * (x > 0)


class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, activation='relu'):
        self.activation = activation

    def fit(self, x, y):
        x, y = check_X_y(x, y)
        if self.activation == 'sigmoid':
            activation = expit
        elif self.activation == 'relu':
            activation = relu
        else:
            activation = self.activation
        self.elm_ = ELMRegressor(activation_func=activation)
        self.elm_.fit(x, y)
        return self

    def predict_proba(self, x):
        if not self.elm_:
            msg = "This {}} instance is not fitted yet. Call 'fit' with "\
                   "appropriate arguments before using this estimator.".format(type(self).__name__)

            raise NotFittedError(msg)
        x = check_array(x, accept_sparse=True)
        prob_pos = self.elm_.predict(x)
        # Ensures that probabilities are between 0 and 1
        prob_pos = np.clip(prob_pos, 0, 1)
        return prob_pos

    def predict(self, x):
        return np.round(self.predict_proba(x))

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"activation": self.activation}


class ExtremeLearningMachine(BaseClassifier):

    def __init__(self, io_dir):
        super().__init__(io_dir)
        self.clf = None
        self.model_path: str = path.join(self.io_dir, "model_svm.pkl")

    def fit(self, x, y):
        # Using the standard implementation of sigmoid throws a overflow warning.
        # We use scipy's implementation instead
        self.clf = ELMClassifier()
        self.clf.fit(x, y)

    def predict(self, x):
        assert self.clf is not None, "Classifier must be trained first"
        return self.clf.predict(x)

    def predict_proba(self, x):
        assert self.clf is not None, "Classifier must be trained first"
        return self.clf.predict_proba(x)

    def param_search(self, x, y):
        param_grid = {
            'activation': ['sigmoid', 'tanh', 'relu']
        }
        clf = ELMClassifier()
        cv_clf = GridSearchCV(clf, param_grid, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = cv_clf.fit(x, y).best_estimator_
        #print(cv_clf.cv_results_)
        return self.clf.get_params()

    def load_model(self):
        self.clf = joblib.load(self.model_path)

    def save_model(self):
        if self.clf is not None:
            joblib.dump(self.clf, self.model_path)
