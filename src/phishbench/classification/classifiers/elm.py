import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_X_y, check_array
from sklearn_extensions.extreme_learning_machines import ELMRegressor

from ..core import BaseClassifier


def relu(x):
    return x * (x > 0)


class _ELMClassifier(BaseEstimator, ClassifierMixin):
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
            msg = "This {}} instance is not fitted yet. Call 'fit' with " \
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
        super().__init__(io_dir, "model_elm.pkl")

    def fit(self, x, y):
        # Using the standard implementation of sigmoid throws a overflow warning.
        # We use scipy's implementation instead
        self.clf = _ELMClassifier()
        self.clf.fit(x, y)

    def param_search(self, x, y):
        param_grid = {
            'activation': ['sigmoid', 'tanh', 'relu']
        }
        clf = _ELMClassifier()
        cv_clf = GridSearchCV(clf, param_grid, n_jobs=-1, pre_dispatch='2*n_jobs')
        self.clf = cv_clf.fit(x, y).best_estimator_
        return self.clf.get_params()

    def predict_proba(self, x):
        return self.clf.predict_proba(x)
