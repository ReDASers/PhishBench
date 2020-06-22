import os
from typing import List

from . import settings as classification_settings
from ..utils import Globals


class BaseClassifier:

    def __init__(self, io_dir):
        self.io_dir = io_dir

    def fit(self, x, y):
        """
        Trains the classifier
        Parameters
        ----------
        x: array-like or sparse matrix of shape (n,f)
            Training vectors, where n is the number of samples and f is the number of features.
        y: array-like of shape (n)
            Target values, with 0 being legitimate and 1 being phishing
        Returns
        -------
            None
        """
        pass

    # TODO Discuss how to add weighted training
    # Option 1: Have classifiers that support it perform weighted training in the fit function
    # Option 2: Have a separate weighted_fit function

    def param_search(self, x, y):
        """
        Performs parameter search to find the best parameters.
        Parameters
        ----------
        x: array-like or sparse matrix of shape (n,f)
            Training vectors, where n is the number of samples and f is the number of features.
        y: array-like of shape (n)
            Target values, with 0 being legitimate and 1 being phishing
        Returns
        -------
        dict:
            The best parameters.
        """
        pass

    def predict(self, x):
        """
        Parameters
        ----------
        x: array-like or sparse matrix of shape (n,f)
            Test vectors, where n is the number of samples and f is the number of features
        Returns
        -------
        array-like of shape (n)
            The predicted class values
        """
        pass

    def predict_proba(self, x):
        """
        Parameters
        ----------
        x: array-like or sparse matrix of shape (n,f)
            Test vectors, where n is the number of samples and f is the number of features
        Returns
        -------
        array-like of shape (n)
            The probability of each test vector being phish
        """
        pass

    def load_model(self):
        """
        Loads the model from self.io_dir
        Returns
        -------
            None
        """
        pass

    def save_model(self):
        """
        Saves the model to self.io_dir
        Returns
        -------
            None
        """
        pass

    @property
    def name(self):
        return type(self).__name__


def load_internal_classifiers():
    from . import classifiers
    return load_classifiers(classifiers)


def load_classifiers(source) -> List[type]:
    attrs = [getattr(source, x) for x in dir(source)]
    attrs = [x for x in attrs if isinstance(x, type) and issubclass(x, BaseClassifier)]
    return attrs


def train_classifiers(x_train, y_train, io_dir):
    if not os.path.isdir(io_dir):
        os.makedirs(io_dir)

    classifiers: List[BaseClassifier] = [x(io_dir) for x in load_internal_classifiers()]

    for classifier in classifiers:
        if classification_settings.load_models():
            classifier.load_model()
        else:
            classifier.fit(x_train, y_train)

        if classification_settings.save_models():
            classifier.save_model()
    return classifiers
