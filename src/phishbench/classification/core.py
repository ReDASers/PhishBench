import itertools
import os
from typing import List

import joblib
from scipy.sparse import issparse

from . import classifiers
from . import settings as classification_settings
from ..utils.reflection_utils import load_local_modules


class BaseClassifier:

    def __init__(self, io_dir, save_file):
        self.io_dir = io_dir
        self.model_path: str = os.path.join(self.io_dir, save_file)
        self.clf = None

    def fit(self, x, y):
        """
        Trains the classifier. If being used as a wrapper for a scikit-learn style classifier, then implementations of
        this function should store the trained underlying classifier in `self.clf`. Other implementations should
        also override predict and predict_proba
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

    def fit_weighted(self, x, y):
        print("{} does not support weighted training. Performing regular training.".format(self.name))
        self.fit(x, y)

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
        print("{} does not support parameter search. Performing regular training.".format(self.name))
        self.fit(x, y)

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
        assert self.clf is not None, "Classifier must be trained first"
        return self.clf.predict(x)

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
        assert self.clf is not None, "Classifier must be trained first"
        return self.clf.predict_proba(x)[:, 1]

    def load_model(self):
        """
        Loads the model from `self.io_dir`
        Returns
        -------
            None
        """
        self.clf = joblib.load(self.model_path)

    def save_model(self):
        """
        Saves the model to `self.io_dir`
        Returns
        -------
            None
        """
        assert self.clf is not None, "Classifier must be trained first"
        joblib.dump(self.clf, self.model_path)

    @property
    def name(self):
        return type(self).__name__


def load_internal_classifiers(filter_classifiers=True):
    """
    Loads internal classifiers and classifiers from the working directory
    Parameters
    ----------
    filter_classifiers: bool
        Whether or not to use the config to filter the classifiers

    Returns
    -------
        A list of classifiers
    """
    modules = load_local_modules()
    modules.append(classifiers)
    loaded_classifiers = [load_classifiers_from_module(module, filter_classifiers) for module in modules]
    loaded = list(itertools.chain.from_iterable(loaded_classifiers))
    return loaded


def load_classifiers_from_module(source, filter_classifiers=True) -> List[type]:
    """
    Loads classifiers from a module
    Parameters
    ----------
    source
        The module to load the classifiers from
    filter_classifiers
        Whether or not to use the config to filter the classifiers
    Returns
    -------
        A list of classifiers
    """
    module_classifiers: List[type] = list()
    for attr_name in dir(source):
        attr = getattr(source, attr_name)
        if isinstance(attr, type) and issubclass(attr, BaseClassifier):
            module_classifiers.append(attr)
    if filter_classifiers:
        return list(filter(classification_settings.is_enabled, module_classifiers))
    return module_classifiers


def train_classifiers(x_train, y_train, io_dir, verbose=1):
    """
    Train classifiers on the provided data
    Parameters
    ----------
    x_train
        The training features
    y_train
        The training labels
    io_dir
        The folder to interact with
    verbose
        Whether or not to print progress info to stdout.
        `0` prints nothing. `1` prints the classifiers being trained
    Returns
    -------
        A list of trained classifiers
    """
    if issparse(x_train):
        x_train = x_train.toarray()
    if not os.path.isdir(io_dir):
        os.makedirs(io_dir)

    classifiers: List[BaseClassifier] = [x(io_dir) for x in load_internal_classifiers()]

    for classifier in classifiers:
        if verbose > 0:
            print("Training {}.".format(classifier.name))
        if classification_settings.load_models():
            classifier.load_model()
        elif classification_settings.weighted_training():
            classifier.fit_weighted(x_train, y_train)
        elif classification_settings.param_search():
            classifier.param_search(x_train, y_train)
        else:
            classifier.fit(x_train, y_train)

        if classification_settings.save_models():
            classifier.save_model()

    return classifiers
