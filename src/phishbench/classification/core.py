import itertools
import os
from typing import List

from scipy.sparse import issparse

from . import classifiers
from . import settings as classification_settings
from .base_classifier import BaseClassifier
from ..utils.reflection_utils import load_local_modules


def load_classifiers(filter_classifiers=True):
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

    classifiers: List[BaseClassifier] = [x(io_dir) for x in load_classifiers()]

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
