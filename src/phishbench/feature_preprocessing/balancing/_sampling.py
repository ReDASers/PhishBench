"""
Implements dataset balancing
"""
import os

import joblib

from . import settings
from ._methods import METHODS
from ...utils import phishbench_globals


def run_sampling(x_train, y_train):
    """
    Runs the enabled sampling algorithms

    Parameters
    ----------
    x_train
        The training set features
    y_train
        The training set labels

    Returns
    -------
    feature_dict
        A dictionary containing the selected features from both the train set and the test set.
    """

    feature_dict = {}
    enabled_methods = {name: f for name, f in METHODS.items() if settings.method_enabled(name)}
    for method_name, method in enabled_methods.items():
        method_dir = os.path.join(phishbench_globals.output_dir, "Sampling", method_name)
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)
        result = method(x_train, y_train)
        if result is not None:
            feature_dict[method_name] = result
            joblib.dump(result, os.path.join(method_dir, f"{method_name} features.pkl"))

    return feature_dict
