"""
Implements dataset balancing
"""
import os

import joblib

from . import settings
from ._methods import METHODS


def run_sampling(x_train, y_train, output_dir):
    """
    Runs the enabled sampling algorithms

    Parameters
    ----------
    x_train
        The training set features
    y_train
        The training set labels
    output_dir:
        Folder to output the selected features

    Returns
    -------
    feature_dict
        A dictionary containing the selected features from both the train set and the test set.
    """

    feature_dict = {}
    enabled_methods = {name: f for name, f in METHODS.items() if settings.method_enabled(name)}
    for method_name, method in enabled_methods.items():
        result = method(x_train, y_train)
        if result is not None:
            feature_dict[method_name] = result
            joblib.dump(result, os.path.join(output_dir, f"{method_name} features.pkl"))

    return feature_dict
