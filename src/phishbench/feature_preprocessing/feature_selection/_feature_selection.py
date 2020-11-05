"""
Contains implementations of feature selection functions
"""
import math
import os

import joblib

from . import settings
from ._methods import METHODS
from ...utils import phishbench_globals


def transform_features(selection_model, x_train, x_test, output_dir):
    """
    Transforms the features
    Parameters
    ----------
    selection_model:
        The feature selector
    x_train
        The training set features
    x_test
        The test set features
    output_dir:
        The folder to output pickled features

    Returns
    -------
    x_train_selection:
        The selected features from x_train
    x_test_selection:
        The selected features from x_test. `None` if `x_test` is `None`
    """
    x_train_selection = selection_model.transform(x_train)
    joblib.dump(x_train_selection, os.path.join(output_dir, "best_features_train.pkl"))
    if x_test is not None:
        x_test_selection = selection_model.transform(x_test)
        joblib.dump(x_test_selection, os.path.join(output_dir, "best_features_test.pkl"))
        return x_train_selection, x_test_selection
    return x_train_selection, None


def run_feature_extraction(x_train, x_test, y_train, feature_names):
    """
    Runs the enabled feature selection algorithms

    Parameters
    ----------
    x_train
        The training set features
    x_test
        The test set features
    y_train
        The training set labels
    feature_names
        The names of the features

    Returns
    -------
    x_train_dict:
        A dictionary mapping the method name to the selected features from x_train
    x_test_dict:
        A dictionary mapping the method name to the selected features from x_test
    """
    num_features = min(settings.num_features(), x_train.shape[1])

    x_train_dict = {
        'None': x_train
    }
    x_test_dict = {
        'None': x_test
    }
    enabled_methods = {name: f for name, f in METHODS.items() if settings.method_enabled(name)}
    for method_name, method in enabled_methods.items():
        method_dir = os.path.join(phishbench_globals.output_dir, "Feature Selection", method_name)
        if not os.path.exists(method_dir):
            os.makedirs(method_dir)

        # Rank features
        selection_model, ranking = method(x_train, y_train, num_features)
        ranking = [0 if math.isnan(x) else x for x in ranking]
        ranking = sorted(zip(feature_names, ranking), key=lambda x: x[1], reverse=True)

        # Write rankings to file
        with open(os.path.join(method_dir, "ranking.txt"), 'w', errors="ignore") as f:
            for feature_name, rank in ranking:
                f.write(f"{feature_name}: {rank}\n")

        joblib.dump(selection_model, os.path.join(method_dir, "selection_model.pkl"))

        # Transform features
        x_train_selection, x_test_selection = transform_features(selection_model, x_train, x_test, method_dir)
        x_train_dict[method_name] = x_train_selection
        if x_test_selection is not None:
            x_test_dict[method_name] = x_test_selection

    return x_train_dict, x_test_dict
