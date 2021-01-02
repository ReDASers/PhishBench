"""
Contains code to run preprocessing
"""
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from . import balancing
from . import feature_selection
from . import settings
from ..utils import phishbench_globals


def process_vectorized_features(x_train, y_train, x_test, feature_names, output_dir: str):
    """
    Processes vectorized features

    Parameters
    ==========
    x_train:
        A scipy sparse array containing the vectorized training set features
    y_train:
        A numpy array containing the vectorized training set labels
    x_test:
        A scipy sparse array containing the test set features
    feature_names:
        The names of the vectorized features
    output_dir:
        Where to output files to

    Returns
    =======

    x_train_dict2:
        A two-layer dict containing the selected features from the train sets
        Hierarchy: balancing method -> feature selection method -> features
    x_test_dict2
        A two-layer dict containing the selected features the test set
        Hierarchy: balancing method -> feature selection method -> features
    y_train_dict:
        A dict mapping balancing methods to training set labels
    """
    if settings.min_max_scaling():
        x_train = x_train.toarray()
        x_test = x_test.toarray()
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    train_samples = {
        'None': (x_train, y_train)
    }
    sampling_dir = os.path.join(output_dir, "Sampling")
    if not os.path.isdir(sampling_dir):
        os.mkdir(sampling_dir)
    if settings.dataset_balancing():
        train_samples.update(balancing.run_sampling(x_train, y_train, sampling_dir))

    x_train_dict2 = {}
    x_test_dict2 = {}
    y_train_dict = {}
    for balancing_method in train_samples:
        x_train, y_train = train_samples[balancing_method]
        if len(np.unique(y_train)) == 1:
            print(f"{balancing_method} produced samples with only one class. Omitting")
            phishbench_globals.logger.warning(f"{balancing_method} produced samples with only one class. Omitting")
            continue
        y_train_dict[balancing_method] = y_train
        # Feature Selection
        if settings.feature_selection():
            feature_output_dir = os.path.join(output_dir, "Feature Selection", balancing_method)
            # x_test should be the same no matter the sampling method
            x_train_dict, x_test_dict = feature_selection.run_feature_extraction(
                x_train, x_test, y_train, feature_names, feature_output_dir)
        else:
            x_train_dict = {
                'None': x_train
            }
            x_test_dict = {
                'None': x_test
            }
        x_train_dict2[balancing_method] = x_train_dict
        x_test_dict2[balancing_method] = x_test_dict
    return x_train_dict2, x_test_dict2, y_train_dict
