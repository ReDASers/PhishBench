import argparse
import configparser
import errno
import os.path
from importlib.util import spec_from_file_location, module_from_spec
from typing import *
import inspect

from phishbench import Classifiers, Evaluation_Metrics
from phishbench.dataset import Imbalanced_Dataset

from phishbench.feature_extraction.email.reflection import FeatureType, features as features_module


def update_list():

    list_classifiers = []
    list_evaluation_metrics = []
    list_imbalanced_dataset = []

    for a in dir(Classifiers):
        element=getattr(Classifiers, a)
        if inspect.isfunction(element):
            list_classifiers.append(a)

    for a in dir(Imbalanced_Dataset):
        element=getattr(Imbalanced_Dataset, a)
        if inspect.isfunction(element):
            list_imbalanced_dataset.append(a)

    for a in dir(Evaluation_Metrics):
        element=getattr(Evaluation_Metrics, a)
        if inspect.isfunction(element):
            list_evaluation_metrics.append(a)

    return list_classifiers, list_evaluation_metrics, list_imbalanced_dataset


def load_module_from_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_features(features_module) -> List[Callable]:
    """
    Loads the PhishBench features from a module

    Parameters
    ----------
    features_module : The module to load features from

    """
    # loads all features from module
    features = [getattr(features_module, x) for x in dir(features_module)]
    features = [x for x in features if hasattr(x, 'feature_type') and hasattr(x, 'config_name')]
    return features


def _generateConfig(args):
    config = configparser.ConfigParser()

    config["Summary"] = {}
    c_summary = config["Summary"]
    c_summary["Path"] = "summary.txt"

    config['Email or URL feature Extraction'] = {}
    c_email_url = config['Email or URL feature Extraction']
    c_email_url["extract_features_emails"] = "False"
    c_email_url["extract_features_urls"] = "True"

    config['Extraction'] = {}
    c_extraction = config['Extraction']
    c_extraction["Feature Extraction"] = "True"
    c_extraction["Training Dataset"] = "True"
    c_extraction["Testing Dataset"] = "True"

    config['Dataset Path'] = {}
    c_dataset = config['Dataset Path']
    c_dataset["path_legitimate_training"] = "Dataset_all/Dataset_legit_urls"
    c_dataset["path_phishing_training"] = "Dataset_all/Dataset_phish_urls"
    c_dataset["path_legitimate_testing"] = "Dataset_all/Dataset_legit_urls"
    c_dataset["path_phishing_testing"] = "Dataset_all/Dataset_legit_urls"

    config['Preprocessing'] = {}
    c_preprocessing = config['Preprocessing']
    # c_Preprocessing['mean_scaling']= "True"
    c_preprocessing['mix_max_scaling'] = "True"
    # c_Preprocessing['abs_scaler']= "True"
    # c_Preprocessing['normalize']= "True"

    config["Feature Selection"] = {}
    c_selection = config["Feature Selection"]
    c_selection["Select Best Features"] = "True"
    c_selection["Number of Best Features"] = "80"
    c_selection["Feature Ranking Only"] = "False"
    c_selection["Recursive Feature Elimination"] = "False"
    c_selection["Information Gain"] = "True"
    c_selection["Gini"] = "False"
    c_selection["Chi-2"] = "False"

    config['Classification'] = {}
    c_classification = config['Classification']
    c_classification["Running the Classifiers"] = "True"
    c_classification["Save Models"] = "True"

    config.add_section('Feature Types')
    for feature_type in FeatureType:
        config.set('Feature Types', feature_type.value, str(True))
        config.add_section(feature_type.value)

    features = _load_features(features_module)
    if args.feature_file:
        module = load_module_from_file(args.feature_file)
        features.extend(_load_features(module))

    for feature in features:
        config.set(feature.feature_type.value, feature.config_name, str(True))

    for feature_type in FeatureType:
        count = len(config[feature_type.value])
        print("%d features of type %s have been loaded " % (count, feature_type))

    list_classifiers, list_evaluation_metrics, list_imbalanced_dataset = update_list()

    config['Classifiers'] = {}
    c_classifiers = config['Classifiers']
    for classifier in list_classifiers:
        c_classifiers[classifier] = "True"

    config['Imbalanced Datasets'] = {}
    c_imbalanced = config['Imbalanced Datasets']
    c_imbalanced["load_imbalanced_dataset"] = "False"
    for imbalanced in list_imbalanced_dataset:
        c_imbalanced[imbalanced] = "True"

    config['Evaluation Metrics'] = {}
    c_metrics = config['Evaluation Metrics']
    for metric in list_evaluation_metrics:
        c_metrics[metric] = "True"

    with open(args.config_file, 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("-f", "--feature_file", help="The file containing the feature implementations", type=str,
                        default=None)
    parser.add_argument("-c", "--config_file", help="The config file to use.", type=str, default='Default_Config.ini')
    args = parser.parse_args()

    _generateConfig(args)
