"""
The PhishBench Config Generator

This script generates PhishBench configuration files.
"""
# pylint: disable=missing-function-docstring
import argparse
import configparser
import inspect

import phishbench.classification as classification
import phishbench.feature_preprocessing.sampling as sampling
import phishbench.input.settings as input_settings
import phishbench.settings
from phishbench.classification.core import load_classifiers
from phishbench.evaluation import settings as evaluation_settings
from phishbench.evaluation.core import load_metrics
from phishbench.feature_extraction import settings as extraction_settings
from phishbench.feature_extraction.email import features as internal_email_features
from phishbench.feature_extraction.reflection import load_features, FeatureType
from phishbench.feature_extraction.url import features as internal_url_features
from phishbench.feature_preprocessing.feature_selection import settings as selection_settings


def make_config():
    list_imbalanced_dataset = update_list()
    config = configparser.ConfigParser()

    config[phishbench.settings.PB_SECTION] = phishbench.settings.DEFAULT_SETTINGS
    config[input_settings.DATASET_PATH_SECTION] = input_settings.DEFAULT_SETTINGS

    config['Extraction'] = {}
    config['Extraction']["Training Dataset"] = "True"
    config['Extraction']["Testing Dataset"] = "True"

    config['Features Export'] = {}
    config['Features Export']['csv'] = "True"

    config['Preprocessing'] = {}
    preprocessing_section = config['Preprocessing']
    # preprocessing_section['mean_scaling']= "True"
    preprocessing_section['min_max_scaling'] = "True"
    # preprocessing_section['abs_scaler']= "True"
    # preprocessing_section['normalize']= "True"

    config[selection_settings.FEATURE_SELECTION_SECTION] = selection_settings.DEFAULT_FEATURE_SELECTION_SETTINGS
    config[selection_settings.SELECTION_METHODS_SECTION] = selection_settings.DEFAULT_METHODS_SETTINGS

    config['Imbalanced Datasets'] = {}
    imbalanced_section = config['Imbalanced Datasets']
    imbalanced_section["load_imbalanced_dataset"] = "False"
    for imbalanced in list_imbalanced_dataset:
        imbalanced_section[imbalanced] = "True"

    config[classification.settings.CLASSIFICATION_SECTION] = classification.settings.DEFAULT_SETTINGS

    config[classification.settings.CLASSIFIERS_SECTION] = {
        x.__name__: "True" for x in load_classifiers(filter_classifiers=False)
    }

    config[evaluation_settings.EVALUATION_SECTION] = {
        x.config_name: "True" for x in load_metrics(filter_metrics=False)
    }

    config["Summary"] = {}
    config["Summary"]["Path"] = "summary.txt"

    config[extraction_settings.EMAIL_TYPE_SECTION] = \
        extraction_settings.EMAIL_TYPE_SETTINGS

    config[extraction_settings.URL_TYPE_SECTION] = extraction_settings.URL_TYPE_SETTINGS

    internal_features = [internal_email_features, internal_url_features]
    reflection_features = load_features(internal_features=internal_features, filter_features=None)

    for feature_type in FeatureType:
        print(feature_type.value)
        config[feature_type.value] = {
            feature.config_name: "True" for feature in reflection_features if
            feature.feature_type == feature_type
        }

    return config


def update_list():
    list_imbalanced_dataset = []

    for member in dir(sampling.Imbalanced_Dataset):
        element = getattr(sampling.Imbalanced_Dataset, member)
        if inspect.isfunction(element):
            list_imbalanced_dataset.append(member)

    return list_imbalanced_dataset


def main():
    # The entrypoint of the script
    parser = argparse.ArgumentParser(description='PhishBench Config Generator')
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("-f", "--config_file", help="The name of the config file to generate.",
                        type=str, default='Config_file.ini')
    args = parser.parse_args()

    print("Generating PhishBench Config")
    config = make_config()

    print("Saving to ", args.config_file)
    with open(args.config_file, 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    main()
