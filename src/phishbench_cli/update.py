"""
The PhishBench Config Generator

This script generates PhishBench configuration files.
"""
# pylint: disable=missing-function-docstring
import argparse
import configparser

import phishbench.classification as classification
import phishbench.feature_preprocessing.settings as preprocessing_settings
import phishbench.feature_preprocessing.balancing as balancing
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


def make_config() -> configparser.ConfigParser:
    """
    Constructs a default config, which can be be written to a file using the `write` function

    Returns
    =======
    config
        A `configparser.ConfigParser` object with default PhishBench settings.
    """
    config = configparser.ConfigParser()

    config[phishbench.settings.PB_SECTION] = phishbench.settings.DEFAULT_SETTINGS
    config[input_settings.DATASET_PATH_SECTION] = input_settings.DEFAULT_SETTINGS

    config['Extraction'] = {}
    config['Extraction']["Training Dataset"] = "True"
    config['Extraction']["Testing Dataset"] = "True"

    config['Features Export'] = {}
    config['Features Export']['csv'] = "True"

    config[preprocessing_settings.SECTION_NAME] = preprocessing_settings.DEFAULTS

    config[selection_settings.FEATURE_SELECTION_SECTION] = selection_settings.DEFAULT_FEATURE_SELECTION_SETTINGS
    config[selection_settings.SELECTION_METHODS_SECTION] = selection_settings.DEFAULT_METHODS_SETTINGS

    config[balancing.settings.SAMPLING_SECTION] = balancing.settings.DEFAULT_SAMPLING_SETTINGS

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
