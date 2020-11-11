"""
The PhishBench Config Generator

This script generates PhishBench configuration files.
"""
# pylint: disable=missing-function-docstring
import argparse
import configparser

import phishbench.classification as classification
import phishbench.evaluation as evaluation
import phishbench.evaluation.settings
import phishbench.feature_extraction as extraction
import phishbench.feature_extraction.settings
import phishbench.feature_preprocessing as preprocessing
import phishbench.feature_preprocessing.balancing
import phishbench.feature_preprocessing.balancing.settings
import phishbench.feature_preprocessing.settings
import phishbench.input as pb_input
import phishbench.input.settings
import phishbench.settings
from phishbench.classification.core import load_classifiers
from phishbench.evaluation.core import load_metrics
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
    config[pb_input.settings.DATASET_PATH_SECTION] = pb_input.settings.DEFAULT_SETTINGS

    config['Extraction'] = {}
    config['Extraction']["Training Dataset"] = "True"
    config['Extraction']["Testing Dataset"] = "True"
    config['Extraction']["Split Dataset"] = "False"

    config['Features Export'] = {}
    config['Features Export']['csv'] = "True"

    config[preprocessing.settings.SECTION_NAME] = preprocessing.settings.DEFAULTS

    config[selection_settings.FEATURE_SELECTION_SECTION] = selection_settings.DEFAULT_FEATURE_SELECTION_SETTINGS
    config[selection_settings.SELECTION_METHODS_SECTION] = selection_settings.DEFAULT_METHODS_SETTINGS

    config[preprocessing.balancing.settings.SECTION] = preprocessing.balancing.settings.DEFAULT_SETTINGS

    config[classification.settings.SECTION] = classification.settings.DEFAULT_SETTINGS

    config[classification.settings.CLASSIFIERS_SECTION] = {
        x.__name__: "True" for x in load_classifiers(filter_classifiers=False)
    }

    config[evaluation.settings.SECTION] = {
        x.config_name: "True" for x in load_metrics(filter_metrics=False)
    }

    config[extraction.settings.EMAIL_TYPE_SECTION] = \
        extraction.settings.EMAIL_TYPE_SETTINGS

    config[extraction.settings.URL_TYPE_SECTION] = extraction.settings.URL_TYPE_SETTINGS

    internal_features = [internal_email_features, internal_url_features]
    reflection_features = load_features(internal_features=internal_features, filter_features=None)

    for feature_type in FeatureType:
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
