from configparser import ConfigParser
from phishbench.feature_extraction.reflection import FeatureType


def get_mock_config() -> ConfigParser:
    config = ConfigParser()
    config['Features'] = {}
    for ftype in FeatureType:
        config[ftype.value] = {}

    config['Classifiers'] = {}
    config['Imbalanced Datasets'] = {}
    config['Evaluation Metrics'] = {}
    config['Preprocessing'] = {}
    config["Feature Selection"] = {}
    config['Dataset Path'] = {}
    config['Email or URL feature Extraction'] = {}
    config['Extraction'] = {}
    config['Features Format'] = {}
    config['Classification'] = {}
    config["Summary"] = {}
    return config
