import argparse
import configparser
import inspect

import phishbench.Features as Features
import phishbench.classification as classification
import phishbench.dataset.Imbalanced_Dataset as Imbalanced_Dataset
import phishbench.dataset.settings as dataset_settings
from phishbench.classification.core import load_classifiers
from phishbench.evaluation import settings as evaluation_settings
from phishbench.evaluation.core import load_metrics
from phishbench.feature_extraction import settings as extraction_settings
from phishbench.feature_extraction.email import features as internal_email_features
from phishbench.feature_extraction.reflection import load_features, FeatureType
from phishbench.feature_extraction.url import features as internal_url_features


def make_config(list_features, list_imbalanced_dataset):
    config = configparser.ConfigParser()

    config[dataset_settings.DATASET_PATH_SECTION] = dataset_settings.DEFAULT_SETTINGS

    config['Email or URL feature Extraction'] = {}
    proccess_section = config['Email or URL feature Extraction']
    proccess_section["extract_features_emails"] = "False"
    proccess_section["extract_features_urls"] = "True"

    config['Extraction'] = {}
    extraction_section = config['Extraction']
    extraction_section["Feature Extraction"] = "True"
    extraction_section["Training Dataset"] = "True"
    extraction_section["Testing Dataset"] = "True"

    config['Features Export'] = {}
    features_format_section = config['Features Export']
    features_format_section['csv'] = "True"

    config['Preprocessing'] = {}
    preprocessing_section = config['Preprocessing']
    # preprocessing_section['mean_scaling']= "True"
    preprocessing_section['min_max_scaling'] = "True"
    # preprocessing_section['abs_scaler']= "True"
    # preprocessing_section['normalize']= "True"

    config["Feature Selection"] = {}
    feature_selection_section = config["Feature Selection"]
    feature_selection_section["select best features"] = "True"
    feature_selection_section["Number of Best Features"] = "80"
    feature_selection_section["Recursive Feature Elimination"] = "False"
    feature_selection_section["Information Gain"] = "True"
    feature_selection_section["Gini"] = "False"
    feature_selection_section["Chi-2"] = "False"
    feature_selection_section["with Tfidf"] = "False"

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
    summary_section = config["Summary"]
    summary_section["Path"] = "summary.txt"

    config["Support Files"] = {}
    config["Support Files"]["path_alexa_data"] = "\\path_to_alexa\\top-1m.csv"

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

    c_url_features = config['URL_Features']
    for feature in list_features:
        if feature.startswith("URL_"):
            c_url_features[feature.replace('URL_', '')] = "True"

    c_html_features = config[FeatureType.URL_WEBSITE.value]
    for feature in list_features:
        if feature.startswith("HTML_"):
            c_html_features[feature.replace('HTML_', '')] = "True"

    c_network_features = config[FeatureType.URL_NETWORK.value]
    for feature in list_features:
        if feature.startswith("Network_"):
            c_network_features[feature.replace('Network_', '')] = "True"

    javascript_features_section = config[FeatureType.URL_WEBSITE_JAVASCRIPT.value]
    for feature in list_features:
        if feature.startswith("Javascript_"):
            javascript_features_section[feature.replace('Javascript_', '')] = "True"
    return config


def update_list():
    list_features = []
    list_imbalanced_dataset = []
    for member in dir(Features):
        element = getattr(Features, member)
        if inspect.isfunction(element):
            list_features.append(member)

    for member in dir(Imbalanced_Dataset):
        element = getattr(Imbalanced_Dataset, member)
        if inspect.isfunction(element):
            list_imbalanced_dataset.append(member)

    return list_features, list_imbalanced_dataset


def main():

    # The entrypoint of the script
    parser = argparse.ArgumentParser(description='PhishBench Config Generator')
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("-f", "--config_file", help="The name of the config file to generate.",
                        type=str, default='Config_file.ini')
    args = parser.parse_args()

    list_features, list_imbalanced_dataset = update_list()
    config = make_config(list_features, list_imbalanced_dataset)
    print("Generating PhishBench Config")

    print("Saving to ", args.config_file)
    with open(args.config_file, 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    main()
