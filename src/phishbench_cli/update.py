import configparser
import inspect

import phishbench.Features as Features
import phishbench.classification as classification
import phishbench.dataset.settings as dataset_settings
import phishbench.dataset.Imbalanced_Dataset as Imbalanced_Dataset
import phishbench.feature_extraction.email as email_extraction
from phishbench.classification.core import load_classifiers
from phishbench.evaluation import settings as evaluation_settings
from phishbench.evaluation.core import load_metrics
from phishbench.feature_extraction.email.email_features import load_features as load_email_features


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

    config[email_extraction.settings.FEATURE_TYPE_SECTION] = \
        email_extraction.settings.FEATURE_TYPE_SETTINGS
    config['URL_Feature_Types'] = {}
    config['URL_Feature_Types']['URL'] = "False"
    config['URL_Feature_Types']['Network'] = "False"
    config['URL_Feature_Types']['HTML'] = "False"
    config['URL_Feature_Types']['JavaScript'] = "False"

    reflection_features = load_email_features(filter_features=False)

    config[email_extraction.FeatureType.EMAIL_BODY.value] = {
        feature.config_name: "True" for feature in reflection_features if
        feature.feature_type == email_extraction.FeatureType.EMAIL_BODY
    }

    config[email_extraction.FeatureType.HEADER.value] = {
        feature.config_name: "True" for feature in reflection_features if
        feature.feature_type == email_extraction.FeatureType.HEADER
    }

    config['HTML_Features'] = {}
    c_html_features = config['HTML_Features']
    for feature in list_features:
        if feature.startswith("HTML_"):
            c_html_features[feature.replace('HTML_', '')] = "True"

    config['URL_Features'] = {}
    c_url_features = config['URL_Features']
    for feature in list_features:
        if feature.startswith("URL_"):
            c_url_features[feature.replace('URL_', '')] = "True"

    config['Network_Features'] = {}
    c_network_features = config['Network_Features']
    for feature in list_features:
        if feature.startswith("Network_"):
            c_network_features[feature.replace('Network_', '')] = "True"

    config['Javascript_Features'] = {}
    javascript_features_section = config['Javascript_Features']
    for feature in list_features:
        if feature.startswith("Javascript_"):
            javascript_features_section[feature.replace('Javascript_', '')] = "True"

    with open('Config_file.ini', 'w') as configfile:
        config.write(configfile)


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
    # execute only if run as a script
    list_features, list_imbalanced_dataset = update_list()
    make_config(list_features, list_imbalanced_dataset)


if __name__ == "__main__":
    main()
