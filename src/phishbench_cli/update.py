import configparser
import inspect

import phishbench.Features as Features
import phishbench.classification as classification
import phishbench.dataset.Imbalanced_Dataset as Imbalanced_Dataset
from phishbench.classification.core import load_internal_classifiers
from phishbench.evaluation import settings as evaluation_settings
from phishbench.evaluation.core import load_internal_metrics
from phishbench.feature_extraction.email.reflection import FeatureType
from phishbench.feature_extraction.email.reflection import load_internal_features as load_email_features


def make_config(list_features, list_classifiers, list_imbalanced_dataset, list_evaluation_metrics):
    config = configparser.ConfigParser()

    config['Dataset Path'] = {}
    dataset_section = config['Dataset Path']
    dataset_section["path_legitimate_training"] = "Dataset_all/Dataset_legit_urls"
    dataset_section["path_phishing_training"] = "Dataset_all/Dataset_phish_urls"
    dataset_section["path_legitimate_testing"] = "Dataset_all/Dataset_legit_urls"
    dataset_section["path_phishing_testing"] = "Dataset_all/Dataset_legit_urls"

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

    config[classification.settings.CLASSIFIERS_SECTION] = {}
    classifiers_section = config[classification.settings.CLASSIFIERS_SECTION]
    for classifier in list_classifiers:
        classifiers_section[classifier] = "True"

    config[evaluation_settings.EVALUATION_SECTION] = {}
    metrics_section = config[evaluation_settings.EVALUATION_SECTION]
    for metric in list_evaluation_metrics:
        metrics_section[metric] = "True"

    config["Summary"] = {}
    summary_section = config["Summary"]
    summary_section["Path"] = "summary.txt"

    config["Support Files"] = {}
    config["Support Files"]["path_alexa_data"] = "\\path_to_alexa\\top-1m.csv"

    config['Email_Features'] = {}
    config['Email_Features']['extract header features'] = "False"
    config['Email_Features']['extract body features'] = "False"

    config['URL_Feature_Types'] = {}
    config['URL_Feature_Types']['URL'] = "False"
    config['URL_Feature_Types']['Network'] = "False"
    config['URL_Feature_Types']['HTML'] = "False"
    config['URL_Feature_Types']['JavaScript'] = "False"

    reflection_features = load_email_features(filter_features=False)

    config['Email_Header_Features'] = {
        feature.config_name: "True" for feature in reflection_features if feature.feature_type == FeatureType.HEADER
    }
    #  header_features = config['Email_Header_Features']
    # for feature in list_features:
    #     if feature.startswith("Email_Header_"):
    #         header_features[feature.replace('Email_Header_', '')] = "True"

    config['Email_Body_Features'] = {
        feature.config_name: "True" for feature in reflection_features if feature.feature_type == FeatureType.EMAIL_BODY
    }
    # body_features = config['Email_Body_Features']
    # for feature in list_features:
    #     if feature.startswith("Email_Body_"):
    #         body_features[feature.replace('Email_Body_', '')] = "True"

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

    list_classifiers = [x.__name__ for x in load_internal_classifiers(filter_classifiers=False)]

    for member in dir(Imbalanced_Dataset):
        element = getattr(Imbalanced_Dataset, member)
        if inspect.isfunction(element):
            list_imbalanced_dataset.append(member)

    list_evaluation_metrics = [x.config_name for x in load_internal_metrics(filter_metrics=False)]

    return list_features, list_classifiers, list_imbalanced_dataset, list_evaluation_metrics


def main():
    # execute only if run as a script
    list_features, list_classifiers, list_imbalanced_dataset, list_evaluation_metrics = update_list()
    make_config(list_features, list_classifiers, list_imbalanced_dataset, list_evaluation_metrics)


if __name__ == "__main__":
    main()
