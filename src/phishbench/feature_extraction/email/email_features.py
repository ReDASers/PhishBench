import re

from phishbench import Features
from phishbench.input import input as pb_input
from phishbench.utils import Globals
from . import reflection


def extract_dataset_features(legit_dataset_folder, phish_dataset_folder):
    Globals.logger.info("Extracting email features. Legit: %s Phish: %s", legit_dataset_folder, phish_dataset_folder)

    legit_features, legit_corpus = extract_email_features(legit_dataset_folder)
    phish_features, phish_corpus = extract_email_features(phish_dataset_folder)

    feature_list_dict_train = legit_features + phish_features

    labels_train = [0] * len(legit_features) + [1] * len(phish_features)
    corpus_train = legit_corpus + phish_corpus
    return feature_list_dict_train, labels_train, corpus_train


def extract_email_features(dataset_path):
    """
    :param dataset_path:
        The folder containing the dataset
    :return:
    List[Dict]:
        The extracted features
    List[str]:
        The corpus of emails
    """

    print("Extracting Email features from %s", dataset_path)
    Globals.logger.info("Extracting Email features from %s", dataset_path)

    emails, _ = pb_input.read_dataset_email(dataset_path)

    feature_dict_list = list()

    features = reflection.load_internal_features()

    for email_msg in emails:
        feature_values, _ = reflection.extract_features_from_single_email(features, email_msg)
        feature_dict_list.append(feature_values)

    corpus = [msg.body.text for msg in emails]

    return feature_dict_list, corpus


def get_url(body):
    url_regex = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', flags=re.IGNORECASE | re.MULTILINE)
    url = re.findall(url_regex, body)
    return url


def email_url_features(url_list, list_features, list_time):
    if Globals.config["Email_Features"]["extract body features"] == "True":
        Globals.logger.debug("Extracting email URL features")
        #Features.Email_URL_Number_Url(url_All, list_features, list_time)
        Features.Email_URL_Number_Diff_Domain(url_list, list_features, list_time)
        Features.Email_URL_Number_link_at(url_list, list_features, list_time)
        Features.Email_URL_Number_link_sec_port(url_list, list_features, list_time)