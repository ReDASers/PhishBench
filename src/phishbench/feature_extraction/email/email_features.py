"""
This module contains code for email feature extraction.
"""
from typing import List, Callable

from tqdm import tqdm

from . import reflection
from ...input import input as pb_input
from ...input.email_input.models import EmailMessage
from ...utils import phishbench_globals


def extract_labeled_dataset(legit_dataset_folder: str, phish_dataset_folder: str):
    """
    Extracts features from a dataset of emails split in two folders
    :param legit_dataset_folder:
        The folder containing emails of the phishing class
    :param phish_dataset_folder:
        The folder containing emails of the legitimate class
    :return:
    List[Dict]:
        The extracted features
    List[int]:
        The labels of corresponding to the features
    List[str]:
        The corpus of emails
    """
    features = reflection.load_internal_features()
    print("Loaded {} features".format(len(features)))

    phishbench_globals.logger.info("Extracting email features. Legit: %s Phish: %s",
                                   legit_dataset_folder, phish_dataset_folder)

    print("Loading emails from {}".format(legit_dataset_folder))
    legit_emails, _ = pb_input.read_dataset_email(legit_dataset_folder)
    print("Extracting features")
    legit_features, legit_corpus = extract_email_features(legit_emails, features)

    print("Loading emails from{}".format(phish_dataset_folder))
    phish_emails, _ = pb_input.read_dataset_email(phish_dataset_folder)
    print("Extracting features")
    phish_features, phish_corpus = extract_email_features(phish_emails, features)

    feature_list_dict_train = legit_features + phish_features
    labels_train = [0] * len(legit_features) + [1] * len(phish_features)
    corpus_train = legit_corpus + phish_corpus

    return feature_list_dict_train, labels_train, corpus_train


def extract_unlabeled_dataset(dataset_folder: str):
    """
    Extracts features from a set of unlabeled urls
    :param dataset_folder:
        The folder containing the emails to extract features from
    :return:
    List[Dict]:
        The extracted features
    List[str]:
        The corpus of emails
    """
    features = reflection.load_internal_features()
    print("Loaded {} features".format(len(features)))

    Globals.logger.info("Extracting unlabeled email features. %s", dataset_folder)

    emails = pb_input.read_dataset_email(dataset_folder)

    features, corpus = extract_email_features(emails, features)

    return features, corpus


def extract_email_features(emails: List[EmailMessage], features: List[Callable]):
    """
    :param emails:
        The dataset of emails
    :param features:
        A list of features to extract
    :return:
    List[Dict]:
        The extracted features
    List[str]:
        The corpus of emails
    """
    feature_dict_list = list()

    for email_msg in tqdm(emails):
        feature_values, _ = reflection.extract_features_from_single_email(features, email_msg)
        feature_dict_list.append(feature_values)

    corpus = [msg.body.text for msg in emails]

    return feature_dict_list, corpus

# def get_url(body):
#     url_regex = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', flags=re.IGNORECASE | re.MULTILINE)
#     url = re.findall(url_regex, body)
#     return url
#
#
# def email_url_features(url_list, list_features, list_time):
#     if Globals.config["Email_Features"]["extract body features"] == "True":
#         Globals.logger.debug("Extracting email URL features")
#         # Features.Email_URL_Number_Url(url_All, list_features, list_time)
#         Features.Email_URL_Number_Diff_Domain(url_list, list_features, list_time)
#         Features.Email_URL_Number_link_at(url_list, list_features, list_time)
#         Features.Email_URL_Number_link_sec_port(url_list, list_features, list_time)
