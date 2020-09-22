"""
This module contains code for email feature extraction.
"""
import time
from typing import List, Dict, Tuple

from tqdm import tqdm

from . import features as local_features
from ..reflection import FeatureClass, FeatureType, load_features
from ...input import email_input
from ...input.email_input.models import EmailMessage
from ...utils import phishbench_globals


def extract_labeled_dataset(legit_dataset_folder: str, phish_dataset_folder: str):
    """
    Extracts features from a dataset of emails split in two folders

    Parameters
    ----------
    legit_dataset_folder: str
        The folder containing legitimate emails
    phish_dataset_folder
        The folder containing phishing emails

    Returns
    -------
    feature_values: List[Dict]]
        The feature values in a list of dictionaries. Features are mapped `config_name` to value.
    labels: List[int]
        The class labels for the dataset
    corpus List[str]:
        The email bodies corresponding to each feature set
    """

    features = [feature() for feature in load_features(local_features, 'Email')]
    print("Loaded {} features".format(len(features)))

    phishbench_globals.logger.info("Extracting email features. Legit: %s Phish: %s",
                                   legit_dataset_folder, phish_dataset_folder)

    print("Loading emails from {}".format(legit_dataset_folder))
    legit_emails, _ = email_input.read_dataset_email(legit_dataset_folder)
    print("Loading emails from {}".format(phish_dataset_folder))
    phish_emails, _ = email_input.read_dataset_email(phish_dataset_folder)

    emails = legit_emails + phish_emails
    labels = [0] * len(legit_emails) + [1] * len(phish_emails)

    print("Extracting features")
    feature_values, corpus = extract_features_list_email(emails, features)

    return feature_values, labels, corpus


def extract_features_list_email(emails: List[EmailMessage], features: List[FeatureClass]):
    """
    Extracts features from a list of `EmailMessage` objects

    Parameters
    ----------
    emails: List[EmailMessage]
        The emails to extract features from
    features: List
        The features to extract

    Returns
    -------
    feature_list_dict: List[Dict[str]]
        A list of dicts containing the extracted features
    corpus:
        A list of the email texts
    """
    feature_list_dict = list()

    for email_msg in tqdm(emails):
        feature_values, _ = extract_features_from_single_email(features, email_msg)
        feature_list_dict.append(feature_values)

    corpus = [msg.body.text for msg in emails]

    return feature_list_dict, corpus


def extract_features_from_single_email(features: List[FeatureClass], email_msg: EmailMessage) -> Tuple[Dict, Dict]:
    """
    Extracts multiple features from a single email

    Parameters
    ----------
    features: List
        The features to extract
    email_msg: EmailMessage
        The email to extract the features from

    Returns
    -------
    feature_values: Dict
        The extracted feature values
    extraction_times: Dict
        The time it took to extract each feature
    """
    dict_feature_values = dict()
    dict_feature_times = dict()

    for feature in features:
        result, ex_time = extract_single_feature_email(feature, email_msg)
        if isinstance(result, dict):
            temp_dict = {feature.config_name + "." + key: value for key, value in result.items()}
            dict_feature_values.update(temp_dict)
        else:
            dict_feature_values[feature.config_name] = result
        dict_feature_times[feature.config_name] = ex_time

    return dict_feature_values, dict_feature_times


def extract_single_feature_email(feature: FeatureClass, email_msg: EmailMessage):
    """
    Extracts a single feature from a single email

    Parameters
    ----------
    feature: FeatureClass
        The feature to extract
    email_msg: EmailMessage
        The email to extract the feature from

    Returns
    -------
    feature_value
        The value of the feature
    ex_time: float
        The time to extract the feature
    """
    # pylint: disable=broad-except
    phishbench_globals.logger.debug(feature.config_name)
    start = time.process_time()
    try:
        if feature.feature_type == FeatureType.EMAIL_BODY:
            feature_value = feature.extract(email_msg.body)
        elif email_msg.header is not None:
            feature_value = feature.extract(email_msg.header)
        else:
            raise ValueError('Email Message must have a header!')
    except Exception:
        error_string = "Error extracting {}".format(feature.config_name)
        phishbench_globals.logger.warning(error_string, exc_info=True)
        feature_value = -1
    end = time.process_time()
    ex_time = end - start
    return feature_value, ex_time

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
