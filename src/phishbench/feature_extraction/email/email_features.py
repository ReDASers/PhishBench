"""
This module contains code for email feature extraction.
"""
import time
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm

from . import features as local_features
from ..reflection import FeatureClass, FeatureType, load_features
from ...input import email_input
from ...input.email_input.models import EmailMessage
from ...utils import phishbench_globals


def create_new_features() -> List[FeatureClass]:
    """
    Gets Email features

    Returns
    -------
    features:
        A list of instantiated features
    """
    features = [x() for x in load_features(local_features, 'Email')]
    print(f"Loaded {len(features)} features")
    return features


def extract_labeled_dataset(legit_dataset_folder: str, phish_dataset_folder: str,
                            features: Optional[List[FeatureClass]] = None):
    """
    Extracts features from a dataset of emails split in two folders

    Parameters
    ----------
    legit_dataset_folder: str
        The folder containing legitimate emails
    phish_dataset_folder: str
        The folder containing phishing emails
    features: Optional[List[FeatureClass]]
        A list of feature objects or `None`. If `None`, then this function will load and instantiate new instances of
        the features

    Returns
    -------
    feature_values: List[Dict]
        The feature values in a list of dictionaries. Features are mapped `config_name` to value.
    labels: List[int]
        The class labels for the dataset
    features: List[FeatureClass]
        The feature instances used.
    """
    phishbench_globals.logger.info("Extracting email features. Legit: %s Phish: %s",
                                   legit_dataset_folder, phish_dataset_folder)

    legit_emails, _ = email_input.read_dataset_email(legit_dataset_folder)
    phish_emails, _ = email_input.read_dataset_email(phish_dataset_folder)

    emails = legit_emails + phish_emails
    labels = [0] * len(legit_emails) + [1] * len(phish_emails)

    print("Extracting features")
    if features is None:
        features = create_new_features()
        for feature in features:
            feature.fit(emails, labels)
    feature_values = extract_features_list(emails, features)

    return feature_values, labels, features


def extract_features_list(emails: List[EmailMessage], features: List[FeatureClass]):
    """
    Extracts features from a list of `EmailMessage` objects

    Parameters
    ----------
    emails: List[EmailMessage]
        The emails to extract features from
    features: List[FeatureClass]
        The features to extract

    Returns
    -------
    feature_list_dict: List[Dict[str]]
        A list of dicts containing the extracted features
    """
    if not isinstance(emails, List):
        raise TypeError("emails must be an EmailMessage object")

    if features is None:
        features = create_new_features()
    feature_list_dict = []

    for email_msg in tqdm(emails):
        feature_values, _ = extract_features_from_single(features, email_msg)
        feature_list_dict.append(feature_values)

    return feature_list_dict


def extract_features_from_single(features: List[FeatureClass], email_msg: EmailMessage) -> Tuple[Dict, Dict]:
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
    # pylint: disable=duplicate-code
    if not isinstance(email_msg, EmailMessage):
        raise TypeError("email_msg must be an EmailMessage object")

    dict_feature_values = {}
    dict_feature_times = {}

    for feature in features:
        result, ex_time = extract_single_feature_email(feature, email_msg)
        if isinstance(result, dict):
            temp_dict = {f"{feature.config_name}.{key}": value for key, value in result.items()}
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
    if not isinstance(email_msg, EmailMessage):
        raise TypeError("email_msg must be an EmailMessage object")

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
        phishbench_globals.logger.warning("Error extracting %s", feature.config_name, exc_info=True)
        feature_value = feature.default_value
    end = time.process_time()
    ex_time = end - start
    return feature_value, ex_time
