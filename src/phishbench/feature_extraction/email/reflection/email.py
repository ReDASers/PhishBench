import time
from typing import List, Dict, Callable, Union

from phishbench.input.email_input.models import EmailMessage
from phishbench.utils import Globals
from .core import FeatureType, load_internal_features


def extract_single_feature_email(feature: Callable, email_msg: EmailMessage):
    Globals.logger.debug(feature.config_name)
    start = time.process_time()
    try:
        if feature.feature_type == FeatureType.EMAIL_BODY:
            feature_value = feature(email_msg.body)
        elif email_msg.header is not None:
            feature_value = feature(email_msg.header)
        else:
            raise ValueError('Email Message must have a header!')
    except Exception as e:
        print("exception: " + str(e))
        Globals.logger.error("%s with traceback", e, str(e.__traceback__).replace("\n", " "))
        feature_value = -1
    end = time.process_time()
    ex_time = end - start
    return feature_value, ex_time


def extract_features_from_single_email(features: List[Callable], email_msg: EmailMessage) -> Union[Dict, Dict]:
    dict_feature_values = dict()
    dict_feature_times = dict()

    for feature in features:
        result, ex_time = extract_single_feature_email(feature, email_msg)
        if isinstance(result, Dict):
            temp_dict = {feature.config_name + "." + key: value for key, value in result.items()}
            dict_feature_values.update(temp_dict)
        else:
            dict_feature_values[feature.config_name] = result
        dict_feature_times[feature.config_name] = ex_time

    return dict_feature_values, dict_feature_times


def extract_features_emails(emails: List[EmailMessage]) -> Union[List[Dict], List[Dict]]:
    """
    Runs feature extraction on all emails in a folder

    Parameters
    ----------
    emails: List[EmailMessage] List of emails to extract features from

    Returns
    -------
    feature_dict_list:
        List[Dict] List of features
    time_dict_list:
        List[Dict] List of times for feature extraction
    """
    feature_dict_list = list()
    time_dict_list = list()

    features = load_internal_features()

    for email_msg in emails:
        feature_values, feature_times = extract_features_from_single_email(features, email_msg)
        feature_dict_list.append(feature_values)
        time_dict_list.append(feature_times)

    return feature_dict_list, time_dict_list
