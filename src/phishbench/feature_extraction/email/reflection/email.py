import time
from typing import List, Dict, Callable, Union

from phishbench.input.email_input.models import EmailMessage
from phishbench.utils import Globals
from .core import FeatureType


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
