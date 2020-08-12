from enum import Enum, unique
from functools import wraps


@unique
class FeatureType(Enum):
    EMAIL_BODY = 'Email_Body_Features'
    EMAIL_HEADER = 'Email_Header_Features'
    URL_RAW = 'URL_Features'
    URL_NETWORK = 'URL_Network_Features'
    URL_WEBSITE = 'URL_HTML_Features'
    URL_WEBSITE_JAVASCRIPT = 'URL_Javascript_Features'


def register_feature(feature_type: FeatureType, config_name: str):
    """
    Registers a feature for use in Phishbench
    Parameters
    ----------
    feature_type: FeatureType
        The type of feature
    config_name
        The name of the feature in the config file

    """

    def wrapped(feature_function):
        @wraps(feature_function)
        def wrapped_f(*args, **kwargs):
            return feature_function(*args, **kwargs)

        wrapped_f.config_name = config_name
        wrapped_f.feature_type = feature_type
        return wrapped_f

    return wrapped
