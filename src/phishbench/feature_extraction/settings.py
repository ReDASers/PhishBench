"""
Settings for feature extraction
"""
from .reflection import FeatureType
from ..utils import phishbench_globals

EMAIL_TYPE_SECTION = 'Email_Feature_Types'

EMAIL_TYPE_SETTINGS = {
    feature_type.value: "True" for feature_type in FeatureType if feature_type.value.startswith('Email')
}

URL_TYPE_SECTION = 'URL_Feature_Types'

URL_TYPE_SETTINGS = {
    feature_type.value: "True" for feature_type in FeatureType if feature_type.value.startswith('URL')
}


def feature_type_enabled(feature_type: FeatureType) -> bool:
    """
    Whether or not a feature type is enabled

    Parameters
    ----------
    feature_type: FeatureType
        The feature type to check
    """
    if feature_type.value.startswith("Email"):
        return phishbench_globals.config[EMAIL_TYPE_SECTION].getboolean(feature_type.value)
    return phishbench_globals.config[URL_TYPE_SECTION].getboolean(feature_type.value)


def extract_body_enabled():
    """
    Whether or not to extract email body features
    """
    return feature_type_enabled(FeatureType.EMAIL_BODY)


def extract_header_enabled() -> bool:
    """
    Whether or not to extract header features
    """
    return feature_type_enabled(FeatureType.EMAIL_HEADER)


def download_url_flag() -> bool:
    """
    Whether or not PhishBench needs to download urls
    """
    return feature_type_enabled(FeatureType.URL_NETWORK) or feature_type_enabled(FeatureType.URL_WEBSITE)
