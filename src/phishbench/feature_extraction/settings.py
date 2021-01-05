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


def feature_enabled(feature_type: FeatureType, feature_name: str) -> bool:
    """
    Whether or not a feature is enabled

    Parameters
    ----------
    feature_type: FeatureType
        The feature type to check
    feature_name: str
        The feature type to check
    """
    return feature_type_enabled(feature_type) and phishbench_globals.config[feature_type.value].getboolean(feature_name)


def download_url_flag() -> bool:
    """
    Whether or not PhishBench needs to download urls
    """
    return feature_type_enabled(FeatureType.URL_NETWORK) or feature_type_enabled(FeatureType.URL_WEBSITE)
