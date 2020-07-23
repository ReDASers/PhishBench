from .reflection.core import FeatureType
from ...utils import Globals

FEATURE_TYPE_SECTION = 'Email_Feature_Types'

FEATURE_TYPE_SETTINGS = {
    feature_type.value: "True" for feature_type in FeatureType
}


def feature_type_enabled(feature_type: FeatureType) -> bool:
    return Globals.config[FEATURE_TYPE_SECTION].getboolean(feature_type.value)


def extract_body_enabled():
    return feature_type_enabled(FeatureType.EMAIL_BODY)


def extract_header_enabled():
    return feature_type_enabled(FeatureType.HEADER)
