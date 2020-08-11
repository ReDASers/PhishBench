from ..reflection import FeatureType
from ...utils import phishbench_globals

FEATURE_TYPE_SECTION = 'Email_Feature_Types'

FEATURE_TYPE_SETTINGS = {
    feature_type.value: "True" for feature_type in FeatureType
}


def feature_type_enabled(feature_type: FeatureType) -> bool:
    return phishbench_globals.config[FEATURE_TYPE_SECTION].getboolean(feature_type.value)


def extract_body_enabled():
    return feature_type_enabled(FeatureType.EMAIL_BODY)


def extract_header_enabled():
    return feature_type_enabled(FeatureType.EMAIL_HEADER)
