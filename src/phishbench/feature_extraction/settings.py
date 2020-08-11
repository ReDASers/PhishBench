from phishbench.feature_extraction.reflection import FeatureType
from phishbench.utils import phishbench_globals

EMAIL_TYPE_SECTION = 'Email_Feature_Types'

EMAIL_TYPE_SETTINGS = {
    feature_type.value: "True" for feature_type in FeatureType if feature_type.value.startswith('Email')
}


def feature_type_enabled(feature_type: FeatureType) -> bool:
    return phishbench_globals.config[EMAIL_TYPE_SECTION].getboolean(feature_type.value)


def extract_body_enabled():
    return feature_type_enabled(FeatureType.EMAIL_BODY)


def extract_header_enabled():
    return feature_type_enabled(FeatureType.EMAIL_HEADER)
