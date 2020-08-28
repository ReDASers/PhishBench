from enum import Enum, unique


@unique
class FeatureType(Enum):
    EMAIL_BODY = 'Email_Body_Features'
    EMAIL_HEADER = 'Email_Header_Features'
    URL_RAW = 'URL_Features'
    URL_NETWORK = 'URL_Network_Features'
    URL_WEBSITE = 'URL_HTML_Features'
    URL_WEBSITE_JAVASCRIPT = 'URL_Javascript_Features'


class FeatureMC(type):
    """
    The metaclass for Feature Classes
    """
    def __new__(cls, name, bases, attrs):
        if not isinstance(attrs['feature_type'], FeatureType):
            raise SyntaxError("feature_type must have be an instance of FeatureType")
        if not isinstance(attrs['config_name'], str):
            raise SyntaxError("config_name must have be a string")

        if not attrs['fit'].__code__.co_argcount == 3:
            raise SyntaxError("fit must have signature fit(self, corpus, label)")
        if not attrs['extract'].__code__.co_argcount == 2:
            raise SyntaxError("extract must have signature fit(self, x)")

        x = type.__new__(cls, name, bases, attrs)
        x.config_name = attrs['config_name']
        x.feature_type = attrs['feature_type']
        return x


def _do_nothing(self, x, y):
    # pylint: disable=unused-argument
    pass


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
    def register_feature_decorator(feature_function):
        def extract(self, x):
            # pylint: disable=unused-argument
            return feature_function(x)
        attrs = dict()
        attrs['config_name'] = config_name
        attrs['feature_type'] = feature_type
        attrs['extract'] = extract
        attrs['fit'] = _do_nothing
        feature_class = FeatureMC(config_name, (), attrs)
        return feature_class

    return register_feature_decorator
