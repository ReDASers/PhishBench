"""
Models for feature reflection
"""
from typing import Any, Callable
from enum import Enum, unique


@unique
class FeatureType(Enum):
    """
    The types of features that can be extracted
    """
    EMAIL_BODY = 'Email_Body_Features'
    EMAIL_HEADER = 'Email_Header_Features'
    URL_RAW = 'URL_Features'
    URL_NETWORK = 'URL_Network_Features'
    URL_WEBSITE = 'URL_HTML_Features'
    URL_WEBSITE_JAVASCRIPT = 'URL_Javascript_Features'


class FeatureClass:
    """
    A type hint stub for features
    """
    config_name: str
    feature_type: FeatureType
    default_value: Any

    def __init__(self):
        raise SyntaxError('This is a stub for type hinting and should not be instantiated.')

    def fit(self, corpus, labels):
        """
        Fits the feature extractor onto the corpus

        Parameters
        ----------
        corpus: List
            The dataset corpus
        labels: List
            The labels for the dataset
        """

    def extract(self, x):
        """
        Extracts the feature

        Parameters
        ----------
        x:
            The data point to extract the feature from

        Returns
        -------
            The feature value extracted from `x`
        """

    def load_state(self, filename):
        """
        Loads the feature state from a file
        """

    def save_state(self, filename):
        """
        Saves the feature state to a file
        """


class FeatureMC(type):
    """
    The metaclass for Feature Classes
    """
    def __new__(cls, name, bases, attrs):
        if 'default_value' not in attrs:
            raise SyntaxError("features must have a default value")
        if not isinstance(attrs['feature_type'], FeatureType):
            raise SyntaxError("feature_type must have be an instance of FeatureType")
        if not isinstance(attrs['config_name'], str):
            raise SyntaxError("config_name must have be a string")

        if attrs['fit'] is not _do_nothing and attrs['fit'].__code__.co_argcount != 3:
            raise SyntaxError("fit must have signature fit(self, corpus, label)")
        if not attrs['extract'].__code__.co_argcount == 2:
            raise SyntaxError("extract must have signature fit(self, x)")
        if attrs['save_state'] is not _do_nothing and attrs['save_state'].__code__.co_argcount != 2:
            raise SyntaxError("extract must have signature fit(self, x)")
        if attrs['load_state'] is not _do_nothing and attrs['load_state'].__code__.co_argcount != 2:
            raise SyntaxError("extract must have signature fit(self, x)")

        x = type.__new__(cls, name, bases, attrs)
        x.config_name = attrs['config_name']
        x.feature_type = attrs['feature_type']
        return x


def _do_nothing(self, *args):
    # pylint: disable=unused-argument
    pass


def register_feature(feature_type: FeatureType, config_name: str, default_value=-1):
    """
    Registers a feature for use in Phishbench

    Parameters
    ----------
    feature_type: FeatureType
        The type of feature
    config_name
        The name of the feature in the config file
    default_value
        The value to use if there is an error
    """
    def register_feature_decorator(feature_function: Callable):
        def extract(self, x):
            # pylint: disable=unused-argument
            return feature_function(x)
        attrs = {
            'config_name': config_name,
            'feature_type': feature_type,
            'extract': extract,
            'fit': _do_nothing,
            'save_state': _do_nothing,
            'load_state': _do_nothing,
            'default_value': default_value
        }
        feature_class = FeatureMC(config_name, (), attrs)
        feature_class.__doc__ = feature_function.__doc__
        return feature_class

    return register_feature_decorator
