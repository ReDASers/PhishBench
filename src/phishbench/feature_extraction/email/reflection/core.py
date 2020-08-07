import importlib
import inspect
from enum import Enum, unique
from functools import wraps
from typing import List, Callable

from .. import settings
from ....utils import phishbench_globals


@unique
class FeatureType(Enum):
    EMAIL_BODY = "Email_Body_Features"
    HEADER = "Email_Header_Features"


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
    def wrapped(fn):
        @wraps(fn)
        def wrapped_f(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapped_f.config_name = config_name
        wrapped_f.feature_type = feature_type
        return wrapped_f

    return wrapped


def _check_feature(feature: Callable) -> bool:
    feature_type: FeatureType = feature.feature_type
    if feature_type.value not in phishbench_globals.config:
        return False
    return phishbench_globals.config[feature_type.value].getboolean(feature.config_name)


def load_features(source, filter_features=True) -> List[Callable]:
    """
    Loads the PhishBench features from a module

    :param source: The module or name of the module to load the features from
    :param filter_features: Whether or not to filter the features based on Globals.config.

    :return: A list containing all the features in the module.
    """
    if isinstance(source, str):
        features_module = importlib.import_module(source)
    elif inspect.ismodule(source):
        features_module = source
    else:
        raise ValueError("source must be a module or string")

    # loads all features from module
    features = [getattr(features_module, x) for x in dir(features_module)]
    features = [x for x in features if hasattr(x, 'feature_type') and hasattr(x, 'config_name')]

    if filter_features:
        # remove disabled features
        for feature_type in FeatureType:
            if not settings.feature_type_enabled(feature_type):
                features = [f for f in features if f.feature_type != feature_type]
        features = [f for f in features if _check_feature(f)]
    return features
