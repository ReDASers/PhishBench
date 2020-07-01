import importlib
import inspect
from enum import Enum, unique
from functools import wraps
from typing import List, Callable

from phishbench.utils import Globals


@unique
class FeatureType(Enum):
    EMAIL_BODY = "Email_Body_Features"
    HEADER = "Email_Header_Features"


def register_feature(feature_type: FeatureType, config_name: str):
    def wrapped(fn):
        @wraps(fn)
        def wrapped_f(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapped_f.config_name = config_name
        wrapped_f.feature_type = feature_type
        return wrapped_f

    return wrapped


def load_internal_features() -> List[Callable]:
    from . import features
    return load_features(features)


def _check_feature(feature: Callable) -> bool:
    feature_type: FeatureType = feature.feature_type
    if feature_type.value not in Globals.config:
        return False
    return Globals.config[feature_type.value].getboolean(feature.config_name)


def load_features(source) -> List[Callable]:
    """
    Loads the PhishBench features from a module

    :param source: The module or name of the module to load the features from
    :feature_type module_name: str

    :return: A list containing all the features in the module
    """
    print(type(source))
    if isinstance(source, str):
        features_module = importlib.import_module(source)
    elif inspect.ismodule(source):
        features_module = source
    else:
        raise ValueError("source must be a module or string")
    # loads all features from module
    features = [getattr(features_module, x) for x in dir(features_module)]
    features = [x for x in features if hasattr(x, 'feature_type') and hasattr(x, 'config_name')]

    # remove disabled features
    for feature_type in FeatureType:
        if not Globals.config['Feature Types'].getboolean(feature_type.value):
            features = [f for f in features if f.feature_type != feature_type]
    features = [f for f in features if _check_feature(f)]
    print(features)
    return features



