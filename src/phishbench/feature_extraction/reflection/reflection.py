"""
This module contains the core reflection functions for feature extraction.
"""
import inspect
import itertools
from types import ModuleType
from typing import List, Callable

from .reflect import FeatureType
from .. import settings
from ...utils import phishbench_globals
from ...utils.reflection_utils import load_local_modules


def _check_feature(feature: Callable) -> bool:
    """
    Checks whether or not a feature is enabled.
    Parameters
    ----------
    feature: Callable
        The feature to check

    Returns
    -------
        `True` if the feature is enabled. `False` otherwise.
    """
    feature_type: FeatureType = feature.feature_type
    if feature_type.value not in phishbench_globals.config:
        return False
    return phishbench_globals.config[feature_type.value].getboolean(feature.config_name)


def load_features(internal_features=None, filter_features=None) -> List[Callable]:
    """
    Loads all features

    Parameters
    ----------
    internal_features: Union[ModuleType, List]
        The module or a list of modules to load internal features from
    filter_features: Union[str, None]
        Whether or not to filter the features
    Returns
    -------
        A list of feature functions
    """
    modules = load_local_modules()
    if internal_features:
        if isinstance(internal_features, ModuleType):
            modules.append(internal_features)
        else:
            modules.extend(internal_features)
    loaded_features = [load_features_from_module(module, filter_features) for module in modules]
    features = list(itertools.chain.from_iterable(loaded_features))
    return features


def load_features_from_module(features_module, filter_features=None) -> List[Callable]:
    """
    Loads features from a module
    Parameters
    ----------
    features_module: ModuleType
        The module to import features from
    filter_features: Union[str, None]
        Whether or not to load features based on `phishbench.utils.phishbench_globals.config`

    Returns
    -------
    A list of features in the module.
    """
    if not inspect.ismodule(features_module):
        raise ValueError("source must be a module or string")
    # loads all features from module
    features = [getattr(features_module, x) for x in dir(features_module)]
    features = [x for x in features if hasattr(x, 'feature_type') and hasattr(x, 'config_name')]

    if filter_features:
        if filter_features not in ('Email', 'URL'):
            raise ValueError('filter_features must either be "Email" or "URL"')
        enabled_types = [feat_type for feat_type in FeatureType if settings.feature_type_enabled(feat_type)]
        enabled_types = [feat_type for feat_type in enabled_types if feat_type.value.startswith(filter_features)]
        features = [f for f in features if f.feature_type in enabled_types and _check_feature(f)]

    return features
