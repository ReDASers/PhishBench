import inspect
from typing import List, Callable

from . import settings
from ..reflection import FeatureType
from ...utils import phishbench_globals


def _check_feature(feature: Callable) -> bool:
    feature_type: FeatureType = feature.feature_type
    if feature_type.value not in phishbench_globals.config:
        return False
    return phishbench_globals.config[feature_type.value].getboolean(feature.config_name)


def load_features_from_module(features_module, filter_features=True) -> List[Callable]:
    """
    Loads features from a module
    Parameters
    ----------
    features_module: ModuleType
        The module to import features from
    filter_features: bool
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
        enabled_types = [feat_type for feat_type in FeatureType if settings.feature_type_enabled(feat_type)]
        features = [f for f in features if f.feature_type in enabled_types and _check_feature(f)]

    return features
