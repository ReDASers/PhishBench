"""
Settings for the `feature_preprocessing.feature_selection` module.
"""
from ._methods import METHODS

from ...utils import phishbench_globals

FEATURE_SELECTION_SECTION = "Feature Selection"
DEFAULT_FEATURE_SELECTION_SETTINGS = {
    "number of best features": "80",
    "with Tfidf": "True"
}

SELECTION_METHODS_SECTION = "Feature Selection Methods"
DEFAULT_METHODS_SETTINGS = {
    name: "True" for name in METHODS
}


def num_features() -> int:
    """
    The number of top features to select
    """
    return phishbench_globals.config[FEATURE_SELECTION_SECTION].getint("number of best features")


def with_tfidf() -> bool:
    """
    Whether or not to include tfidf features
    """
    return phishbench_globals.config[FEATURE_SELECTION_SECTION].getboolean("with Tfidf")


def method_enabled(method: str) -> bool:
    """
    Whether or not a method is enabled

    Parameters
    ----------
    method: str
        The name of the method to check
    """
    if method not in phishbench_globals.config[SELECTION_METHODS_SECTION]:
        return False
    return phishbench_globals.config[SELECTION_METHODS_SECTION].getboolean(method)
