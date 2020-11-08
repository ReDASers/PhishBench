"""
Settings for the classification module
"""
from ..utils import phishbench_globals

SECTION = 'Classification'
CLASSIFIERS_SECTION = 'Classifiers'

DEFAULT_SETTINGS = {
    "param search": "True",
    # "Rounds": "1",
    "load models": "False",
    "save models": "True",
    "weighted": "True"
}


def load_models() -> bool:
    """
    Whether or not to load the models from disk
    """
    return phishbench_globals.config[SECTION].getboolean("load models")


def save_models() -> bool:
    """
    Whether or not to save the models to disk
    """
    return phishbench_globals.config[SECTION].getboolean("save models")


def param_search() -> bool:
    """
    Whether or not to perform parameter search
    """
    return phishbench_globals.config[SECTION].getboolean("param search")


def weighted_training() -> bool:
    """
    Whether or not to perform weighted training
    """
    return phishbench_globals.config[SECTION].getboolean("weighted")


def is_enabled(classifier: type) -> bool:
    """
    Whether or not a classifier is enabled
    Parameters
    ----------
    classifier:
        The classifier class to check

    Returns
    -------
        Whether or not the classifier is enabled
    """
    return phishbench_globals.config[CLASSIFIERS_SECTION].getboolean(classifier.__name__)

# def num_rounds() -> int:
#     return int(Globals.config[CLASSIFICATION_SECTION]["Rounds"])
