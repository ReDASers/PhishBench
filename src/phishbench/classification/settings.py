from ..utils import Globals

CLASSIFICATION_SECTION = 'Classification'
CLASSIFIERS_SECTION = 'Classifiers'

DEFAULT_SETTINGS = {
    "Run Classifiers": "True",
    # "Rounds": "1",
    "load models": "False",
    "save models": "True",
    "weighted": "True"
}


def load_models():
    return Globals.config[CLASSIFICATION_SECTION].getboolean("load models")


def save_models():
    return Globals.config[CLASSIFICATION_SECTION].getboolean("save models")


def run_classifiers():
    return Globals.config[CLASSIFICATION_SECTION].getboolean("Run Classifiers")


def weighted_training():
    return Globals.config[CLASSIFICATION_SECTION].getboolean("weighted")


def is_enabled(classifier: type):
    return Globals.config[CLASSIFIERS_SECTION].getboolean(classifier.__name__)

# def num_rounds() -> int:
#     return int(Globals.config[CLASSIFICATION_SECTION]["Rounds"])
