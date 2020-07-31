from ..utils import globals

CLASSIFICATION_SECTION = 'Classification'
CLASSIFIERS_SECTION = 'Classifiers'

DEFAULT_SETTINGS = {
    "Run Classifiers": "True",
    "param search": "True",
    # "Rounds": "1",
    "load models": "False",
    "save models": "True",
    "weighted": "True"
}


def load_models():
    return globals.config[CLASSIFICATION_SECTION].getboolean("load models")


def save_models():
    return globals.config[CLASSIFICATION_SECTION].getboolean("save models")


def run_classifiers():
    return globals.config[CLASSIFICATION_SECTION].getboolean("Run Classifiers")


def param_search():
    return globals.config[CLASSIFICATION_SECTION].getboolean("param search")


def weighted_training():
    return globals.config[CLASSIFICATION_SECTION].getboolean("weighted")


def is_enabled(classifier: type):
    return globals.config[CLASSIFIERS_SECTION].getboolean(classifier.__name__)

# def num_rounds() -> int:
#     return int(Globals.config[CLASSIFICATION_SECTION]["Rounds"])
