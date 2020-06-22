from ..utils import Globals

CLASSIFICATION_SECTION = 'Classification'

DEFAULT_SETTINGS = {
    "Run Classifiers": "True",
    # "Rounds": "1",
    "load models": "False",
    "save models": "True"
}


def load_models():
    return Globals.config[CLASSIFICATION_SECTION].get("load models")


def save_models():
    return Globals.config[CLASSIFICATION_SECTION].get("save models")


def run_classifiers():
    return Globals.config[CLASSIFICATION_SECTION].getboolean("Run Classifiers")

# def num_rounds() -> int:
#     return int(Globals.config[CLASSIFICATION_SECTION]["Rounds"])
