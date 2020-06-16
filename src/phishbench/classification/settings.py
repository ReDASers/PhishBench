from ..utils import Globals

CLASSIFICATION_SECTION = 'Classification'

DEFAULT_SETTINGS = {
    "Run Classifiers": "True",
    "Rounds": "1",
    "load models": "False",
    "save models": "True"
}


def get_num_rounds() -> int:
    return int(Globals.config[CLASSIFICATION_SECTION]["Rounds"])