"""
Contains settings for the input module
"""
from ..utils import phishbench_globals

DATASET_PATH_SECTION = 'Dataset Path'

DEFAULT_SETTINGS = {
    "path_legit_train": "dataset/legit_train",
    "path_phish_train": "dataset/phish_train",
    "path_legit_test": "dataset/legit_test",
    "path_phish_test": "dataset/phish_test",
}


def train_legit_path() -> str:
    """
    Gets the path of the legitimate training samples as a string
    """
    return phishbench_globals.config[DATASET_PATH_SECTION]['path_legit_train']


def train_phish_path():
    """
    Gets the path of the phishing training samples as a string
    """
    return phishbench_globals.config[DATASET_PATH_SECTION]['path_phish_train']


def test_legit_path():
    """
    Gets the path of the legitimate testing samples as a string
    """
    return phishbench_globals.config[DATASET_PATH_SECTION]['path_legit_test']


def test_phish_path():
    """
    Gets the path of the phishing testing samples as a string
    """
    return phishbench_globals.config[DATASET_PATH_SECTION]['path_phish_test']
