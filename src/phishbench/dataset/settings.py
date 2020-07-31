from ..utils import globals

DATASET_PATH_SECTION = 'Dataset Path'

DEFAULT_SETTINGS = {
    "path_legit_train": "dataset/legit_train",
    "path_phish_train": "dataset/phish_train",
    "path_legit_test": "dataset/legit_test",
    "path_phish_test": "dataset/phish_test",
}


def train_legit_path():
    return globals.config[DATASET_PATH_SECTION]['path_legit_train']


def train_phish_path():
    return globals.config[DATASET_PATH_SECTION]['path_phish_train']


def test_legit_path():
    return globals.config[DATASET_PATH_SECTION]['path_legit_test']


def test_phish_path():
    return globals.config[DATASET_PATH_SECTION]['path_phish_test']
