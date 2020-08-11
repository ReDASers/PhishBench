from ..utils import phishbench_globals

DATASET_PATH_SECTION = 'Dataset Path'

DEFAULT_SETTINGS = {
    "path_legit_train": "dataset/legit_train",
    "path_phish_train": "dataset/phish_train",
    "path_legit_test": "dataset/legit_test",
    "path_phish_test": "dataset/phish_test",
    "path_modified_legit_text": "dataset/modified_legit_text",
    "path_modified_phish_text": "dataset/modified_phish_text",
}


def train_legit_path():
    return phishbench_globals.config[DATASET_PATH_SECTION]['path_legit_train']


def train_phish_path():
    return phishbench_globals.config[DATASET_PATH_SECTION]['path_phish_train']


def test_legit_path():
    return phishbench_globals.config[DATASET_PATH_SECTION]['path_legit_test']


def test_phish_path():
    return phishbench_globals.config[DATASET_PATH_SECTION]['path_phish_test']


def legit_path_adv():
    return phishbench_globals.config[DATASET_PATH_SECTION]['path_modified_legit_text']


def phish_path_adv():
    return phishbench_globals.config[DATASET_PATH_SECTION]['path_modified_phish_text']
