from typing import List
from ..utils import Globals
import os

CLASSIFICATION_SECTION = 'Classification'


class BaseClassifier:

    def __init__(self, io_dir):
        self.io_dir = io_dir

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def predict_proba(self, x):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass


def load_internal_classifiers():
    from . import classifiers
    return load_classifiers(classifiers)


def load_classifiers(source) -> List[type]:
    attrs = [getattr(source, x) for x in dir(source)]
    attrs = [x for x in attrs if isinstance(x, type) and issubclass(x, BaseClassifier)]
    return attrs


def train_classifiers(x_train, y_train, io_dir):
    if not os.path.isdir(io_dir):
        os.makedirs(io_dir)
    classifiers: List[type] = load_internal_classifiers()
    classifiers: List[BaseClassifier] = [x(io_dir) for x in classifiers]
    for classifier in classifiers:
        if Globals.config[CLASSIFICATION_SECTION].getboolean('load models'):
            classifier.load_model()
        else:
            classifier.fit(x_train, y_train)
        if Globals.config[CLASSIFICATION_SECTION].getboolean('save models'):
            classifier.save_model()
    return classifiers
