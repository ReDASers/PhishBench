"""
This module handles the training of classifiers.

Built in classifiers can be found in the `classifiers` subpackage. Users can write custom classifiers by subclassing
the `BaseClassifier` class.
"""
from . import settings
from ._base_classifier import BaseClassifier
from ._core import train_classifiers, load_classifiers_from_module, load_classifiers
