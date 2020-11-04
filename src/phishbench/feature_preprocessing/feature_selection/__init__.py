"""
This module contains code for feature selection
"""
from ._feature_selection import run_feature_extraction
from ._methods import METHODS
from ._methods import chi_squared
from ._methods import gini
from ._methods import information_gain
from ._methods import rfe