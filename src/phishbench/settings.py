"""
This module contains core settings for phishbench
"""
from .utils import phishbench_globals

PB_SECTION = 'PhishBench'


DEFAULT_SETTINGS = {
    'Mode': 'URL ; Options are "URL" or "Email"',
    'feature extraction': 'True',
    'feature selection': 'True',
    'classification': 'True'
}


def mode() -> str:
    """
    Gets the mode PhishBench is in
    """
    mode_str = phishbench_globals.config[PB_SECTION].get('Mode').strip()
    if mode_str.lower().startswith('url'):
        return 'URL'
    if mode_str.lower().startswith('email'):
        return 'Email'
    raise ValueError('Mode must either be email or url')


def feature_extraction() -> bool:
    """
    Whether or not to extract features
    """
    return phishbench_globals.config[PB_SECTION].getboolean('feature extraction')


def feature_selection() -> bool:
    """
    Whether or not to perform feature selection
    """
    return phishbench_globals.config[PB_SECTION].getboolean('feature selection')


def classification() -> bool:
    """
    Whether or not to run classifiers
    """
    return phishbench_globals.config[PB_SECTION].getboolean('classification')
