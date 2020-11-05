"""
Settings for the sampling class
"""

from ._methods import METHODS
from ...utils import phishbench_globals

SAMPLING_SECTION = "Dataset Balancing"

DEFAULT_SAMPLING_SETTINGS = {
    name: "True" for name in METHODS
}


def method_enabled(method: str) -> bool:
    """
    Whether or not a method is enabled

    Parameters
    ----------
    method: str
        The name of the method to check
    """
    if method not in phishbench_globals.config[SAMPLING_SECTION]:
        return False
    return phishbench_globals.config[SAMPLING_SECTION].getboolean(method)