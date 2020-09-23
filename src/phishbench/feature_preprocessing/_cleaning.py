"""
This module contains feature cleaning code
"""
from typing import List, Dict

from ..utils import phishbench_globals


def clean_features(feature_values: List[Dict]):
    """
    Cleans features for training
    Parameters
    ----------
    feature_values: List[Dict]
        The extracted features to clean
    """
    phishbench_globals.logger.debug('Cleaning')
    for feature_dict in feature_values:
        for key, value in feature_dict.items():
            if value in ["None", "N/A", "NaN", None]:
                feature_dict[key] = -1
                phishbench_globals.logger.debug("Value of %s changed from %s to -1", key, value)
    phishbench_globals.logger.info("Finished cleaning")