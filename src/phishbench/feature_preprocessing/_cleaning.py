"""
This module contains feature cleaning code
"""
from typing import Iterable, Dict

import numpy as np
import scipy.sparse
from tqdm import tqdm

from ..utils import phishbench_globals

NONES = {"None", "N/A", "NaN", None}


def clean_features(feature_values: Iterable[Dict]):
    """
    Cleans features for training, replacing all `None` values with `-1`
    Parameters
    ----------
    feature_values: List[Dict]
        The extracted features to clean
    """
    phishbench_globals.logger.debug('Cleaning')
    for feature_dict in tqdm(feature_values):
        for key, value in feature_dict.items():
            if isinstance(value, np.ndarray) or scipy.sparse.issparse(value):
                continue
            if value in NONES:
                feature_dict[key] = -1
                phishbench_globals.logger.debug("Value of %s changed from %s to -1", key, value)
    phishbench_globals.logger.info("Finished cleaning")
