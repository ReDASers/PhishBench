"""
Contains feature vectorization code
"""
from typing import List, Dict, Iterable

from sklearn.feature_extraction import DictVectorizer
import numpy as np

def _prod(vals: Iterable):
    """
    Returns the product of all values in an iterable
    Parameters
    ----------
    vals: Iterable
        The iterable to reduce
    Returns
    -------
        The product of all values in `vals`
    """
    result = 1
    for x in vals:
        result *= x
    return result


def split_dict(raw: Dict):
    """
    Splits a dictionary of features into scalar and vector features. A feature is a vector feature if it has a shape
    whose product is greater than 1
    Parameters
    ----------
    raw

    Returns
    -------

    """
    scalar_features = dict()
    vector_features = dict()
    for key, value in raw.items():
        if hasattr(value, 'shape'):
            size = _prod(value.shape)
            value = np.reshape(size)
        elif hasattr(value, '__len__'):
            size = len(value)
        else:
            size = 1

        if size == 1:
            scalar_features[key] = value
        else:
            vector_features[key] = value
    return scalar_features, vector_features


class Vectorizer:

    def __init__(self):
        self.vectorizer = DictVectorizer()
        self.array_features = dict()

    def fit_transform(self, features: List[Dict]):
        split = map(split_dict, features)
        scalar_features = [x[0] for x in split]
        vector_features = [x[1] for x in split]
        x = self.vectorizer.fit_transform(scalar_features) 
