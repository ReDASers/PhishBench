"""
Contains feature vectorization code
"""
from collections import OrderedDict
from typing import List, Dict

import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction import DictVectorizer


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
            if len(value.shape) == 0:
                size = 1
            elif len(value.shape) == 1:
                size = value.shape[0]
            else:
                size = value.shape[1]
        else:
            size = 1

        if size == 1:
            scalar_features[key] = value
        else:
            vector_features[key] = value
    return scalar_features, vector_features


class Vectorizer:
    """
    A custom vectorizer for Phishbench features.
    This class supports scalar, numpy array, and scipy sparse array feature values.

    Attributes
    ----------
    scalar_vectorizer: DictVectorizer
        The vectorizer used for scalar features
    array_feature_indicies:
        The feature indecies for the vector features
    """

    def __init__(self):
        """
        Constructs a new Vectorizer
        """
        self.scalar_vectorizer = DictVectorizer()
        self.array_feature_indicies = OrderedDict()

    def fit_transform(self, features: List[Dict]):
        """
        Fits a training set to this dict Vectorizer and returns the vectorized features
        Parameters
        ----------
        features: List[Dict]
            The raw feature values
        Returns
        -------
        A `scipy.sparse` matrix of vectorized features
        """
        split = list(map(split_dict, features))
        scalar_features = [x[0] for x in split]
        vector_features = [x[1] for x in split]
        x = self.scalar_vectorizer.fit_transform(scalar_features)
        if len(vector_features) == 0:
            return x
        for key in vector_features[0].keys():
            values = [features[key] for features in vector_features]
            if isinstance(values[0], np.ndarray):
                x_key = np.array(values)
            else:
                # is scipy sparse
                x_key = vstack(values)
            self.array_feature_indicies[key] = (x.shape[1], x.shape[1] + x_key.shape[1])
            x = hstack([x, x_key])
        return x

    def transform(self, features: List[Dict]):
        """
        Transforms a dataset of features
        Parameters
        ----------
        features: List[Dict]
            The raw feature values
        Returns
        -------
        A `scipy.sparse` matrix of vectorized features
        """
        split = list(map(split_dict, features))
        scalar_features = [x[0] for x in split]
        vector_features = [x[1] for x in split]
        x = self.scalar_vectorizer.fit_transform(scalar_features)
        if len(self.array_feature_indicies) == 0:
            return x
        for key in self.array_feature_indicies.keys():
            values = [features[key] for features in vector_features]
            if isinstance(values[0], np.ndarray):
                x_key = np.array(values)
            else:
                # is scipy sparse
                x_key = vstack(values)
            self.array_feature_indicies[key] = (x.shape[1], x.shape[1] + x_key.shape[1])
            x = hstack([x, x_key])
        return x
